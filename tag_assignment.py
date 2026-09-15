"""
tag_assignment.py
==================
Beregner prototype-embeddings for tags og seed-eksempler fra PostgreSQL,
embedder alle job/app-dokumenter og -paragraffer, og
tildeler hver tekst-enhed de tags den ligner mest via cosine similarity —
adskilt pr. top-kategori (fx "motivation", "teknisk_kompetence", ...), så en
paragraf kan få ét eller flere gode tags PR. kategori i stedet for kun ét
globalt bedste tag.

Resultatet gemmes i Postgres-tabellen `text_tags` (target_type, target_id,
tag_key, similarity_score, rank_in_category), og bruges i praksis sådan:

  - Ved indeksering: kør dette script EFTER tekst- og paragraf-importen,
    så alle dokumenter/paragraffer allerede findes i Postgres.
  - Ved query-tid: embed den nye jobopslags-paragraf/tekst med samme model,
    sammenlign mod de samme tag-prototyper (se `assign_tags_to_text()`), og
    brug de tildelte tags til reranking/filtrering af FAISS-similarity-hits
    (fx: foretræk kandidater der deler tags med query-teksten).

Brug:
    python tag_assignment.py                     # tagger alt i DB'en
    python tag_assignment.py --min-score 0.35     # justér tærskel
    python tag_assignment.py --top-k 3            # max tags pr. kategori pr. tekst
"""

import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import psycopg2
import psycopg2.extras

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    pass

from sentence_transformers import SentenceTransformer

from OLD.build_FAISS_and_populate_DB import POSTGRESQL_DSN, EMBEDDING_MODEL_PATH


# Standard: hvor mange tags en tekst maksimalt kan få PR. top-kategori, og
# hvor høj cosine similarity der som minimum kræves for at et tag tildeles.
DEFAULT_TOP_K_PER_CATEGORY = 3
DEFAULT_MIN_SCORE = 0.35

CREATE_TAG_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS tags (
    id              SERIAL PRIMARY KEY,
    tag_key         TEXT NOT NULL UNIQUE,
    top_category    TEXT NOT NULL,
    taxonomy_version TEXT NOT NULL DEFAULT '1.0',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tag_seeds (
    id          SERIAL PRIMARY KEY,
    tag_key     TEXT NOT NULL REFERENCES tags(tag_key) ON DELETE CASCADE,
    seed_index  INTEGER NOT NULL,
    seed_text   TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now(),
    UNIQUE (tag_key, seed_index),
    UNIQUE (tag_key, seed_text)
);

CREATE TABLE IF NOT EXISTS text_tags (
    id                  SERIAL PRIMARY KEY,
    target_type         TEXT NOT NULL,
    target_id           TEXT NOT NULL,
    tag_key             TEXT NOT NULL REFERENCES tags(tag_key) ON DELETE CASCADE,
    similarity_score    REAL NOT NULL,
    rank_in_category    INTEGER NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE (target_type, target_id, tag_key)
);

CREATE INDEX IF NOT EXISTS idx_text_tags_target
    ON text_tags(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_text_tags_tag
    ON text_tags(tag_key);
CREATE INDEX IF NOT EXISTS idx_text_tags_score
    ON text_tags(target_type, tag_key, similarity_score DESC);
"""


# ── Taksonomi + prototype-embeddings ────────────────────────────────────────

def fetch_tags_from_db(conn) -> List[Tuple[str, str, List[str]]]:
    """Henter [(tag_key, top_category, seed_examples), ...] fra PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.tag_key, t.top_category, s.seed_index, s.seed_text
            FROM tags t
            LEFT JOIN tag_seeds s ON s.tag_key = t.tag_key
            ORDER BY t.top_category, t.tag_key, s.seed_index
            """
        )
        rows = cur.fetchall()

    tags: Dict[str, Tuple[str, List[str]]] = {}
    for tag_key, top_category, seed_index, seed_text in rows:
        if tag_key not in tags:
            tags[tag_key] = (top_category, [])
        if seed_text is not None:
            tags[tag_key][1].append(seed_text)

    flat_tags = [
        (tag_key, top_category, seeds)
        for tag_key, (top_category, seeds) in tags.items()
    ]
    missing_seeds = [tag_key for tag_key, _, seeds in flat_tags if not seeds]
    if missing_seeds:
        raise ValueError(
            "Følgende tags i databasen har ingen seed-eksempler i tag_seeds: "
            + ", ".join(missing_seeds)
        )
    if not flat_tags:
        raise ValueError("Databasen indeholder ingen tags i tabellen tags.")
    return flat_tags


def build_tag_prototypes(
    model: SentenceTransformer, flat_tags: List[Tuple[str, str, List[str]]]
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Beregner én prototype-vektor pr. tag = L2-normaliseret gennemsnit af dets
    seed-embeddings. Returnerer (tag_keys, top_categories, prototype_matrix).
    """
    tag_keys, top_categories, prototypes = [], [], []
    for tag_key, top_category, seeds in flat_tags:
        seed_embeddings = model.encode(seeds, convert_to_numpy=True).astype("float32")
        # Normalisér hvert seed FØR gennemsnit, så lange/korte seeds vægter ens.
        norms = np.linalg.norm(seed_embeddings, axis=1, keepdims=True)
        seed_embeddings = seed_embeddings / np.clip(norms, 1e-8, None)
        prototype = seed_embeddings.mean(axis=0)
        prototype = prototype / np.clip(np.linalg.norm(prototype), 1e-8, None)

        tag_keys.append(tag_key)
        top_categories.append(top_category)
        prototypes.append(prototype)

    return tag_keys, top_categories, np.vstack(prototypes).astype("float32")


def upsert_tags(conn, flat_tags: List[Tuple[str, str, List[str]]], taxonomy_version: str) -> None:
    with conn.cursor() as cur:
        rows = [(tag_key, top_category, taxonomy_version) for tag_key, top_category, _ in flat_tags]
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO tags (tag_key, top_category, taxonomy_version)
            VALUES %s
            ON CONFLICT (tag_key) DO UPDATE SET
                top_category = EXCLUDED.top_category,
                taxonomy_version = EXCLUDED.taxonomy_version
            """,
            rows,
        )
# ── Tekst-enheder der skal tagges ───────────────────────────────────────────

def fetch_targets(conn) -> List[Tuple[str, str, str]]:
    """
    Returnerer [(target_type, target_id, text), ...] for alle dokumenter og
    paragraffer der skal tagges. target_id = case_id for dokumenter,
    paragraph_id for paragraffer.
    """
    targets: List[Tuple[str, str, str]] = []
    with conn.cursor() as cur:
        cur.execute("SELECT case_id, raw_text FROM job_documents")
        targets += [("job_document", case_id, text) for case_id, text in cur.fetchall()]

        cur.execute("SELECT case_id, raw_text FROM app_documents")
        targets += [("app_document", case_id, text) for case_id, text in cur.fetchall()]

        cur.execute("SELECT paragraph_id, text FROM job_paragraphs")
        targets += [("job_paragraph", pid, text) for pid, text in cur.fetchall()]

        cur.execute("SELECT paragraph_id, text FROM app_paragraphs")
        targets += [("app_paragraph", pid, text) for pid, text in cur.fetchall()]

    return targets


# ── Similarity tagging ───────────────────────────────────────────────────────

def assign_tags_to_embeddings(
    text_embeddings: np.ndarray,      # (N, dim), L2-normaliseret
    tag_keys: List[str],
    top_categories: List[str],
    prototypes: np.ndarray,           # (num_tags, dim), L2-normaliseret
    top_k_per_category: int,
    min_score: float,
) -> List[List[Dict[str, Any]]]:
    """
    For hver tekst-embedding: beregn cosine similarity mod alle tag-prototyper,
    gruppér pr. top_category, behold top_k bedste pr. kategori der er over
    min_score. Returnerer liste (én pr. tekst) af lister af
    {"tag_key", "top_category", "score", "rank_in_category"}.
    """
    # (N, num_tags) similarity matrix — begge sider allerede normaliserede.
    sims = text_embeddings @ prototypes.T

    categories = sorted(set(top_categories))
    cat_to_tag_indices: Dict[str, List[int]] = {cat: [] for cat in categories}
    for i, cat in enumerate(top_categories):
        cat_to_tag_indices[cat].append(i)

    results: List[List[Dict[str, Any]]] = []
    for row in sims:
        assigned: List[Dict[str, Any]] = []
        for cat, tag_indices in cat_to_tag_indices.items():
            cat_scores = [(idx, float(row[idx])) for idx in tag_indices]
            cat_scores.sort(key=lambda pair: pair[1], reverse=True)
            for rank, (idx, score) in enumerate(cat_scores[:top_k_per_category], start=1):
                if score < min_score:
                    continue
                assigned.append(
                    {
                        "tag_key": tag_keys[idx],
                        "top_category": cat,
                        "score": score,
                        "rank_in_category": rank,
                    }
                )
        results.append(assigned)
    return results


def persist_text_tags(
    conn,
    target_type: str,
    target_id: str,
    assigned_tags: List[Dict[str, Any]],
) -> None:
    """Erstat tags for én tekst-enhed i samme PostgreSQL-transaktion."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM text_tags WHERE target_type = %s AND target_id = %s",
            (target_type, target_id),
        )
        if not assigned_tags:
            return
        rows = [
            (target_type, target_id, t["tag_key"], t["score"], t["rank_in_category"])
            for t in assigned_tags
        ]
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO text_tags (target_type, target_id, tag_key, similarity_score, rank_in_category)
            VALUES %s
            """,
            rows,
        )


def persist_all_text_tags(
    conn,
    targets: List[Tuple[str, str, str]],
    assignments: List[List[Dict[str, Any]]],
) -> int:
    """Persister alle tekst-tag-relationer atomisk og returnerer antal relationer."""
    if len(targets) != len(assignments):
        raise ValueError("Antallet af targets og tag-assignment-resultater skal være ens.")

    with conn.cursor() as cur:
        cur.execute("DELETE FROM text_tags")
        rows = []
        for (target_type, target_id, _), assigned_tags in zip(targets, assignments):
            rows.extend(
                (
                    target_type,
                    target_id,
                    tag["tag_key"],
                    tag["score"],
                    tag["rank_in_category"],
                )
                for tag in assigned_tags
            )

        if rows:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO text_tags
                    (target_type, target_id, tag_key, similarity_score, rank_in_category)
                VALUES %s
                """,
                rows,
            )
    return len(rows)


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-8, None)


# ── Query-tid: tag en enkelt ny tekst (fx et indkommende jobopslag) ─────────

def assign_tags_to_text(
    model: SentenceTransformer,
    tag_keys: List[str],
    top_categories: List[str],
    prototypes: np.ndarray,
    text: str,
    top_k_per_category: int = DEFAULT_TOP_K_PER_CATEGORY,
    min_score: float = DEFAULT_MIN_SCORE,
) -> List[Dict[str, Any]]:
    """Bruges ved query-tid til at tagge et nyt jobopslag/paragraf on-the-fly."""
    embedding = embed_texts(model, [text])
    return assign_tags_to_embeddings(
        embedding, tag_keys, top_categories, prototypes, top_k_per_category, min_score
    )[0]


# ── Main: batch-tag alt i databasen ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K_PER_CATEGORY)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    args = parser.parse_args()

    print("=" * 70)
    print("  Taksonomisk tagging via embedding similarity")
    print("=" * 70)

    model = SentenceTransformer(EMBEDDING_MODEL_PATH)

    conn = psycopg2.connect(POSTGRESQL_DSN)
    try:
        print("Kontrollerer/opretter tag-tabeller i Postgres ...")
        with conn.cursor() as cur:
            cur.execute(CREATE_TAG_TABLES_SQL)
        conn.commit()

        print("Henter taksonomiske tags og seed-eksempler fra Postgres ...")
        flat_tags = fetch_tags_from_db(conn)
        print(f"Indlæst {len(flat_tags)} tags fra databasen.")

        print("Beregner prototype-embeddings pr. tag ...")
        tag_keys, top_categories, prototypes = build_tag_prototypes(model, flat_tags)

        print("Henter dokumenter og paragraffer der skal tagges ...")
        targets = fetch_targets(conn)
        print(f"  {len(targets)} tekst-enheder fundet.")

        if not targets:
            conn.commit()
            print("Ingen tekster at tagge, afslutter.")
            return

        texts = [t[2] for t in targets]
        print("Embedder tekster ...")
        text_embeddings = embed_texts(model, texts)

        print("Matcher mod tag-prototyper ...")
        assignments = assign_tags_to_embeddings(
            text_embeddings, tag_keys, top_categories, prototypes,
            top_k_per_category=args.top_k, min_score=args.min_score,
        )

        print("Gemmer tekst-tag-relationer i text_tags ...")
        total_assignments = persist_all_text_tags(conn, targets, assignments)
        conn.commit()

        print(f"Færdig. {total_assignments} tag-tildelinger gemt for {len(targets)} tekst-enheder.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
