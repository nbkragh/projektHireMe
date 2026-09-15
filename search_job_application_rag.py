"""
search_job_application_rag.py
=============================

Søger et nyt jobopslag mod de PostgreSQL/FAISS-data, som bygges af
build_faiss_indexes.py og tag_assignment.py.

Flow:
    ny jobtekst
      -> sentence split + paragraphize()
      -> hele jobopslag via job_documents-FAISS
      -> query-paragraffer via job_paragraphs-FAISS
      -> app-paragraffer via job_app_paragraph_matches
      -> vector-score + tag-overlap
      -> few-shot prompt med hele dokumenter og svar-paragraffer

Eksempel:
    python search_job_application_rag.py job.txt --document-k 3 --paragraph-k 5
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import faiss
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

from cleanup_canonicalize_semanticchunkify import paragraphize


DEFAULT_DSN = os.environ.get(
    "POSTGRES_DSN",
)
DEFAULT_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
)
DEFAULT_FAISS_DIR = Path(os.environ.get("FAISS_DIR", "faiss_indexes"))
DEFAULT_TAG_TOP_K = 3
DEFAULT_TAG_MIN_SCORE = 0.35


def split_sentences(text: str) -> List[str]:
    """Enkel robust sentence split før den eksisterende paragraphize()."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÆØÅ0-9])|\n+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def make_query_paragraphs(text: str, model: SentenceTransformer) -> List[str]:
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("Jobopslagsteksten indeholder ingen sætninger.")

    query_case = {
        "case_id": "query",
        "job_sentences": sentences,
        "app_sentences": sentences,
    }
    segmented = paragraphize([query_case], model)
    return [paragraph["text"] for paragraph in segmented[0]["job_paragraphs"]]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype("float32", copy=False)
    faiss.normalize_L2(vectors)
    return vectors


def load_query_tags(
    conn,
    model: SentenceTransformer,
    query_text: str,
    top_k: int,
    min_score: float,
) -> Set[str]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT t.tag_key, t.top_category, s.seed_index, s.seed_text
            FROM tags t
            JOIN tag_seeds s ON s.tag_key = t.tag_key
            ORDER BY t.top_category, t.tag_key, s.seed_index
            """
        )
        rows = cursor.fetchall()

    grouped: Dict[str, Tuple[str, List[str]]] = {}
    for tag_key, category, _, seed_text in rows:
        grouped.setdefault(tag_key, (category, []))[1].append(seed_text)

    tag_keys = []
    categories = []
    prototypes = []
    for tag_key, (category, seeds) in grouped.items():
        seed_vectors = normalize_vectors(
            model.encode(seeds, convert_to_numpy=True).astype("float32")
        )
        prototype = seed_vectors.mean(axis=0, keepdims=True)
        prototype = normalize_vectors(prototype)[0]
        tag_keys.append(tag_key)
        categories.append(category)
        prototypes.append(prototype)

    if not prototypes:
        raise ValueError("Ingen tags/seed-eksempler fundet i PostgreSQL.")

    query_vector = normalize_vectors(
        model.encode([query_text], convert_to_numpy=True).astype("float32")
    )[0]
    scores = query_vector @ np.vstack(prototypes).T
    selected: Set[str] = set()
    for category in sorted(set(categories)):
        indices = [i for i, value in enumerate(categories) if value == category]
        ranked = sorted(indices, key=lambda i: float(scores[i]), reverse=True)
        selected.update(
            tag_keys[i]
            for i in ranked[:top_k]
            if float(scores[i]) >= min_score
        )
    return selected


def fetch_faiss_hits(
    conn,
    index: faiss.Index,
    collection: str,
    query_vector: np.ndarray,
    k: int,
) -> List[Dict[str, Any]]:
    distances, positions = index.search(query_vector.reshape(1, -1), k)
    hits = []
    with conn.cursor() as cursor:
        for score, position in zip(distances[0], positions[0]):
            if position < 0:
                continue
            cursor.execute(
                """
                SELECT source_table, source_id
                FROM faiss_id_map
                WHERE collection_name = %s AND faiss_index = %s
                """,
                (collection, int(position)),
            )
            mapping = cursor.fetchone()
            if not mapping:
                continue
            hits.append(
                {
                    "score": float(score),
                    "source_table": mapping[0],
                    "source_id": int(mapping[1]),
                }
            )
    return hits


def fetch_document_hit(conn, source_id: int, table: str) -> Dict[str, Any]:
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT id, case_id, raw_text FROM {table} WHERE id = %s",
            (source_id,),
        )
        row = cursor.fetchone()
    if not row:
        raise LookupError(f"Databasepost {table}/{source_id} findes ikke.")
    return {"id": row[0], "case_id": row[1], "text": row[2]}


def fetch_paragraph_hit(conn, source_id: int, table: str) -> Dict[str, Any]:
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT id, paragraph_id, case_id, text FROM {table} WHERE id = %s",
            (source_id,),
        )
        row = cursor.fetchone()
    if not row:
        raise LookupError(f"Databasepost {table}/{source_id} findes ikke.")
    return {
        "id": row[0],
        "paragraph_id": row[1],
        "case_id": row[2],
        "text": row[3],
    }


def fetch_tags(conn, target_type: str, target_id: str) -> Set[str]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT tag_key FROM text_tags
            WHERE target_type = %s AND target_id = %s
            """,
            (target_type, target_id),
        )
        return {row[0] for row in cursor.fetchall()}


def fetch_mapped_app_paragraphs(
    conn,
    job_paragraph_id: str,
    limit: int,
) -> List[Dict[str, Any]]:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT ap.id, ap.paragraph_id, ap.case_id, ap.text,
                   m.similarity_score, m.rank
            FROM job_app_paragraph_matches m
            JOIN app_paragraphs ap ON ap.paragraph_id = m.app_paragraph_id
            WHERE m.job_paragraph_id = %s
            ORDER BY m.rank
            LIMIT %s
            """,
            (job_paragraph_id, limit),
        )
        return [
            {
                "id": row[0],
                "paragraph_id": row[1],
                "case_id": row[2],
                "text": row[3],
                "mapping_score": float(row[4]),
                "mapping_rank": row[5],
            }
            for row in cursor.fetchall()
        ]


def build_prompt(
    query_text: str,
    full_document_examples: List[Dict[str, Any]],
    paragraph_examples: List[Dict[str, Any]],
) -> str:
    documents = "\n\n".join(
        f"--- Eksempel {i}: tidligere jobopslag ---\n{item['job_text']}\n"
        f"--- Tilhørende ansøgning ---\n{item['app_text']}"
        for i, item in enumerate(full_document_examples, start=1)
    )
    paragraphs = "\n\n".join(
        f"--- Kravparagraf ---\n{item['job_paragraph']}\n"
        f"--- Tidligere svarparagraf ---\n{item['app_paragraph']}"
        for item in paragraph_examples
    )
    return f"""Skriv en målrettet jobansøgning på dansk til opslaget nedenfor.

Brug de tidligere eksempler som stil- og argumentationsreference.
Nævn kun konkrete teknologier og værktøjer, som enten fremgår af opslaget
eller er dokumenterede relevante kompetencer hos kandidaten.
Opfind ikke erfaringer.

=== NYT JOBOPSLAG ===
{query_text}

=== HELE TIDLIGERE EKSEMPLER ===
{documents}

=== KRAV -> SVAR-EKSEMPLER ===
{paragraphs}
"""


def search(
    query_text: str,
    model: SentenceTransformer,
    conn,
    faiss_dir: Path,
    document_k: int,
    paragraph_k: int,
    tag_top_k: int,
    tag_min_score: float,
) -> Dict[str, Any]:
    query_paragraphs = make_query_paragraphs(query_text, model)
    query_tags = load_query_tags(
        conn, model, query_text, tag_top_k, tag_min_score
    )

    document_index = faiss.read_index(str(faiss_dir / "job_documents.faiss"))
    document_vector = normalize_vectors(
        model.encode([query_text], convert_to_numpy=True).astype("float32")
    )[0]
    document_hits = fetch_faiss_hits(
        conn, document_index, "job_documents", document_vector, document_k
    )

    full_documents = []
    for hit in document_hits:
        job = fetch_document_hit(conn, hit["source_id"], "job_documents")
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT raw_text FROM app_documents WHERE case_id = %s",
                (job["case_id"],),
            )
            app_row = cursor.fetchone()
        if not app_row:
            continue
        job_tags = fetch_tags(conn, "job_document", job["case_id"])
        full_documents.append(
            {
                "case_id": job["case_id"],
                "vector_score": hit["score"],
                "tag_overlap": len(query_tags & job_tags),
                "job_text": job["text"],
                "app_text": app_row[0],
            }
        )
    full_documents.sort(
        key=lambda item: (item["tag_overlap"], item["vector_score"]),
        reverse=True,
    )

    paragraph_index = faiss.read_index(str(faiss_dir / "job_paragraphs.faiss"))
    paragraph_examples = []
    for query_paragraph in query_paragraphs:
        query_vector = normalize_vectors(
            model.encode([query_paragraph], convert_to_numpy=True).astype("float32")
        )[0]
        paragraph_hits = fetch_faiss_hits(
            conn, paragraph_index, "job_paragraphs", query_vector, paragraph_k
        )
        for hit in paragraph_hits:
            job_paragraph = fetch_paragraph_hit(
                conn, hit["source_id"], "job_paragraphs"
            )
            job_tags = fetch_tags(
                conn, "job_paragraph", job_paragraph["paragraph_id"]
            )
            overlap = len(query_tags & job_tags)
            for app_paragraph in fetch_mapped_app_paragraphs(conn, job_paragraph["paragraph_id"], 3):
                app_tags = fetch_tags(conn, "app_paragraph", app_paragraph["paragraph_id"])
                paragraph_examples.append(
                    {
                        "query_paragraph": query_paragraph,
                        "job_paragraph": job_paragraph["text"],
                        "app_paragraph": app_paragraph["text"],
                        "case_id": job_paragraph["case_id"],
                        "vector_score": hit["score"],
                        "tag_overlap": overlap + len(query_tags & app_tags),
                        "mapping_score": app_paragraph["mapping_score"],
                    }
                )

    paragraph_examples.sort(
        key=lambda item: (
            item["tag_overlap"],
            item["vector_score"],
            item["mapping_score"],
        ),
        reverse=True,
    )
    paragraph_examples = paragraph_examples[:paragraph_k]

    return {
        "query_tags": sorted(query_tags),
        "query_paragraphs": query_paragraphs,
        "full_document_examples": full_documents[:document_k],
        "paragraph_examples": paragraph_examples,
        "few_shot_prompt": build_prompt(
            query_text,
            full_documents[:document_k],
            paragraph_examples,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_text_file", type=Path)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--faiss-dir", type=Path, default=DEFAULT_FAISS_DIR)
    parser.add_argument("--document-k", type=int, default=3)
    parser.add_argument("--paragraph-k", type=int, default=5)
    parser.add_argument("--tag-top-k", type=int, default=DEFAULT_TAG_TOP_K)
    parser.add_argument("--tag-min-score", type=float, default=DEFAULT_TAG_MIN_SCORE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    query_text = args.job_text_file.read_text(encoding="utf-8")
    model = SentenceTransformer(args.model)
    conn = psycopg2.connect(args.dsn)
    try:
        result = search(
            query_text,
            model,
            conn,
            args.faiss_dir,
            args.document_k,
            args.paragraph_k,
            args.tag_top_k,
            args.tag_min_score,
        )
    finally:
        conn.close()

    if args.output:
        args.output.write_text(result["few_shot_prompt"] + "\n", encoding="utf-8")
        print(f"Resultat gemt i {args.output}")

    print(result)


if __name__ == "__main__":
    main()
