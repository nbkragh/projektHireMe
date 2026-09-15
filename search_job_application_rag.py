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
import os
import re
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

import faiss
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

from cleanup_canonicalize_semanticchunkify import paragraphize

try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    print("Import of python-dotenv failed. python-dotenv ikke installeret.")
    sys.exit(1)
    
DEFAULT_DSN = os.environ.get(
    "POSTGRES_DSN",
)
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
)
DEFAULT_FAISS_DIR = Path(os.environ.get("FAISS_DIR", "faiss_indexes"))
DEFAULT_COMPETENCY_TAG_FILE = Path(
    os.environ.get("COMPETENCY_TAG_FILE", "kompetencer_og_erfaringer_tags.txt")
)
DEFAULT_TAG_TOP_K = 5
DEFAULT_TAG_MIN_SCORE = 0.35


# Opdeler jobteksten i rensede sætninger, som kan sendes videre til paragraphize().
def split_sentences(text: str) -> List[str]:
    """Enkel robust sentence split før den eksisterende paragraphize()."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÆØÅ0-9])|\n+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# Danner query-paragraffer ved først at splitte teksten og derefter genbruge paragraphize().
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


# Konverterer vektorer til float32 og L2-normaliserer dem til cosine-lignende FAISS-søgning.
def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype("float32", copy=False)
    faiss.normalize_L2(vectors)
    return vectors


# Finder kompetenceord i opslaget:
# 1. Læs ét tag eller én frase pr. linje og fjern eventuel linjenummerering.
# 2. Normaliserer case og tegn, men bevarer både tekst med og uden mellemrum.
# 3. Søger efter grundformen og enkle danske entals-/flertalsvariationer.
# 4. Tillader substring-match, så tags også findes i bindestreger og sammensatte ord.
def find_competency_matches(
    query_text: str,
    tag_file: Path,
) -> List[str]:
    tag_text = tag_file.read_text(encoding="utf-8")
    query_lower = query_text.casefold()
    query_compact = re.sub(r"[^0-9a-zæøå]+", "", query_lower)
    matches = []
    seen = set()

    for raw_line in tag_text.splitlines():
        term = re.sub(r"^\s*\d+\.\s*", "", raw_line).strip()
        term = re.sub(r"\s+", " ", term).strip(" \t.,;:")
        if not term:
            continue

        normalized_term = term.casefold()
        variants = {normalized_term}
        if len(normalized_term) > 2:
            if normalized_term.endswith("er"):
                variants.add(normalized_term[:-2])
            elif normalized_term.endswith("s"):
                variants.add(normalized_term[:-1])
            else:
                variants.update(
                    {normalized_term + "s", normalized_term + "er"}
                )

        matched = False
        for variant in variants:
            compact_variant = re.sub(r"[^0-9a-zæøå]+", "", variant)
            if len(compact_variant) <= 1:
                matched = bool(
                    re.search(
                        rf"(?<![a-zæøå]){re.escape(compact_variant)}(?![a-zæøå])",
                        query_lower,
                    )
                )
            else:
                matched = compact_variant in query_compact
            if matched:
                break

        if matched and normalized_term not in seen:
            matches.append(term)
            seen.add(normalized_term)

    return matches


# Henter de korteste ansøgningsparagraffer, som indeholder mindst ét opslag-match.
# Søgningen normaliserer databaseteksten, så mellemrum, bindestreger og case ikke
# forhindrer match mod sammensatte ord eller simple entals-/flertalsvariationer.
def fetch_competency_app_paragraphs(
    conn,
    competency_matches: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    if not competency_matches or limit <= 0:
        return []

    variants = set()
    for term in competency_matches:
        normalized = term.casefold()
        term_variants = {normalized}
        if len(normalized) > 2:
            if normalized.endswith("er"):
                term_variants.add(normalized[:-2])
            elif normalized.endswith("s"):
                term_variants.add(normalized[:-1])
            else:
                term_variants.update({normalized + "s", normalized + "er"})
        variants.update(
            re.sub(r"[^0-9a-zæøå]+", "", variant)
            for variant in term_variants
        )

    variants = {variant for variant in variants if len(variant) > 1}
    if not variants:
        return []

    conditions = " OR ".join(
        "regexp_replace(lower(ap.text), '[^a-zæøå0-9]', '', 'g') LIKE %s"
        for _ in variants
    )
    parameters = [f"%{variant}%" for variant in sorted(variants)]
    parameters.append(limit)

    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT ap.id, ap.paragraph_id, ap.case_id, ap.text
            FROM app_paragraphs ap
            WHERE {conditions}
            ORDER BY length(ap.text), ap.id
            LIMIT %s
            """,
            parameters,
        )
        rows = cursor.fetchall()

    return [
        {
            "id": row[0],
            "paragraph_id": row[1],
            "case_id": row[2],
            "text": row[3],
        }
        for row in rows
    ]


# Finder relevante tags for den nye jobtekst:
# 1. Hent tags og seed-eksempler fra databasen.
# 2. Byg ét normaliseret prototypevektor pr. tag ud fra dets seeds.
# 3. Sammenlign query-vektoren med alle prototyper.
# 4. Vælg de bedste tags pr. kategori over minimumsscoren.
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


# Oversætter FAISS-resultater til databaseposter:
# 1. Søg efter de k nærmeste vektorer.
# 2. Slå hver FAISS-position op i faiss_id_map.
# 3. Returnér score og den tilhørende databaseidentifikator.
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


# Henter én heltekstdokumentpost fra den angivne database-tabel.
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

# Henter én paragrafpost ud fra den database-id, som FAISS-søgningen returnerede.
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


# Henter alle tags, der er gemt for en bestemt dokument- eller paragrafpost.
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


# Henter de bedst rangerede ansøgningsparagraffer, som er mappet til en jobparagraf.
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


# Samler hele dokumenteksempler og krav-svar-paragraffer i den endelige few-shot prompt.
def build_prompt(
    query_text: str,
    full_document_examples: List[Dict[str, Any]],
    paragraph_examples: List[Dict[str, Any]],
    competency_matches: List[str],
    competency_app_paragraphs: List[Dict[str, Any]],
) -> str:
    documents = "\n\n".join(
        f"--- Eksempel {i}: tidligere jobopslag ---\n{item['job_text']}\n"
        f"--- Tilhørende ansøgning ---\n{item['app_text']}"
        for i, item in enumerate(full_document_examples, start=1)
    )
    paragraphs = "\n\n".join(
        f"--- Kravparagraf ---\n{item['job_paragraph']}\n"
        f"--- Tilhørende svarparagraf ---\n{item['app_paragraph']}"
        for item in paragraph_examples
    )
    competency_list = "\n".join(
        f"- {match}" for match in competency_matches
    ) or "- Ingen direkte kompetence-, færdigheds- eller kvalitetsmatches fundet."
    competency_examples = "\n\n".join(
        f"--- Kort ansøgnings-eksempel ---\n{item['text']}"
        for item in competency_app_paragraphs
    ) or "- Ingen matchende ansøgningsparagraffer fundet."


    return f"""Du er en AI-assistent, der hjælper med at skrive en målrettede jobansøgning til det nye jobopslag ved
      at analysere tidligere jobopslag og tilhørende ansøgningseksempler, og tidligere kravparagraffer med deres tilhørende svarparagraffer.
    
Brug de tidligere eksempler som stil- og argumentationsreference.
Nævn kun konkrete teknologier og værktøjer, som enten fremgår af opslaget
eller er dokumenterede relevante kompetencer hos kandidaten.
Opfind ikke erfaringer.

Hard requirements:
- Skriv på samme sprog som kildeteksterne
- Anvend samme tone og sproglige stil som i kildeteksterne.
- Brug udelukkende konkrete erfaringer, kvaliteter, kompetencer og teknologier - opfind IKKE fakta!
- Prioriter match mod stillingsopslaget og vis tydelig motivation for virksomheden i samme tone og personlighed som kildeteksterne.
- Skriv en overskrift til ansøgningen der matcher jobtitlen og tonen i kildeteksterne.
- Brug IKKE tankestreger (—) i ansøgningsteksten. Brug i stedet kolon, komma eller skriv sætningen om.
- Undgå omstændelige metaformuleringer som 'stillingen kombinerer noget, jeg er motiveret af'. Skriv direkte, fx 'jeg er motiveret af at'.
- Undgå at spejle stillingsopslaget unødigt med formuleringer som 'det matcher jeres behov'. Skriv i stedet direkte hvad jeg kan bidrage med.
- Undgå selvnedtonende eller kompetencenedskrivende formuleringer som 'jeg kommer ikke med en tung profil' eller 'min primære erfaring er ikke'. 
- Fremhæv dokumenterede styrker neutralt og uden forbehold.
- Brug gerne kompetencer, færdigheder og kvaliteter fra matchlisten nedenfor,
  når de er relevante og kan dokumenteres i kandidatens kildetekster.
- Nævn ikke et match fra listen som kandidatens erfaring, hvis kildeteksterne
  ikke dokumenterer erfaringen. Opfind aldrig erfaring.

=== NYT JOBOPSLAG ===
Dette er det nye jobopslag, som kandidaten skal du skal skrive en ansøgning til for kandidaten:
{query_text}

=== MATCHENDE KOMPETENCER, FÆRDIGHEDER OG KVALITETER ===
Disse termer er fundet i det nye jobopslag og må gerne nævnes, når de kan
understøttes af kandidatens dokumenterede erfaring:
{competency_list}

=== KORTE ANSØGNINGSEKSEMPLER MED MATCHENDE KOMPETENCER ===
Brug disse korte eksempler på formulering af matchende kompetencer og erfaringsreferencer:
{competency_examples}

=== HELE TIDLIGERE EKSEMPLER ===
Dette er et eksempel på hele tidligere ansøgningstekster, som kandidaten har skrevet til lignende jobopslag:
{documents}

=== KRAV -> SVAR-EKSEMPLER ===
Her er eksempler på, hvordan kravene i jobopslaget kan besvares i ansøgningsteksten:
{paragraphs}
"""


# Udfører den samlede retrieval:
# 1. Danner query-paragraffer og query-tags.
# 2. Henter hele tidligere jobopslag via FAISS og finder deres ansøgninger.
# 3. Henter lignende jobparagraffer og deres mappede ansøgningsparagraffer.
# 4. Grupperer på jobparagraf, så hver valgt jobparagraf kun bidrager én gang.
# 5. Vælger den bedste ansøgningsparagraf for hver jobparagraf og bygger prompten.
def search(
    query_text: str,
    model: SentenceTransformer,
    conn,
    faiss_dir: Path,
    document_k: int,
    paragraph_k: int,
    tag_top_k: int,
    tag_min_score: float,
    competency_tag_file: Path,
    competency_paragraph_k: int,
) -> Dict[str, Any]:
    query_paragraphs = make_query_paragraphs(query_text, model)
    competency_matches = find_competency_matches(query_text, competency_tag_file)
    competency_app_paragraphs = fetch_competency_app_paragraphs(
        conn, competency_matches, competency_paragraph_k
    )
    query_tags = load_query_tags(conn, model, query_text, tag_top_k, tag_min_score)

    document_index = faiss.read_index(str(faiss_dir / "job_documents.faiss"))
    document_vector = normalize_vectors(
        model.encode([query_text], convert_to_numpy=True).astype("float32")
    )[0]
    document_hits = fetch_faiss_hits(conn, document_index, "job_documents", document_vector, document_k)

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
            job_tag_overlap = len(query_tags & job_tags)
            for app_paragraph in fetch_mapped_app_paragraphs(conn, job_paragraph["paragraph_id"], 3):
                app_tags = fetch_tags(conn, "app_paragraph", app_paragraph["paragraph_id"])
                app_tag_overlap = len(query_tags & app_tags)
                paragraph_examples.append(
                    {
                        "query_paragraph": query_paragraph,
                        "job_paragraph_id": job_paragraph["paragraph_id"],
                        "job_paragraph": job_paragraph["text"],
                        "app_paragraph": app_paragraph["text"],
                        "case_id": job_paragraph["case_id"],
                        "vector_score": hit["score"],
                        "job_tag_overlap": job_tag_overlap,
                        "app_tag_overlap": app_tag_overlap,
                        "tag_overlap": job_tag_overlap + app_tag_overlap,
                        "mapping_score": app_paragraph["mapping_score"],
                    }
                )

    # Vælg først forskellige jobparagraffer. For hver valgt jobparagraf
    # beholdes derefter den ansøgningsparagraf, der matcher bedst.
    grouped_paragraphs: Dict[Any, List[Dict[str, Any]]] = {}
    for candidate in paragraph_examples:
        grouped_paragraphs.setdefault(
            candidate["job_paragraph_id"], []
        ).append(candidate)

    selected_paragraphs = []
    for candidates in grouped_paragraphs.values():
        best_job_candidate = max(
            candidates,
            key=lambda item: (
                item["job_tag_overlap"],
                item["vector_score"],
            ),
        )
        best_app_candidate = max(
            candidates,
            key=lambda item: (
                item["app_tag_overlap"],
                item["mapping_score"],
                item["vector_score"],
            ),
        )
        selected_paragraphs.append(
            {
                **best_job_candidate,
                "app_paragraph": best_app_candidate["app_paragraph"],
                "app_tag_overlap": best_app_candidate["app_tag_overlap"],
                "mapping_score": best_app_candidate["mapping_score"],
                "tag_overlap": (
                    best_job_candidate["job_tag_overlap"]
                    + best_app_candidate["app_tag_overlap"]
                ),
            }
        )

    selected_paragraphs.sort(
        key=lambda item: (
            item["job_tag_overlap"],
            item["vector_score"],
            item["app_tag_overlap"],
            item["mapping_score"],
        ),
        reverse=True,
    )
    paragraph_examples = selected_paragraphs[:paragraph_k]

    return {
        "query_tags": sorted(query_tags),
        "competency_matches": competency_matches,
        "competency_app_paragraphs": competency_app_paragraphs,
        "query_paragraphs": query_paragraphs,
        "full_document_examples": full_documents[:document_k],
        "paragraph_examples": paragraph_examples,
        "few_shot_prompt": build_prompt(
            query_text,
            full_documents[:document_k],
            paragraph_examples,
            competency_matches,
            competency_app_paragraphs,
        ),
    }


# Parser CLI-argumenter, indlæser model og input, kører søgningen og gemmer eventuelt prompten.
def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_text_file", type=Path)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--faiss-dir", type=Path, default=DEFAULT_FAISS_DIR)
    parser.add_argument(
        "--competency-tag-file",
        type=Path,
        default=DEFAULT_COMPETENCY_TAG_FILE,
        help="Fil med kompetence-, færdigheds- og kvalitetstags, ét pr. linje.",
    )
    parser.add_argument("--document-k", type=int, default=1)
    parser.add_argument("--paragraph-k", type=int, default=5)
    parser.add_argument(
        "--competency-paragraph-k",
        type=int,
        default=5,
        help="Antal korte ansøgningsparagraffer med kompetence-match i prompten.",
    )
    parser.add_argument("--tag-top-k", type=int, default=DEFAULT_TAG_TOP_K)
    parser.add_argument("--tag-min-score", type=float, default=DEFAULT_TAG_MIN_SCORE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    query_text = args.job_text_file.read_text(encoding="utf-8")
    model = SentenceTransformer(EMBEDDING_MODEL)
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
            args.competency_tag_file,
            args.competency_paragraph_k,
        )
    finally:
        conn.close()

    if args.output:
        args.output.write_text(result["few_shot_prompt"] + "\n", encoding="utf-8")
        print(f"Resultat gemt i {args.output}")

    print(result)


if __name__ == "__main__":
    main()
