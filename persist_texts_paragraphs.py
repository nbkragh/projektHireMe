"""
persist_segmented_ready.py
==========================

Persisterer segmented_and_ready.json i PostgreSQL.

Inputformat:
    [
      {
        "case": "...",
        "case_id": "c0",
        "job_text": "...",
        "app_text": "...",
        "job_paragraphs": [{"paragraph_id": "...", "text": "..."}],
        "app_paragraphs": [{"paragraph_id": "...", "text": "..."}]
      }
    ]

Scriptet opretter de nødvendige tabeller, hvis de ikke findes, og kan køres
flere gange uden at oprette dubletter. Data opdateres via ON CONFLICT.

Eksempler:
    python persist_segmented_ready.py
    python persist_segmented_ready.py --input segmented_and_ready.json
    python persist_segmented_ready.py --dsn "dbname=jobapp_rag user=postgres"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv

    load_dotenv(".env")
except ImportError:
    pass


DEFAULT_INPUT = os.environ.get("SEGMENTED_READY_JSON", "segmented_and_ready.json")
DEFAULT_DSN = os.environ.get(
    "POSTGRES_DSN",
    "dbname=jobapp_rag user=postgres host=localhost port=5432",
)
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_TOP_N_MATCHES = 3


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS cases (
    case_id     TEXT PRIMARY KEY,
    case_name   TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS job_documents (
    id         SERIAL PRIMARY KEY,
    case_id    TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    raw_text   TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (case_id)
);

CREATE TABLE IF NOT EXISTS app_documents (
    id         SERIAL PRIMARY KEY,
    case_id    TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    raw_text   TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (case_id)
);

CREATE TABLE IF NOT EXISTS job_paragraphs (
    id           SERIAL PRIMARY KEY,
    paragraph_id TEXT NOT NULL UNIQUE,
    case_id      TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    text         TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app_paragraphs (
    id                SERIAL PRIMARY KEY,
    paragraph_id      TEXT NOT NULL UNIQUE,
    case_id           TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    text              TEXT NOT NULL,
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_job_paragraphs_case
    ON job_paragraphs(case_id);
CREATE INDEX IF NOT EXISTS idx_app_paragraphs_case
    ON app_paragraphs(case_id);

CREATE TABLE IF NOT EXISTS job_app_paragraph_matches (
    id                  SERIAL PRIMARY KEY,
    case_id             TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    job_paragraph_id    TEXT NOT NULL REFERENCES job_paragraphs(paragraph_id) ON DELETE CASCADE,
    app_paragraph_id    TEXT NOT NULL REFERENCES app_paragraphs(paragraph_id) ON DELETE CASCADE,
    similarity_score    REAL NOT NULL,
    rank                INTEGER NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE (job_paragraph_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_paragraph_matches_job
    ON job_app_paragraph_matches(job_paragraph_id);
CREATE INDEX IF NOT EXISTS idx_paragraph_matches_case
    ON job_app_paragraph_matches(case_id);
"""


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("segmented_and_ready.json skal indeholde en JSON-liste.")
    if not all(isinstance(case, dict) for case in data):
        raise ValueError("Alle elementer i JSON-listen skal være case-objekter.")
    return data


def required_text(case: Dict[str, Any], key: str) -> str:
    value = case.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{key!r} skal være en tekststreng.")
    return value


def validate_paragraphs(
    case_id: str, paragraphs: Any, field_name: str
) -> Iterable[Tuple[str, str]]:
    if paragraphs is None:
        return []
    if not isinstance(paragraphs, list):
        raise ValueError(f"{field_name} i {case_id!r} skal være en liste.")

    validated = []
    for index, paragraph in enumerate(paragraphs):
        if not isinstance(paragraph, dict):
            raise ValueError(f"En paragraf i {field_name} for {case_id!r} er ikke et objekt.")

        paragraph_id = paragraph.get("paragraph_id")
        text = paragraph.get("text")
        if not isinstance(paragraph_id, str) or not paragraph_id:
            raise ValueError(f"Paragraf {index} i {case_id!r} mangler paragraph_id.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Paragraf {paragraph_id!r} mangler tekst.")

        validated.append((paragraph_id, text))
    return validated


def upsert_case(cursor, case: Dict[str, Any]) -> str:
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("En case mangler et gyldigt case_id.")

    cursor.execute(
        """
        INSERT INTO cases (case_id, case_name)
        VALUES (%s, %s)
        ON CONFLICT (case_id) DO UPDATE SET
            case_name = EXCLUDED.case_name
        """,
        (case_id, case["case"]),
    )
    return case_id


def upsert_document(cursor, table: str, case_id: str, raw_text: str) -> None:
    if not raw_text:
        return
    if table not in {"job_documents", "app_documents"}:
        raise ValueError(f"Uventet dokumenttabel: {table}")
    cursor.execute(
        f"""
        INSERT INTO {table} (case_id, raw_text)
        VALUES (%s, %s)
        ON CONFLICT (case_id) DO UPDATE SET raw_text = EXCLUDED.raw_text
        """,
        (case_id, raw_text),
    )


def upsert_paragraphs(
    cursor,
    table: str,
    case_id: str,
    paragraphs: Iterable[Tuple[str, str]],
) -> int:
    if table not in {"job_paragraphs", "app_paragraphs"}:
        raise ValueError(f"Uventet paragraftabel: {table}")

    count = 0
    for paragraph_id, text in paragraphs:
        cursor.execute(
            f"""
            INSERT INTO {table}
                (paragraph_id, case_id,  text)
            VALUES (%s, %s, %s)
            ON CONFLICT (paragraph_id) DO UPDATE SET
                case_id = EXCLUDED.case_id,

                text = EXCLUDED.text
            """,
            (paragraph_id, case_id, text),
        )
        count += 1
    return count


def fetch_case_paragraphs(cursor, table: str, case_id: str) -> List[Tuple[str, str]]:
    if table not in {"job_paragraphs", "app_paragraphs"}:
        raise ValueError(f"Uventet paragraftabel: {table}")
    cursor.execute(
        f"SELECT paragraph_id, text FROM {table} WHERE case_id = %s ORDER BY id",
        (case_id,),
    )
    return cursor.fetchall()


def persist_paragraph_matches(
    conn,
    model: SentenceTransformer,
    case_ids: Iterable[str],
    top_n: int,
) -> int:
    """Gemmer de top-N mest lignende ansøgningsparagraffer pr. jobparagraf."""
    if top_n < 1:
        raise ValueError("top_n skal være mindst 1.")

    total_matches = 0
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM job_app_paragraph_matches")

        for case_id in case_ids:
            job_rows = fetch_case_paragraphs(cursor, "job_paragraphs", case_id)
            app_rows = fetch_case_paragraphs(cursor, "app_paragraphs", case_id)
            if not job_rows or not app_rows:
                continue

            job_vectors = model.encode(
                [text for _, text in job_rows],
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            app_vectors = model.encode(
                [text for _, text in app_rows],
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")

            job_norms = np.linalg.norm(job_vectors, axis=1, keepdims=True)
            app_norms = np.linalg.norm(app_vectors, axis=1, keepdims=True)
            job_vectors /= np.clip(job_norms, 1e-8, None)
            app_vectors /= np.clip(app_norms, 1e-8, None)
            similarities = job_vectors @ app_vectors.T

            rows = []
            for job_index, (job_paragraph_id, _) in enumerate(job_rows):
                ranked_app_indices = np.argsort(similarities[job_index])[::-1][:top_n]
                for rank, app_index in enumerate(ranked_app_indices, start=1):
                    rows.append(
                        (
                            case_id,
                            job_paragraph_id,
                            app_rows[int(app_index)][0],
                            float(similarities[job_index, app_index]),
                            rank,
                        )
                    )

            psycopg2.extras.execute_values(
                cursor,
                """
                INSERT INTO job_app_paragraph_matches
                    (case_id, job_paragraph_id, app_paragraph_id, similarity_score, rank)
                VALUES %s
                """,
                rows,
            )
            total_matches += len(rows)
    return total_matches


def persist(
    conn,
    cases: List[Dict[str, Any]],
    model: SentenceTransformer,
    top_n_matches: int,
) -> Dict[str, int]:
    counts = {"cases": 0, "documents": 0, "job_paragraphs": 0, "app_paragraphs": 0}
    with conn.cursor() as cursor:
        for case in cases:
            case_id = upsert_case(cursor, case)
            counts["cases"] += 1

            job_text = required_text(case, "job_text")
            app_text = required_text(case, "app_text")
            if job_text:
                upsert_document(cursor, "job_documents", case_id, job_text)
                counts["documents"] += 1
            if app_text:
                upsert_document(cursor, "app_documents", case_id, app_text)
                counts["documents"] += 1

            counts["job_paragraphs"] += upsert_paragraphs(
                cursor,
                "job_paragraphs",
                case_id,
                validate_paragraphs(case_id, case.get("job_paragraphs", []), "job_paragraphs"),
            )
            counts["app_paragraphs"] += upsert_paragraphs(
                cursor,
                "app_paragraphs",
                case_id,
                validate_paragraphs(case_id, case.get("app_paragraphs", []), "app_paragraphs"),
            )
    conn.commit()
    counts["paragraph_matches"] = persist_paragraph_matches(
        conn, model, [case["case_id"] for case in cases], top_n_matches
    )
    conn.commit()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-n-matches", type=int, default=DEFAULT_TOP_N_MATCHES)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Inputfilen findes ikke: {args.input}")

    cases = load_cases(args.input)
    model = SentenceTransformer(args.model)
    conn = psycopg2.connect(args.dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(CREATE_TABLES_SQL)
        conn.commit()
        counts = persist(conn, cases, model, args.top_n_matches)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"Færdig: {counts['cases']} cases importeret.")
    print(f"Dokumenter opdateret: {counts['documents']}")
    print(f"Job-paragraffer opdateret: {counts['job_paragraphs']}")
    print(f"Ansøgnings-paragraffer opdateret: {counts['app_paragraphs']}")
    print(f"Paragrafmatches opdateret: {counts['paragraph_matches']}")


if __name__ == "__main__":
    main()
