"""
build_faiss_indexes.py
======================

Indlæser tekster fra PostgreSQL og bygger FAISS-indekser til:

* job_documents
* app_documents
* job_paragraphs
* app_paragraphs

FAISS indeholder kun normaliserede embeddings. PostgreSQL er fortsat
system-of-record, og faiss_id_map kobler hver indeksposition til den originale
databasepost.

Eksempel:
    python build_faiss_indexes.py
    python build_faiss_indexes.py --model sentence-transformers/all-MiniLM-L6-v2
    python build_faiss_indexes.py --faiss-dir data/faiss
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer

try:
    from dotenv import load_dotenv

    load_dotenv(".env")
except ImportError:
    pass


DEFAULT_DSN = os.environ.get(
    "POSTGRES_DSN",
)
DEFAULT_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
)
DEFAULT_FAISS_DIR = Path(os.environ.get("FAISS_DIR", "faiss_indexes"))

COLLECTIONS = {
    "job_documents": ("job_documents", "raw_text"),
    "app_documents": ("app_documents", "raw_text"),
    "job_paragraphs": ("job_paragraphs", "text"),
    "app_paragraphs": ("app_paragraphs", "text"),
}


def ensure_faiss_id_map(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faiss_id_map (
                id              SERIAL PRIMARY KEY,
                collection_name TEXT NOT NULL,
                source_table    TEXT NOT NULL,
                source_id       INTEGER NOT NULL,
                faiss_index     INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                created_at      TIMESTAMPTZ DEFAULT now(),
                UNIQUE (collection_name, faiss_index),
                UNIQUE (collection_name, source_table, source_id)
            )
            """
        )
    conn.commit()


def fetch_rows(
    conn, table: str, text_column: str
) -> List[Tuple[int, str]]:
    """Hent database-id og tekst i stabil rækkefølge."""
    with conn.cursor() as cursor:
        cursor.execute(
            f"SELECT id, {text_column} FROM {table} "
            f"WHERE {text_column} IS NOT NULL AND {text_column} <> '' ORDER BY id"
        )
        return [(int(row_id), text) for row_id, text in cursor.fetchall()]


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def persist_index(
    conn,
    faiss_dir: Path,
    collection_name: str,
    source_table: str,
    rows: List[Tuple[int, str]],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)

    index_path = faiss_dir / f"{collection_name}.faiss"
    temporary_path = faiss_dir / f"{collection_name}.faiss.tmp"
    faiss.write_index(index, str(temporary_path))
    os.replace(temporary_path, index_path)

    mappings = [
        (collection_name, source_table, source_id, faiss_index, model_name)
        for faiss_index, (source_id, _) in enumerate(rows)
    ]
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM faiss_id_map WHERE collection_name = %s",
            (collection_name,),
        )
        if mappings:
            psycopg2.extras.execute_values(
                cursor,
                """
                INSERT INTO faiss_id_map
                    (collection_name, source_table, source_id, faiss_index, embedding_model)
                VALUES %s
                """,
                mappings,
            )
    conn.commit()
    print(
        f"[{collection_name}] {len(rows)} poster -> "
        f"{index_path} (dimension={embeddings.shape[1]})"
    )


def build_indexes(
    conn,
    model: SentenceTransformer,
    faiss_dir: Path,
    model_name: str,
    batch_size: int,
) -> None:
    faiss_dir.mkdir(parents=True, exist_ok=True)

    for collection_name, (table, text_column) in COLLECTIONS.items():
        rows = fetch_rows(conn, table, text_column)
        if not rows:
            print(f"[{collection_name}] Ingen tekster; springer over.")
            continue

        embeddings = embed_texts(
            model,
            [text for _, text in rows],
            batch_size,
        )
        persist_index(
            conn,
            faiss_dir,
            collection_name,
            table,
            rows,
            embeddings,
            model_name,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--faiss-dir", type=Path, default=DEFAULT_FAISS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size skal være mindst 1.")

    print(f"Indlæser embedding-model: {args.model}")
    model = SentenceTransformer(args.model)

    conn = psycopg2.connect(args.dsn)
    try:
        ensure_faiss_id_map(conn)
        build_indexes(
            conn,
            model,
            args.faiss_dir,
            args.model,
            args.batch_size,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"Færdig. FAISS-indekser ligger i: {args.faiss_dir.resolve()}")


if __name__ == "__main__":
    main()
