"""
persist_tags_taxonomy.py
========================

Persisterer tags_taxonomy.json i PostgreSQL.

Scriptet opretter automatisk disse tabeller, hvis de ikke findes:

* taxonomy_metadata: version og beskrivelse af den importerede taksonomi
* tags: topkategori og tag-definition
* tag_seeds: alle seed_examples med stabil rækkefølge pr. tag

Importen er idempotent. Det er derfor sikkert at køre scriptet igen efter
ændringer i tags_taxonomy.json.

Eksempler:
    python persist_tags_taxonomy.py
    python persist_tags_taxonomy.py --taxonomy tags_taxonomy.json
    python persist_tags_taxonomy.py --dsn "dbname=jobapp_rag user=postgres"

Alternativt kan POSTGRES_DSN og TAGS_TAXONOMY_JSON angives i miljøvariabler
eller i .env.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import psycopg2
import psycopg2.extras

try:
    from dotenv import load_dotenv

    load_dotenv(".env")
except ImportError:
    pass


DEFAULT_DSN = os.environ.get(
    "POSTGRES_DSN",
    "dbname=jobapp_rag user=postgres host=localhost port=5432",
)
DEFAULT_TAXONOMY = os.environ.get("TAGS_TAXONOMY_JSON", "tags_taxonomy.json")


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS taxonomy_metadata (
    taxonomy_name    TEXT PRIMARY KEY,
    taxonomy_version TEXT NOT NULL,
    embedding_note   TEXT,
    source_file      TEXT,
    imported_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tags (
    id              SERIAL PRIMARY KEY,
    tag_key         TEXT NOT NULL UNIQUE,
    top_category    TEXT NOT NULL,
    taxonomy_version TEXT NOT NULL DEFAULT '1.0',
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tags_top_category
    ON tags(top_category);

CREATE TABLE IF NOT EXISTS tag_seeds (
    id              SERIAL PRIMARY KEY,
    tag_key         TEXT NOT NULL REFERENCES tags(tag_key) ON DELETE CASCADE,
    seed_index      INTEGER NOT NULL,
    seed_text       TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE(tag_key, seed_index),
    UNIQUE(tag_key, seed_text)
);

CREATE INDEX IF NOT EXISTS idx_tag_seeds_tag_key
    ON tag_seeds(tag_key);
"""


def load_taxonomy(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        taxonomy = json.load(file)

    if not isinstance(taxonomy, dict):
        raise ValueError("Taksonomien skal være et JSON-objekt.")
    if not isinstance(taxonomy.get("categories"), dict):
        raise ValueError("Taksonomien skal indeholde objektet 'categories'.")
    return taxonomy


def flatten_taxonomy(
    taxonomy: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, int, str]]]:
    """Returnerer tag-rækker og seed-rækker til PostgreSQL."""
    tags: List[Tuple[str, str, str]] = []
    seeds: List[Tuple[str, int, str]] = []

    for top_category, category in taxonomy["categories"].items():
        if not isinstance(category, dict) or not isinstance(category.get("tags"), dict):
            raise ValueError(
                f"Kategorien {top_category!r} skal indeholde et 'tags'-objekt."
            )

        for tag_key, tag_data in category["tags"].items():
            if not isinstance(tag_data, dict):
                raise ValueError(f"Tagget {tag_key!r} skal være et JSON-objekt.")

            seed_examples = tag_data.get("seed_examples")
            if not isinstance(seed_examples, list) or not seed_examples:
                raise ValueError(f"Tagget {tag_key!r} har ingen seed_examples.")
            if not all(isinstance(seed, str) and seed.strip() for seed in seed_examples):
                raise ValueError(
                    f"Alle seed_examples for {tag_key!r} skal være ikke-tomme tekststrenge."
                )

            tags.append((tag_key, top_category, taxonomy.get("taxonomy_version", "1.0")))
            seeds.extend(
                (tag_key, seed_index, seed_text)
                for seed_index, seed_text in enumerate(seed_examples)
            )

    return tags, seeds


def create_tables(conn) -> None:
    with conn.cursor() as cursor:
        cursor.execute(CREATE_TABLES_SQL)


def persist_taxonomy(
    conn,
    taxonomy: Dict[str, Any],
    taxonomy_path: Path,
) -> Tuple[int, int]:
    tags, seeds = flatten_taxonomy(taxonomy)
    taxonomy_version = taxonomy.get("taxonomy_version", "1.0")
    taxonomy_name = taxonomy_path.stem

    with conn.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO taxonomy_metadata
                (taxonomy_name, taxonomy_version, embedding_note, source_file, imported_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (taxonomy_name) DO UPDATE SET
                taxonomy_version = EXCLUDED.taxonomy_version,
                embedding_note = EXCLUDED.embedding_note,
                source_file = EXCLUDED.source_file,
                imported_at = now()
            """,
            (
                taxonomy_name,
                taxonomy_version,
                taxonomy.get("embedding_note"),
                str(taxonomy_path.resolve()),
            ),
        )

        psycopg2.extras.execute_values(
            cursor,
            """
            INSERT INTO tags (tag_key, top_category, taxonomy_version)
            VALUES %s
            ON CONFLICT (tag_key) DO UPDATE SET
                top_category = EXCLUDED.top_category,
                taxonomy_version = EXCLUDED.taxonomy_version
            """,
            tags,
        )

        # Fjern tags/seeds, som ikke længere findes i JSON-filen. Dette holder
        # databasen synkroniseret og undgår forældede tags ved senere imports.
        current_tag_keys = [tag[0] for tag in tags]
        cursor.execute(
            "DELETE FROM tags WHERE NOT (tag_key = ANY(%s))",
            (current_tag_keys,),
        )

        psycopg2.extras.execute_values(
            cursor,
            """
            INSERT INTO tag_seeds (tag_key, seed_index, seed_text)
            VALUES %s
            ON CONFLICT (tag_key, seed_index) DO UPDATE SET
                seed_text = EXCLUDED.seed_text
            """,
            seeds,
        )

        cursor.execute(
            """
            DELETE FROM tag_seeds seed
            WHERE NOT EXISTS (
                SELECT 1
                FROM jsonb_array_elements(%s::jsonb) AS expected(value)
                WHERE (expected.value->>0) = seed.tag_key
                  AND (expected.value->>1)::integer = seed.seed_index
            )
            """,
            (json.dumps(seeds),),
        )

    conn.commit()
    return len(tags), len(seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path(DEFAULT_TAXONOMY),
        help="Sti til tags_taxonomy.json.",
    )
    parser.add_argument(
        "--dsn",
        default=DEFAULT_DSN,
        help="PostgreSQL DSN. Overskriver POSTGRES_DSN.",
    )
    args = parser.parse_args()

    if not args.taxonomy.exists():
        raise FileNotFoundError(f"Taksonomifilen findes ikke: {args.taxonomy}")

    taxonomy = load_taxonomy(args.taxonomy)
    print(f"Indlæser taksonomi: {args.taxonomy}")

    conn = psycopg2.connect(args.dsn)
    try:
        create_tables(conn)
        conn.commit()
        tag_count, seed_count = persist_taxonomy(conn, taxonomy, args.taxonomy)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"Færdig: {tag_count} tags og {seed_count} seed-eksempler persisteret.")


if __name__ == "__main__":
    main()
