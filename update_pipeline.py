"""Process new case folders and reconcile the persisted retrieval corpus.

Run from the repository root with:
    python update_pipeline.py [--dry-run]
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

import psycopg2
from sentence_transformers import SentenceTransformer

import build_faiss_indexes
import chunkify_semantic
import clean_and_canonize
import extract_job_and_app_data
import persist_tags_taxonomy
import persist_texts_paragraphs
import tag_assignment
from case_identity import case_id_from_name, normalize_case_name


def configured_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_case_list(path: Path) -> List[Dict[str, Any]]:
    data = extract_job_and_app_data.load_existing_cases(path)
    return data


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, path)


def validate_case_identities(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_identity: Dict[str, Dict[str, Any]] = {}
    by_case_id: Dict[str, str] = {}
    for case in cases:
        name = case.get("case")
        identity = normalize_case_name(name)
        if identity in by_identity:
            raise ValueError(
                f"Duplicate case folder identity in extracted data: {name!r}."
            )
        case_id = case_id_from_name(name)
        previous_identity = by_case_id.get(case_id)
        if previous_identity is not None and previous_identity != identity:
            raise ValueError(
                f"Stable case ID collision for {name!r} and "
                f"{by_identity[previous_identity]['case']!r}."
            )
        case["case_id"] = case_id
        by_identity[identity] = case
        by_case_id[case_id] = identity
    return by_identity


def find_new_case_folders(
    case_root: Path,
    existing_cases: Dict[str, Dict[str, Any]],
) -> List[Path]:
    folders = sorted(path for path in case_root.iterdir() if path.is_dir())
    seen_folders: Dict[str, str] = {}
    seen_case_ids = {
        case["case_id"]: identity for identity, case in existing_cases.items()
    }
    new_folders = []
    for folder in folders:
        identity = normalize_case_name(folder.name)
        previous_name = seen_folders.get(identity)
        if previous_name is not None:
            raise ValueError(
                f"Source folders {previous_name!r} and {folder.name!r} have "
                "the same case-insensitive identity."
            )
        seen_folders[identity] = folder.name
        case_id = case_id_from_name(folder.name)
        previous_identity = seen_case_ids.get(case_id)
        if previous_identity is not None and previous_identity != identity:
            raise ValueError(
                f"Stable case ID collision for source folders "
                f"{seen_folders[previous_identity]!r} and {folder.name!r}."
            )
        seen_case_ids[case_id] = identity

        existing = existing_cases.get(identity)
        if existing is not None:
            if existing["case"] != folder.name:
                raise ValueError(
                    f"Source folder {folder.name!r} differs only by case or "
                    f"normalization from existing case {existing['case']!r}."
                )
            continue
        new_folders.append(folder)
    return new_folders


def is_complete_case_folder(folder: Path) -> bool:
    return bool(
        extract_job_and_app_data.find_matching_files(folder, ["*opslag*"])
        and extract_job_and_app_data.find_matching_files(
            folder, ["*ansøgning*", "*cover*"]
        )
    )


def run_full_refresh(
    extracted_cases: List[Dict[str, Any]],
    cleaned_path: Path,
    segmented_path: Path,
    taxonomy_path: Path,
    embedding_model_path: Path,
    dsn: str,
    faiss_dir: Path,
    top_n_matches: int,
) -> None:
    model = SentenceTransformer(str(embedding_model_path))

    print("Rengør seneste fulde extracted corpus ...")
    cleaned_cases = clean_and_canonize.cleanup(extracted_cases, model)
    write_json_atomic(cleaned_path, cleaned_cases)

    print("Gendanner semantiske paragraffer for hele korpusset ...")
    segmented_cases = chunkify_semantic.paragraphize(cleaned_cases, model)
    write_json_atomic(segmented_path, segmented_cases)

    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"Taksonomifilen findes ikke: {taxonomy_path}")
    taxonomy = persist_tags_taxonomy.load_taxonomy(taxonomy_path)

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cursor:
            cursor.execute(persist_texts_paragraphs.CREATE_TABLES_SQL)
        conn.commit()
        counts = persist_texts_paragraphs.persist(
            conn, segmented_cases, model, top_n_matches
        )
        print(
            "PostgreSQL opdateret: "
            f"{counts['cases']} cases, "
            f"{counts['job_paragraphs']} job-paragraffer, "
            f"{counts['app_paragraphs']} ansøgnings-paragraffer, "
            f"{counts['paragraph_matches']} krav-svar-matches."
        )

        persist_tags_taxonomy.create_tables(conn)
        conn.commit()
        tag_count, seed_count = persist_tags_taxonomy.persist_taxonomy(
            conn, taxonomy, taxonomy_path
        )
        print(
            f"Taksonomi opdateret: {tag_count} tags og {seed_count} seed-eksempler."
        )

        with conn.cursor() as cursor:
            cursor.execute(tag_assignment.CREATE_TAG_TABLES_SQL)
        conn.commit()
        flat_tags = tag_assignment.fetch_tags_from_db(conn)
        tag_keys, categories, prototypes = tag_assignment.build_tag_prototypes(
            model, flat_tags
        )
        targets = tag_assignment.fetch_targets(conn)
        if targets:
            embeddings = tag_assignment.embed_texts(
                model, [target[2] for target in targets]
            )
            assignments = tag_assignment.assign_tags_to_embeddings(
                embeddings,
                tag_keys,
                categories,
                prototypes,
                top_k_per_category=tag_assignment.DEFAULT_TOP_K_PER_CATEGORY,
                min_score=tag_assignment.DEFAULT_MIN_SCORE,
            )
        else:
            assignments = []
        assignment_count = tag_assignment.persist_all_text_tags(
            conn, targets, assignments
        )
        conn.commit()
        print(
            f"Fuld tagging gennemført: {assignment_count} tag-relationer "
            f"for {len(targets)} tekster."
        )

        build_faiss_indexes.ensure_faiss_id_map(conn)
        build_faiss_indexes.build_indexes(
            conn,
            model,
            faiss_dir,
            str(embedding_model_path),
            batch_size=32,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_update(args: argparse.Namespace) -> None:
    case_root = configured_path(args.case_root, ROOT)
    extracted_path = configured_path(args.extracted, ROOT)
    cleaned_path = configured_path(args.cleaned, ROOT)
    segmented_path = configured_path(args.segmented, ROOT)
    taxonomy_path = configured_path(args.taxonomy, ROOT)
    embedding_model_path = configured_path(args.model, ROOT)
    faiss_dir = configured_path(args.faiss_dir, ROOT)

    for path in (case_root, extracted_path, segmented_path, taxonomy_path):
        if not path.exists():
            raise FileNotFoundError(f"Forventet input findes ikke: {path}")
    if not case_root.is_dir():
        raise NotADirectoryError(f"Kildestien er ikke en mappe: {case_root}")

    cases = load_case_list(extracted_path)
    existing_cases = validate_case_identities(cases)
    new_folders = find_new_case_folders(case_root, existing_cases)
    complete_folders = [folder for folder in new_folders if is_complete_case_folder(folder)]
    incomplete_folders = [folder for folder in new_folders if folder not in complete_folders]

    print(f"Kendte cases i extracted JSON: {len(cases)}")
    print(f"Nye mapper fundet: {len(new_folders)}")
    for folder in incomplete_folders:
        print(f"Springer over indtil begge filer findes: {folder.name}")
    print(f"Nye komplette par, der kan behandles: {len(complete_folders)}")

    if args.dry_run:
        print("Dry-run: ingen filer, database eller indeks er ændret.")
        return

    if not args.dsn:
        raise ValueError("POSTGRES_DSN eller --dsn skal være angivet.")
    if not embedding_model_path.exists():
        raise FileNotFoundError(f"Embedding-modellen findes ikke: {embedding_model_path}")
    if complete_folders and not extract_job_and_app_data.OllamaURL:
        raise ValueError("OLLAMA_URL skal være angivet for at behandle nye mapper.")

    for folder in complete_folders:
        print(f"Ekstraherer nyt case-par: {folder.name}")
        case = extract_job_and_app_data.process_case_folder(folder)
        if case is None:
            raise ValueError(
                f"Case-mappen {folder.name!r} blev ufuldstændig under ekstraktion."
            )
        case["case_id"] = case_id_from_name(folder.name)
        existing_cases[normalize_case_name(folder.name)] = case

    ordered_cases = sorted(existing_cases.values(), key=lambda case: normalize_case_name(case["case"]))
    write_json_atomic(extracted_path, ordered_cases)
    print(f"Opdateret extracted JSON med {len(complete_folders)} nye case-par.")

    run_full_refresh(
        ordered_cases,
        cleaned_path,
        segmented_path,
        taxonomy_path,
        embedding_model_path,
        args.dsn,
        faiss_dir,
        args.top_n_matches,
    )
    print("Opdateringen er gennemført.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", default=os.environ.get("EKSPORT_ROOT", ""))
    parser.add_argument(
        "--extracted",
        default=os.environ.get("EXTRACTED_SENTENCES_JSON", "extracted_sentences.json"),
    )
    parser.add_argument(
        "--cleaned",
        default=os.environ.get("CLEANED_SENTENCES_JSON", "cleaned_sentences.json"),
    )
    parser.add_argument(
        "--segmented",
        default=os.environ.get(
            "SEGMENTED_AND_READY_JSON", "segmented_and_ready.json"
        ),
    )
    parser.add_argument(
        "--taxonomy",
        default=os.environ.get("TAGS_TAXONOMY_JSON", "tags_taxonomy.json"),
    )
    parser.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN"))
    parser.add_argument("--model", default=os.environ.get("EMBEDDING_MODEL", ""))
    parser.add_argument(
        "--faiss-dir", default=os.environ.get("FAISS_DIR", "faiss_indices")
    )
    parser.add_argument(
        "--top-n-matches",
        type=int,
        default=persist_texts_paragraphs.DEFAULT_TOP_N_MATCHES,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_n_matches < 1:
        raise ValueError("--top-n-matches skal være mindst 1.")
    if not args.case_root:
        raise ValueError("EKSPORT_ROOT eller --case-root skal være angivet.")
    if not args.model:
        raise ValueError("EMBEDDING_MODEL eller --model skal være angivet.")
    run_update(args)


if __name__ == "__main__":
    main()
