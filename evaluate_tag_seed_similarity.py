"""
evaluate_tag_seed_similarity.py
================================

Tester en liste af sætninger mod ALLE individuelle ``seed_examples`` i
tags_taxonomy.json. For hver input-sætning rapporteres:

* bedste seed og score for hvert tag
* de bedste tags på tværs af hele taksonomien

Inputfilen skal enten være:

1. En JSON-liste med strenge:
       ["Jeg har erfaring med Python.", "Jeg trives i Scrum-teams."]
2. En almindelig tekstfil med én sætning pr. linje.

Eksempel:
    python evaluate_tag_seed_similarity.py sentences.json --top-k 10
    python evaluate_tag_seed_similarity.py sentences.txt --min-score 0.30

Scriptet bruger samme embedding-model som resten af pipeline'en via
EMBEDDING_MODEL. Similarity er cosine similarity, fordi alle vektorer
L2-normaliseres før dot-produktet beregnes.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "./sentence_transformers_embedding_model"
DEFAULT_SENTENCE_DATA = "extracted_sentences.json"
DEFAULT_TAXONOMY_FILE = "tags_taxonomy.json"
DEFAULT_MIN_SCORE = 0.40
DEFAULT_TOP_K = 3

def load_sentences(file: str) -> List[str]:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    sentences: List[str] = []
    if file.lower().endswith(".json"):
        data = json.loads(text)
        # Understøt både en simpel liste af strenge og case-formatet fra
        # extracted_sentences.json, hvor hvert element er en sentence-record.
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            sentences.extend(data)
        elif isinstance(data, dict):
            cases = data.values()
            for case in cases:
                sentences.extend(
                    item["text"] if isinstance(item, dict) else item
                    for item in case.get("job_sentences", [])
                )
                sentences.extend(
                    item["text"] if isinstance(item, dict) else item
                    for item in case.get("app_sentences", [])
                )
        elif isinstance(data, list):
            for case in data:
                sentences.extend(
                    item["text"] if isinstance(item, dict) else item
                    for item in case.get("job_sentences", [])
                )
                sentences.extend(
                    item["text"] if isinstance(item, dict) else item
                    for item in case.get("app_sentences", [])
                )
    if not sentences:
        raise ValueError(f"Ingen sætninger fundet i {file}.")
    if not all(isinstance(sentence, str) for sentence in sentences):
        raise ValueError("Alle sentences skal være tekststrenge eller records med et 'text'-felt.")
    return [sentence for sentence in sentences if sentence.strip()]

def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


def evaluate_and_update_tax_tags(
    model: SentenceTransformer,
    sentences: List[str],
    tag_file: str,
    top_k: int,
    min_score: float,
) -> List[Dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k skal være mindst 1.")

    sentence_embeddings = normalize_rows(
        model.encode(
            sentences,
            batch_size=32,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")
    )
    
    with open(tag_file, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    results: List[Dict[str, Any]] = []
    for top_category, category in taxonomy["categories"].items():
        for tag_key, tag_data in category["tags"].items():
            seeds = tag_data.get("seed_examples", [])

            if not seeds:
                raise ValueError(f"Tagget {tag_key!r} har ingen seed_examples.")
            
            seed_embeddings = normalize_rows(
                model.encode(
                    seeds,
                    convert_to_numpy=True,
                    show_progress_bar=True,
                ).astype("float32")
            )
            scores = seed_embeddings @ sentence_embeddings.T

            for seed_index, seed in enumerate(seeds):
                ranked_indices = np.argsort(scores[seed_index])[::-1]
                matches = []
                for sentence_index in ranked_indices[:top_k]:
                    score = float(scores[seed_index, sentence_index])
                    if score < min_score:
                        continue
                    matches.append(
                        {
                            "sentence_index": int(sentence_index),
                            "sentence": sentences[int(sentence_index)],
                            "similarity": score,
                        }
                    )

                results.append(
                    {
                        "tag_key": tag_key,
                        "top_category": top_category,
                        "seed_index": seed_index,
                        "seed": seed,
                        "matches": matches,
                    }
                )

    return results
def main() -> None:

    sentences = load_sentences(DEFAULT_SENTENCE_DATA)
    model = SentenceTransformer(DEFAULT_MODEL)
    result = evaluate_and_update_tax_tags(model, sentences, DEFAULT_TAXONOMY_FILE, DEFAULT_TOP_K, DEFAULT_MIN_SCORE)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
