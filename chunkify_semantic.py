import copy
import json, os
import sys
import faiss
import numpy as np
from typing import Sequence, List
from sentence_transformers import SentenceTransformer
"""Semantic Paragraph Chunking"""

def paragraphize(cases_json: List[dict], embeddingmodel: SentenceTransformer):
    # Implement the paragraphization logic here

    def find_local_minima(values: Sequence[float], threshold: float = None) -> List[int]:
        minima = []
        n = len(values)
        for i in range(1, n - 1):
            center = values[i]
            left = values[i - 1]
            right = values[i + 1]

            if center < left and center < right:
                if threshold is None or center <= threshold:
                    minima.append(i)
        return minima

    def moving_average(values: Sequence[float], smooth_window: int = 5) -> List[float]:
        if smooth_window < 1:
            raise ValueError("smooth_window must be >= 1")
        if len(values) < smooth_window:
            return list(values)

        smoothed = []
        half = smooth_window // 2

        # Compute the moving average for each position in the sequence
        for i in range(len(values)):
            start = max(0, i - half)
            end = min(len(values), i + half + 1)
            smoothed.append(sum(values[start:end]) / (end - start))

        return smoothed

    def consecutive_semantic_coherence( embeddings: np.ndarray = None) -> List[float]:
        gaps = []
        for idx in range(1, len(embeddings)):
                semantic_gap = 0.0
                if embeddings is not None:
                    semantic_gap = float(np.dot(embeddings[idx - 1], embeddings[idx]))
                gaps.append(semantic_gap)
        return gaps


    def paragraphize_sentences(sentences, source, case_id):
        # here is where the magic happens
        embeddings = embeddingmodel.encode([s for s in sentences], convert_to_numpy=True, show_progress_bar=True).astype("float32")
        if embeddings.ndim == 1:
            # hvis embedding er en enkelt vektor, reshape den til 2d matrix
            embeddings = embeddings.reshape(1, -1)
        # normaliserer vektorerne for at kunne bruge dot-produkt som cosine-similarity
        faiss.normalize_L2(embeddings)
        # opretter en FAISS-indeks med dot-produkt som målemetode
        #print("Computing semantic gaps between consecutive sentences")
        semantic_gaps = consecutive_semantic_coherence(embeddings)
        #print(f"Semantic gaps: {semantic_gaps}")

        local_minimas = find_local_minima(moving_average(semantic_gaps), 5)
        #print(f"Local minima (potential paragraph cutoffs): {local_minimas}")

        current_paragraph_id_number = 0

        current_paragraph = {"source": source, "case_id": case_id}
        paragraphs = []
        lastSID = 0
        for sID, sentence in enumerate(sentences):
            if sID in local_minimas or sID == len(sentences) - 1:
                paragraf_id = f"{case_id}_{source}_p{current_paragraph_id_number}"
                current_paragraph["paragraph_id"] = paragraf_id
                current_paragraph["text"] = " ".join([s for s in sentences[lastSID:sID+1]])
                paragraphs.append(copy.deepcopy(current_paragraph))
                current_paragraph_id_number += 1
                lastSID = sID+1
        
        return paragraphs
    
    for case_json in cases_json:
        case_json["job_paragraphs"] = paragraphize_sentences(case_json["job_sentences"], "job", case_json["case_id"])
        case_json["app_paragraphs"] = paragraphize_sentences(case_json["app_sentences"], "app", case_json["case_id"])
        del case_json["job_sentences"]
        del case_json["app_sentences"]
    #print(f"Paragraphized cases.{json.dumps(cases_json[:1], ensure_ascii=False, indent=2) }")
    return cases_json

def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
    except ImportError:
        print("Import of python-dotenv failed. python-dotenv ikke installeret.")
        sys.exit(1)       

    print("imma semantically chunkify !")

    import_file = str(os.getenv("CLEANED_SENTENCES_JSON"))
    export_file = str(os.getenv("SEGMENTED_AND_READY_JSON"))
    export_json = {}
    existing_cases_json = []
    with open(import_file, "r", encoding="utf-8") as import_file_content:
        if os.stat(import_file).st_size == 0:
            print(f"Error: '{import_file}' empty")
            raise SystemExit(1)
        else:
            try:
                existing_cases_json = json.load(import_file_content)
            except json.JSONDecodeError:
                print(f"Error: '{import_file}' is not a valid JSON file.")
                raise SystemExit(1)

    print("loading sentence transformer library")
    embeddingmodelLocation = "./"+str(os.getenv("EMBEDDING_MODEL"))
    print(f"loading embedding model: {embeddingmodelLocation}")

    embeddingmodel = SentenceTransformer(embeddingmodelLocation)
    if embeddingmodel is None:
        print("Error: Failed to load embedding model.")
        raise SystemExit(1)

    print("Starting paragraphization")
    export_json = paragraphize(existing_cases_json, embeddingmodel)
    with open(export_file, "w", encoding="utf-8") as export_file_content:
        json.dump(export_json, export_file_content, ensure_ascii=False, indent=4)
            
if __name__ == "__main__":
    main()
