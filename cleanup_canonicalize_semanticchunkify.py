import copy
import json, os
import re
import sys
import faiss
import numpy as np
from typing import Sequence, List
from sentence_transformers import SentenceTransformer
"""Document Ingestion & Semantic Paragraph Chunking"""

remove_phrases = [
    r"(S|s)end (os |venligst )?(din |en )?ansøgning",
    r"(V|v)i glæder os til at høre fra dig",
    r"(H|h)ar du spørgsmål, ",
    r"(A|a)nsøgningsfrist",
    r"(S|s)narest muligt",
    r"(K|k)ontakt os (på|hvis)",
    r"(T|t)lf\.?\s*\+?\d",
    r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})",
    r"(B|b)est regards,",
    r"(V|v)elkommen til at skrive til os",
    r"(K|k)ontakt os (på|hvis)",
    r"(H|h)ilsen",
    r"(M|m)ed venlig hilsen",
    r"(F|f)or yderligere information, ",
    r"(D|d)u er velkommen til at kontakte os",
    r"(T|t)ak for din interesse",
    r"(L|l)æs mere ",
    r"@\w+\.(dk|com)",
    r"løbende samtaler",
    r"(B|b)liv en del af",
    r"(D|d)u vil være med til",
    r"(S|s)end os en ansøgning",
    r"(D|d)in profil",
    r"(V|v)i ? ansøgninger ? løbende",
    r"frokostordning|massageordning|pensionsordning",
    r"uanset køn, alder, religion",
    r"(C|c)pr-nummer",
    r"(H|h)ar du spørgsmål",
    r"(D|d)u har spørgsmål",
    r"(D|d)it CV ",
    r"(D|d)in ansøgning",
    r"(C|c)over letter",
    r"(R|r)esumé",
    r"(J|j)ob application",
    r"(A|a)pplication form",
    r"(R|r)eference letter",
    r"(R|r)ecommendation letter",
    r"(D|d)u kan læse mere"
]

def cleanup(cases_json: List[dict], embeddingmodel: SentenceTransformer):
    # noise removal, fjerner generisk tekst, som ofte optræder i jobopslag og ansøgninger fra casens tekst, 
    # både i hele jobopslagsteksten og  ansøgningsteksten, 
    # samt de segmenterede sætninger, der senere skal samles til paragraffer.
    def cleanup_sentences(sentences):
        cleaned_sentences = []
        for id, sentence in enumerate(sentences):
            if not any(re.search(pattern, sentence) for pattern in remove_phrases):
                cleaned_sentences.append(sentence.strip())
            else:
                pass
                #print(f"Removing sentence: {sentence}")
        return cleaned_sentences


    for case_id, case_json in enumerate(cases_json):
        case_json["case_id"] = "c"+str(case_id)
        case_json["job_sentences"] = cleanup_sentences(case_json.get("job_sentences"))
        case_json["app_sentences"] = cleanup_sentences(case_json.get("app_sentences"))
        for expr in remove_phrases:
            case_json["job_text"] = re.sub(expr, " [] ", case_json.get("job_text"))
            case_json["app_text"] = re.sub(expr, " [] ", case_json.get("app_text"))
        


    alle_sætninger = []
    flatID_to_caseID_mapping = [] #mapping fra flat array indeksering til json struktur, 
    for case_id, case_json in enumerate(cases_json):
        for i, sentence in enumerate(case_json.get("job_sentences")):
            alle_sætninger.append(sentence)
            flatID_to_caseID_mapping.append({"case_id": case_id, "origin": "job_sentences", "sentence_id": i})
    #print("length of sentences:", len(alle_sætninger))
    #print("length of flatID_to_caseID_mapping:", len(flatID_to_caseID_mapping))
    
    textembeddings = embeddingmodel.encode(alle_sætninger, convert_to_numpy=True).astype("float32")

    faiss.normalize_L2(textembeddings)
    faissindex = faiss.IndexFlatIP(textembeddings.shape[1])
    faissindex.add(textembeddings)

    sim_threshold=0.85
    k = 50  # k er antal nearest neighbors der skal returneres, 50 -> søg bredt for at fange tværgående matches
    distances, neighbours_ids = faissindex.search(textembeddings, k)
    num_cases = len(cases_json)
    #gennemgår alle sætningerne
    for i in range(len(alle_sætninger)):
        # for hver sætning gennemgå de nærmeste (k) naboer (nearest neighbors), samt afstanden, som Faiss har beregnet for dem
        # og tæl hvor mange af de nærmeste naboer der har en afstand over sim_threshold - der ligner hinanden meget
        matched_cases = set()
        for distance, neighbour_id in zip(distances[i], neighbours_ids[i]):
            if distance >= sim_threshold:
                matched_cases.add(alle_sætninger[neighbour_id])
        distinctiveness = 1.0 - (len(matched_cases) / num_cases)
        if distinctiveness < 0.9:
            #print(f"Distinctiveness: {distinctiveness}, matches: {len(matched_cases)}, Index: {i}, sentence: {alle_sætninger[i]},  matched_cases: {matched_cases}")
            case_id = flatID_to_caseID_mapping[i]["case_id"]
            origin = flatID_to_caseID_mapping[i]["origin"]
            sentence_id = flatID_to_caseID_mapping[i]["sentence_id"]
            #print(f"Cleaned sentence \"{cases_json[case_id][origin][sentence_id]}\" from case_id {case_id}, origin {origin}")

            del cases_json[case_id][origin][sentence_id]
            
    return cases_json
    


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

    print("imma cleanup canonicalize semantically chunkify !")

    import_file = str(os.getenv("EXTRACTED_SENTENCES_JSON"))
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

    print("Starting similarity search")
    print("loading sentence transformer library")
    embeddingmodelLocation = "./"+str(os.getenv("EMBEDDING_MODEL"))
    print(f"loading embedding model: {embeddingmodelLocation}")

    embeddingmodel = SentenceTransformer(embeddingmodelLocation)
    if embeddingmodel is None:
        print("Error: Failed to load embedding model.")
        raise SystemExit(1)

    print("Starting cleanup")
    cleanup(existing_cases_json, embeddingmodel)
    print("Starting paragraphization")
    export_json = paragraphize(existing_cases_json, embeddingmodel)
    with open(export_file, "w", encoding="utf-8") as export_file_content:
        json.dump(export_json, export_file_content, ensure_ascii=False, indent=4)
            
if __name__ == "__main__":
    main()
