import json, os
import re
import sys
import faiss
from typing import List
from sentence_transformers import SentenceTransformer
"""Document Ingestion: rule-based noise removal and canonicalization (case_id assignment)."""

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
    r"(S|s)enior",
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


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv(".env")
    except ImportError:
        print("Import of python-dotenv failed. python-dotenv ikke installeret.")
        sys.exit(1)

    print("imma clean and canonize !")

    import_file = str(os.getenv("EXTRACTED_SENTENCES_JSON"))
    export_file = str(os.getenv("CLEANED_SENTENCES_JSON"))
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
    export_json = cleanup(existing_cases_json, embeddingmodel)
    with open(export_file, "w", encoding="utf-8") as export_file_content:
        json.dump(export_json, export_file_content, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
