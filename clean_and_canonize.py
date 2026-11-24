import json, os
import re
import sys
from typing import List
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
    r"(R|r)eference letter",
    r"(R|r)ecommendation letter",
    r"(D|d)u kan læse mere"
]

def cleanup(cases_json: List[dict]):
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
            case_json["job_text"] = re.sub(expr, " ", case_json.get("job_text"))
            case_json["app_text"] = re.sub(expr, " ", case_json.get("app_text"))

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

    print("Starting cleanup")
    export_json = cleanup(existing_cases_json)
    with open(export_file, "w", encoding="utf-8") as export_file_content:
        json.dump(export_json, export_file_content, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
