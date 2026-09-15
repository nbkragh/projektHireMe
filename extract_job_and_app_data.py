from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import fnmatch
import sys
import warnings
import requests

"""Extract job and app sentence data from HTML and plain text files. Called Ingestion process."""

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip = False
        self.in_p = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip = True
        if tag == "p":
            self.in_p = True

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = False
        if tag == "p":
            self.in_p = False

    def handle_comment(self, data):
        pass

    def handle_data(self, data):
        if not self.skip and self.in_p:
            stripped = data.strip()
            if stripped:
                self.text_parts.append(stripped)

    def get_text(self) -> list[str]:
        return self.text_parts


# Indlæs .env fra projektmappen hvis python-dotenv er installeret
try:
    from dotenv import load_dotenv
    load_dotenv(".env")
except ImportError:
    print("python-dotenv ikke installeret. Sæt miljøvariabler manuelt.")
    pass  # Kør uden dotenv – sæt miljøvariabler manuelt i stedet

LLMmodel=os.environ.get("OLLAMA_MODEL", "DEFAULT_MODEL")
OllamaURL=os.environ.get("OLLAMA_URL")
result_file=os.environ.get("EXTRACTED_SENTENCES_JSON", "extracted_sentences.json")
eksport_root=os.environ.get("EKSPORT_ROOT")

def extract_text_from_html(file_path: Path) -> list[str]:
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    parser = TextExtractor()
    parser.feed(content)
    text_parts = parser.get_text()
    lines = [x.replace("\n", "") for x in text_parts if not x.startswith("URL")]
    return " ".join(lines)


def extract_text_from_plain(file_path: Path) -> str:
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    lines = [x for x in lines if not x.startswith("URL")]
    return " ".join(lines)

def extract_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".html":
        return extract_text_from_html(file_path)
    return extract_text_from_plain(file_path)


def find_matching_files(folder: Path, patterns, extensions=None):
    matches = []
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if extensions and path.suffix.lower() not in extensions:
            continue
        name_lower = path.name.lower()
        if any(fnmatch.fnmatch(name_lower, pat.lower()) for pat in patterns):
            matches.append(path)
    return sorted(matches)


def build_extraction_prompt(text: str) -> str:
    return f"""
    OPGAVE:
Din opgave er at analysere og opdele det nedenstående TEKST-segment, så resultatet opfylder de VIGTIGE KRAV.

TEKST:
{text}

VIGTIGE KRAV:
- En sætning er en komplet grammatisk enhed, der giver semantisk mening alene, som er afsluttet med ".", "?" eller "!".
- Opdel teksten i TEKST-segmentet ovenfor i sætninger, 
- Sætningerne skal være trimmed for whitespace.
- returner teksten nøjagtigt som den fremstår i kilden, ingen ændringer eller fortolkninger.
- returner IKKE tomme sætninger 
- returner IKKE sætninger der kun indeholder whitespace.
- returner IKKE sætninger der indeholder metadata.
- returner IKKE sætninger der indeholder URL'er.
- returner IKKE sætninger der indeholder kontaktoplysninger.
- returner IKKE sætninger der indeholder personnavne.
- returner IKKE sætninger der indeholder adresser.
- returner IKKE sætninger der indeholder telefonnumre.
- returner IKKE sætninger der indeholder email-adresser.

AFLEVERING:
returner sætningerne i en valid json-liste
"""

def call_ollama_segmentation(
    text: str,
    model: str,
    ollama_url: str,
    timeout: int = 300,
) -> list[dict]:
    #print(ollama_url)
    prompt = build_extraction_prompt(text)
    ollama_response = requests.post(
        ollama_url,
        json={
            "model": model,
            "prompt": prompt,
            "system": "Du er en dygtig tekstsegmenteringsassistent. Din opgave er at analysere en given tekst og opdele den i sætninger",
            "stream": False,
            "format": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "options": {"temperature": 0.0},
        },
        timeout=timeout,
    )

#DEBUG printing!###############################################################################################
    ollama_response.raise_for_status()
    try:
        payload = ollama_response.json()
        payload.pop("context", {})
        message_content = payload.get("response", {})
        if not message_content:
            warnings.warn("Ollama returnerede tomt svar.")

        message_parsed = json.loads(message_content)
    except json.JSONDecodeError:
        warnings.warn("Ollama returnerede ikke gyldig JSON:")
        print("PAYLOAD:")
        print(payload)
        print("MESSAGE_CONTENT:")
        print(message_content)
        raise SystemExit(1)
    

    if not isinstance(message_parsed, list):
        warnings.warn("Ollama JSON-svar skal være en liste af sætninger.")

    return message_parsed


def process_case_folder(folder: Path) -> dict | None:
    """
    Udtrækker jobopslag + ansøgning fra én case-mappe og returnerer et
    result_case-dict, eller None hvis mappen mangler matchende filer.
    """
    job_patterns = ["*opslag*"]
    app_patterns = ["*ansøgning*", "*cover*"]

    job_files = find_matching_files(folder, job_patterns)
    app_files = find_matching_files(folder, app_patterns)

    if not job_files:
        print(f"  [{folder.name}] Ingen jobopslags-fil fundet. Springer over.")
        return None
    if not app_files:
        print(f"  [{folder.name}] Ingen ansøgnings-fil fundet. Springer over.")
        return None

    job_text = extract_text(job_files[0])
    app_text = extract_text(app_files[0])

    job_sentences = call_ollama_segmentation(job_text, LLMmodel, OllamaURL)
    app_sentences = call_ollama_segmentation(app_text, LLMmodel, OllamaURL)

    return {
        "case": str(os.path.basename(folder)),
        "job_text": job_text,
        "job_sentences": job_sentences,
        "app_text": app_text,
        "app_sentences": app_sentences,
    }


def load_existing_cases(path: Path) -> list:
    if not path.exists() or os.stat(path).st_size == 0:
        return []
    with open(path, "r", encoding="utf-8") as file_content:
        try:
            existing_cases_json = json.load(file_content)
        except json.JSONDecodeError:
            print(f"Error: '{path}' is not a valid JSON file.")
            raise SystemExit(1)

    if not isinstance(existing_cases_json, list):
        print(f"'{path}' does not contain a list. Overwriting with a new list.")
        return []
    return existing_cases_json


def upsert_case(existing_cases_json: list, result_case: dict) -> None:
    old_case = [case for case in existing_cases_json if case["case"] == result_case["case"]]
    if old_case:
        print(f"Case '{result_case['case']}' eksisterer allerede. Overskriver.")
        existing_cases_json.remove(old_case[0])
    else:
        print(f"Case '{result_case['case']}' eksisterer ikke. Tilføjer som ny case.")
    existing_cases_json.append(result_case)


def main():
    eksport_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(eksport_root)

    if not eksport_path.is_dir():
        print(f"Error: '{eksport_path}' is not a valid directory.")
        raise SystemExit(1)

    case_folders = sorted(p for p in eksport_path.iterdir() if p.is_dir())
    if not case_folders:
        print(f"Ingen undermapper fundet i '{eksport_path}'.")
        return

    result_path = Path(result_file)
    existing_cases_json = load_existing_cases(result_path)

    print(f"Fundet {len(case_folders)} case-mapper i '{eksport_path}'.")
    print(f"Using LLM model: {LLMmodel} @ {OllamaURL}")
    for idx, folder in enumerate(case_folders):
        print(f"\n=== Behandler case-mappe {idx+1}/{len(case_folders)}: {folder} ===")
        result_case = process_case_folder(folder)
        if result_case is None:
            continue
        upsert_case(existing_cases_json, result_case)

    with open(result_path, "w", encoding="utf-8") as file_content:
        json.dump(existing_cases_json, file_content, ensure_ascii=False, indent=2)

    print(f"\nFærdig. Alle cases skrevet til '{result_path}'.")


if __name__ == "__main__":
    main()
