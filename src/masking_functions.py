"""Core PII masking functions using different strategies (regex, spaCy, LLM)."""

import re
import spacy
from typing import Any, Dict, List, Optional, Tuple
from config import GROQ_API_KEY

# --- Regex-based masking ---
EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_REGEX = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"

def detect_and_mask_pii_regex(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Masks emails and phone numbers using regex and returns detected entities."""
    entities = []
    masked_text = text

    for match in re.finditer(EMAIL_REGEX, masked_text):
        entities.append({
            "text": match.group(0),
            "entity_type": "EMAIL",
            "start": match.start(),
            "end": match.end(),
        })
    masked_text = re.sub(EMAIL_REGEX, "[EMAIL]", masked_text)

    for match in re.finditer(PHONE_REGEX, masked_text):
        entities.append({
            "text": match.group(0),
            "entity_type": "PHONE",
            "start": match.start(),
            "end": match.end(),
        })
    masked_text = re.sub(PHONE_REGEX, "[PHONE]", masked_text)
    
    return masked_text, entities

# --- spaCy-based masking ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def detect_and_mask_pii_spacy(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Masks PII using spaCy NER and returns detected entities."""
    doc = nlp(text)
    masked_text = text
    entities = []
    for ent in reversed(doc.ents):
        if ent.label_ in ["PERSON", "GPE", "LOC", "ORG", "DATE"]:
            entities.append({
                "text": ent.text,
                "entity_type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            })
            masked_text = masked_text[:ent.start_char] + f"[{ent.label_}]" + masked_text[ent.end_char:]
    
    # Sort entities by start position
    entities.sort(key=lambda x: x['start'])
    return masked_text, entities

# --- LLM-based masking (Groq) ---
def detect_and_mask_pii_llm(text: str, api_key: Optional[str] = GROQ_API_KEY) -> Tuple[str, List[Dict[str, Any]]]:
    """Masks PII using an LLM (Groq). Returns empty entity list for now."""
    if not api_key:
        return "[LLM MASKING SKIPPED: GROQ_API_KEY not set]", []
    
    try:
        from groq import Groq
    except ImportError:
        return "[LLM MASKING SKIPPED: 'groq' package not installed]", []

    client = Groq(api_key=api_key)
    prompt = f"""Please mask all personally identifiable information (PII) in the following text. Replace names, emails, phone numbers, addresses, and any other sensitive data with appropriate placeholders like [NAME], [EMAIL], [PHONE], etc.

Text to mask:
"{text}"

Return only the masked text, with no additional commentary."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
        )
        masked_text = chat_completion.choices[0].message.content
        return (masked_text if masked_text else text), []
    except Exception as e:
        return f"[LLM MASKING FAILED: {e}]", []
