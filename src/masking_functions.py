"""Core PII masking functions using different strategies (regex, spaCy, LLM)."""

import re
import spacy
from typing import Any, Dict, List, Optional, Tuple
from config import GROQ_API_KEY

# --- Regex-based masking ---
EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_REGEX = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
PHONE_REGEX_INTL = r"\+\d{1,3}[-.\s]?\d{3,14}"
AADHAR_REGEX = r"\b\d{4}-\d{4}-\d{4}\b"
PAN_REGEX = r"\b[A-Z]{5}\d{4}[A-Z]\b"
SSN_REGEX = r"\b\d{3}-\d{2}-\d{4}\b"
BANK_ACCOUNT_REGEX = r"\b\d{9,18}\b"
IP_ADDRESS_REGEX = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
CRYPTO_WALLET_REGEX = r"\b0x[a-fA-F0-9]{40}\b"
USERNAME_REGEX = r"\b[a-zA-Z][a-zA-Z0-9_]{2,}\b"
PASSWORD_REGEX = r"\b[A-Za-z0-9@#$%^&*!?._-]{6,}\b"
API_KEY_REGEX = r"\b(?:sk_live|ghp|AKIA)[A-Za-z0-9_=-]+\b"
PASSPORT_REGEX = r"\b[Pp]\d{7}\b"
DRIVING_LICENSE_REGEX = r"\b(?:DL|DL-)\d{6,}\b"
DEVICE_ID_REGEX = r"\b[A-Z]{2,5}-\d{6,}\b"


def _append_matches(
    original_text: str,
    pattern: str,
    label: str,
    entities: List[Dict[str, Any]],
    masked_text: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    matches = list(re.finditer(pattern, original_text))
    if not matches:
        return masked_text, entities

    for match in matches:
        entities.append({
            "text": match.group(0),
            "entity_type": label,
            "start": match.start(),
            "end": match.end(),
        })
    masked_text = re.sub(pattern, f"[{label}]", masked_text)
    return masked_text, entities

def detect_and_mask_pii_regex(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Masks emails and phone numbers using regex and returns detected entities."""
    entities = []
    masked_text = text

    for pattern, label in [
        (EMAIL_REGEX, "EMAIL"),
        (PHONE_REGEX, "PHONE"),
        (PHONE_REGEX_INTL, "PHONE"),
        (AADHAR_REGEX, "AADHAR"),
        (PAN_REGEX, "PAN"),
        (SSN_REGEX, "SSN"),
        (BANK_ACCOUNT_REGEX, "BANK_ACCOUNT"),
        (IP_ADDRESS_REGEX, "IP_ADDRESS"),
        (CRYPTO_WALLET_REGEX, "CRYPTO_WALLET"),
        (PASSPORT_REGEX, "PASSPORT"),
        (DRIVING_LICENSE_REGEX, "DRIVING_LICENSE"),
        (DEVICE_ID_REGEX, "DEVICE_ID"),
    ]:
        masked_text, entities = _append_matches(text, pattern, label, entities, masked_text)

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
            label = "NAME" if ent.label_ == "PERSON" else ent.label_
            entities.append({
                "text": ent.text,
                "entity_type": label,
                "start": ent.start_char,
                "end": ent.end_char,
            })
            masked_text = masked_text[:ent.start_char] + f"[{label}]" + masked_text[ent.end_char:]
    
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
