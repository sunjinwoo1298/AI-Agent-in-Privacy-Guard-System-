"""Core PII masking functions using the shared GLiNER backend and LLM."""

from typing import Any, Dict, List, Optional, Tuple
from config import GROQ_API_KEY
from src.ner_backend import load_pii_backend, normalize_pii_label

PII_MASK_LABELS = {
    "NAME",
    "ORG",
    "ADDRESS",
    "EMAIL",
    "PHONE",
    "CREDIT_CARD",
    "BANK_ACCOUNT",
    "SSN",
    "PASSPORT",
    "DRIVING_LICENSE",
    "MEDICAL_ID",
    "VOTER_ID",
    "AADHAR",
    "PAN",
    "IP_ADDRESS",
    "USERNAME",
    "PASSWORD",
    "API_KEY",
    "DEVICE_ID",
    "CRYPTO_WALLET",
    "DOB",
    "MISC",
}

# --- GLiNER masking ---
nlp = load_pii_backend()

def detect_and_mask_pii_gliner(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Masks PII using the GLiNER backend and returns detected entities."""
    if nlp is None:
        return text, []

    doc = nlp(text)
    masked_text = text
    entities = []
    for ent in reversed(doc.ents):
        label = normalize_pii_label(ent.label_)
        if label in PII_MASK_LABELS:
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


def detect_and_mask_pii_spacy(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Backward-compatible alias for older imports."""

    return detect_and_mask_pii_gliner(text)

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
