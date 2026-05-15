"""Core PII masking functions using different strategies (regex, spaCy, LLM)."""

import re
import spacy
from typing import Optional
from config import GROQ_API_KEY

# --- Regex-based masking ---
EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
PHONE_REGEX = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"

def detect_and_mask_pii_regex(text: str) -> str:
    """Masks emails and phone numbers using regex."""
    text = re.sub(EMAIL_REGEX, "[EMAIL]", text)
    text = re.sub(PHONE_REGEX, "[PHONE]", text)
    return text

# --- spaCy-based masking ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def detect_and_mask_pii_spacy(text: str) -> str:
    """Masks PII using spaCy NER."""
    doc = nlp(text)
    masked_text = text
    for ent in reversed(doc.ents):
        if ent.label_ in ["PERSON", "GPE", "LOC", "ORG", "DATE"]:
            masked_text = masked_text[:ent.start_char] + f"[{ent.label_}]" + masked_text[ent.end_char:]
    return masked_text

# --- LLM-based masking (Groq) ---
def detect_and_mask_pii_llm(text: str, api_key: Optional[str] = GROQ_API_KEY) -> str:
    """Masks PII using an LLM (Groq)."""
    if not api_key:
        return "[LLM MASKING SKIPPED: GROQ_API_KEY not set]"
    
    try:
        from groq import Groq
    except ImportError:
        return "[LLM MASKING SKIPPED: 'groq' package not installed]"

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
        return masked_text if masked_text else text
    except Exception as e:
        return f"[LLM MASKING FAILED: {e}]"
