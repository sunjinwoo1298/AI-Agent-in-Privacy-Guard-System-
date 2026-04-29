import pandas as pd
import re
import spacy
from spacy.cli import download

# --- ML-based (spaCy) ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def detect_and_mask_pii_spacy(text):
    doc = nlp(text)
    masked_text = list(text)
    
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            start = ent.start_char
            end = ent.end_char
            masked_text[start:end] = f"[{ent.label_}]"
    
    return "".join(masked_text)

# --- Rule-based (Regex) ---
def detect_and_mask_pii_regex(text):
    # Simple regex for emails and phone numbers
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_regex = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    
    # Mask emails
    text = re.sub(email_regex, '[EMAIL]', text)
    
    # Mask phone numbers
    text = re.sub(phone_regex, '[PHONE]', text)
    
    return text

if __name__ == '__main__':
    # This is a placeholder for where the full pipeline will run.
    # For now, we'll just test our functions.
    
    sample_text_regex = "Contact me at test@example.com or 555-555-5555."
    masked_text_regex = detect_and_mask_pii_regex(sample_text_regex)
    
    print(f"Original (Regex): {sample_text_regex}")
    print(f"Masked (Regex):   {masked_text_regex}")

    sample_text_spacy = "John Doe works at Acme Corp."
    masked_text_spacy = detect_and_mask_pii_spacy(sample_text_spacy)
    print(f"\nOriginal (spaCy): {sample_text_spacy}")
    print(f"Masked (spaCy):   {masked_text_spacy}")
