import pandas as pd
from groq import Groq
# from config import GROQ_API_KEY
from dotenv import load_dotenv
load_dotenv()
import os

from src.ner_backend import load_pii_backend, normalize_pii_label

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- LLM-based (Groq) ---
def detect_and_mask_pii_llm(text):
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_API_KEY":
        return "Error: Groq API key not configured. Please add it to config.py."

    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    Analyze the following text and identify any personally identifiable information (PII) such as names, emails, and phone numbers.
    Your task is to return the original text with the identified PII replaced by a corresponding placeholder (e.g., [NAME], [EMAIL], [PHONE]).
    Do not provide any explanation, only the masked text.

    Text: "{text}"
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
            temperature=0,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error during LLM call: {e}"

# --- ML-based (GLiNER) ---
nlp = load_pii_backend()

def detect_and_mask_pii_gliner(text):
    if nlp is None:
        return text

    doc = nlp(text)

    # Replace entities from right-to-left so offsets remain valid.
    masked = text
    ents = [ent for ent in doc.ents if normalize_pii_label(ent.label_) in ["NAME", "ORG", "ADDRESS", "MISC", "EMAIL", "PHONE", "CREDIT_CARD", "BANK_ACCOUNT", "SSN", "PASSPORT", "DRIVING_LICENSE", "MEDICAL_ID", "VOTER_ID", "AADHAR", "PAN", "IP_ADDRESS", "USERNAME", "PASSWORD", "API_KEY", "DEVICE_ID", "CRYPTO_WALLET", "DOB"]]
    for ent in sorted(ents, key=lambda e: e.start_char, reverse=True):
        masked = masked[: ent.start_char] + f"[{normalize_pii_label(ent.label_)}]" + masked[ent.end_char :]

    return masked


def detect_and_mask_pii_spacy(text):
    """Backward-compatible alias for older imports."""

    return detect_and_mask_pii_gliner(text)

if __name__ == '__main__':
    # This is a placeholder for where the full pipeline will run.
    # For now, we'll just test our functions.
    sample_text_spacy = "John Doe works at Acme Corp."
    masked_text_spacy = detect_and_mask_pii_gliner(sample_text_spacy)
    print(f"Original (GLiNER): {sample_text_spacy}")
    print(f"Masked (GLiNER):   {masked_text_spacy}")

    sample_text_llm = "My name is John Doe and my email is john.doe@example.com."
    masked_text_llm = detect_and_mask_pii_llm(sample_text_llm)
    print(f"\nOriginal (LLM): {sample_text_llm}")
    print(f"Masked (LLM):   {masked_text_llm}")
