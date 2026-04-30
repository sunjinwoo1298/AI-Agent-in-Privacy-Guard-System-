import pandas as pd
import re
import spacy
from spacy.cli import download
from groq import Groq
from config import GROQ_API_KEY

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

# --- ML-based (spaCy) ---
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    print("Downloading spaCy model 'en_core_web_trf'...")
    download("en_core_web_trf")
    nlp = spacy.load("en_core_web_trf")

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

    sample_text_llm = "My name is John Doe and my email is john.doe@example.com."
    masked_text_llm = detect_and_mask_pii_llm(sample_text_llm)
    print(f"\nOriginal (LLM): {sample_text_llm}")
    print(f"Masked (LLM):   {masked_text_llm}")
