import pandas as pd
from src.single_agent import detect_and_mask_pii_regex, detect_and_mask_pii_spacy

def run_single_agent_evaluation():
    """
    Runs the single-agent evaluation on a sample dataset.
    """
    try:
        df = pd.read_csv('data/sample_data.csv')
        print("Original Data:")
        print(df['text'])
        
        # --- Regex Agent ---
        print("\n--- Masking with Regex-based single agent ---")
        df['regex_masked'] = df['text'].apply(detect_and_mask_pii_regex)
        print(df['regex_masked'])

        # --- spaCy Agent ---
        print("\n--- Masking with spaCy-based single agent ---")
        df['spacy_masked'] = df['text'].apply(detect_and_mask_pii_spacy)
        print(df['spacy_masked'])

    except FileNotFoundError:
        print("Error: data/sample_data.csv not found.")
        print("Please ensure you have created the sample data file.")

if __name__ == '__main__':
    run_single_agent_evaluation()

