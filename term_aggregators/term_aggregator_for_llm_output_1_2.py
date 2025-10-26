import pandas as pd
from collections import Counter
import nltk
import os
from nltk.stem import RSLPStemmer

INPUT_FILE_PATH = os.environ["LLM_OUTPUT_FILE"]
OUTPUT_FILE_PATH = os.environ["AGGREGATOR_OUTPUT_FILE"]

def run_term_aggregation():
    def load_terms_from_txt(filepath):
        if not os.path.exists(filepath):
            print(f"ERROR: The file '{filepath}' was not found.")
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                terms_list = [line.strip() for line in f if line.strip()]
            print(f"Success! {len(terms_list)} terms loaded from '{filepath}'.")
            return terms_list
        except Exception as e:
            print(f"ERROR reading the text file: {e}")
            return None

    pt_stemmer = RSLPStemmer()

    raw_terms_list = load_terms_from_txt(INPUT_FILE_PATH)

    if raw_terms_list is not None:
        stemmed_terms = []
        stem_to_readable_map = {}

        print("Starting normalization, stemming, and mapping...")
        for original_term in raw_terms_list:
            if not isinstance(original_term, str):
                continue

            clean_original_term = original_term.strip().lower()

            if len(clean_original_term) < 3:
                continue

            words = clean_original_term.split()
            stemmed_words = [pt_stemmer.stem(p) for p in words]
            final_stem = " ".join(stemmed_words)

            stemmed_terms.append(final_stem)

            if final_stem not in stem_to_readable_map or len(clean_original_term) < len(stem_to_readable_map[final_stem]):
                stem_to_readable_map[final_stem] = clean_original_term

        print("Processing complete.")

        stem_frequencies = Counter(stemmed_terms)

        final_results = []
        for stem, count in stem_frequencies.most_common():
            readable_term = stem_to_readable_map.get(stem, stem) # Use stem itself if not found (fallback)
            final_results.append((readable_term, count))

        print("\n--- Most Common Terms ---")
        for term, count in final_results[:15]:
            print(f"Term: '{term}' | Count: {count}")

        final_df = pd.DataFrame(final_results, columns=['Readable_Term', 'Frequency'])
        final_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
        print(f"\nFinal results successfully saved to '{OUTPUT_FILE_PATH}'")