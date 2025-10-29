import pandas as pd
from collections import Counter
import nltk
import os
import sys
# --- CHANGE 1: Import the correct stemmer ---
from nltk.stem import SnowballStemmer
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE_PATH = os.environ["LLM_OUTPUT_FILE"]
OUTPUT_FILE_PATH = os.environ["AGGREGATOR_OUTPUT_FILE"]

if not INPUT_FILE_PATH or not OUTPUT_FILE_PATH:
    print("ERROR: Environment variables LLM_OUTPUT_FILE and AGGREGATOR_OUTPUT_FILE must be set.")
    sys.exit(1)


def run_term_aggregation():
    def load_terms_from_csv(filepath):
        """Loads terms from the 'Entity' column of a CSV file."""
        if not os.path.exists(filepath):
            print(f"ERROR: The file '{filepath}' was not found.")
            return None
        try:
            df = pd.read_csv(filepath, encoding='utf-8', delimiter=',', header=0, usecols=['Entity'])
            terms_list = df['Entity'].squeeze().tolist()
            print(f"Success! {len(terms_list)} raw terms loaded from '{filepath}'.")
            return terms_list
        except ValueError:
            print(f"ERROR: Column 'Entity' not found in '{filepath}'. Please check the input file.")
            return None
        except Exception as e:
            print(f"ERROR reading the CSV file: {e}")
            return None

    try:
        stemmer = SnowballStemmer("english")
    except LookupError:
        print("ERROR: NLTK Snowball stemmer resource not found. Run nltk.download('punkt') / 'wordnet'.")
        sys.exit(1)

    raw_terms_list = load_terms_from_csv(INPUT_FILE_PATH)

    if raw_terms_list is not None:
        stemmed_terms = []
        stem_to_readable_map = {}

        print("Starting normalization, stemming (English), and mapping...")  # Log updated
        for original_term in raw_terms_list:
            if not isinstance(original_term, str):
                continue

            clean_original_term = original_term.strip().lower()

            if len(clean_original_term) < 3:
                continue

            words = clean_original_term.split()
            stemmed_words = [stemmer.stem(p) for p in words]
            final_stem = " ".join(stemmed_words)

            stemmed_terms.append(final_stem)

            if final_stem not in stem_to_readable_map or len(clean_original_term) < len(
                    stem_to_readable_map[final_stem]):
                stem_to_readable_map[final_stem] = clean_original_term

        print("Processing complete.")

        stem_frequencies = Counter(stemmed_terms)

        final_results = []
        for stem, count in stem_frequencies.most_common():
            readable_term = stem_to_readable_map.get(stem, stem)
            final_results.append((readable_term, count))

        print("\n--- Most Common Terms ---")
        for term, count in final_results[:15]:
            print(f"Term: '{term}' | Count: {count}")

        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except OSError as e:
                print(f"ERROR creating output directory {output_dir}: {e}")
                sys.exit(1)

        try:
            final_df = pd.DataFrame(final_results, columns=['Readable_Term', 'Frequency'])
            final_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
            print(f"\nFinal results successfully saved to '{OUTPUT_FILE_PATH}'")
        except Exception as e:
            print(f"ERROR saving results to CSV file '{OUTPUT_FILE_PATH}': {e}")