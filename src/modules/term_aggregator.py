import pandas as pd
from collections import Counter
import os
import spacy
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import log
from src.utils.csv_io import read_csv, write_csv

load_dotenv()

INPUT_FILE_PATH = os.environ["LLM_OUTPUT_FILE"]
OUTPUT_FILE_PATH = os.environ["AGGREGATOR_OUTPUT_FILE"]

if not INPUT_FILE_PATH or not OUTPUT_FILE_PATH:
    raise RuntimeError("Environment variables LLM_OUTPUT_FILE and AGGREGATOR_OUTPUT_FILE must be set.")


def run_term_aggregation():
    def load_terms_from_csv(filepath):
        """Loads terms from the 'Entity' column of a CSV file."""
        if not os.path.exists(filepath):
            log.error(f"File '{filepath}' not found.")
            return None
        try:
            df = read_csv(filepath, delimiter=',', header=0, usecols=['Entity'])
            terms_list = df['Entity'].squeeze().tolist()
            log.info(f"{len(terms_list)} raw terms loaded from '{filepath}'.")
            return terms_list
        except ValueError:
            log.error(f"Column 'Entity' not found in '{filepath}'.")
            return None
        except Exception as e:
            log.error(f"Reading CSV: {e}")
            return None

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")

    raw_terms_list = load_terms_from_csv(INPUT_FILE_PATH)

    if raw_terms_list is not None:
        stemmed_terms = []
        stem_to_readable_map = {}

        for original_term in tqdm(raw_terms_list, desc="Lemmatizing"):
            if not isinstance(original_term, str):
                continue

            clean_original_term = original_term.strip().lower()

            if len(clean_original_term) < 3:
                continue

            doc = nlp(clean_original_term)
            lemma_key = " ".join([token.lemma_ for token in doc])

            stemmed_terms.append(lemma_key)

            if lemma_key not in stem_to_readable_map or len(clean_original_term) < len(
                    stem_to_readable_map[lemma_key]):
                stem_to_readable_map[lemma_key] = clean_original_term

        stem_frequencies = Counter(stemmed_terms)

        final_results = []
        for stem, count in stem_frequencies.most_common():
            readable_term = stem_to_readable_map.get(stem, stem)
            final_results.append((readable_term, count))

        log.detail(f"Top terms: {', '.join(f'{t} ({c})' for t, c in final_results[:5])}")

        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                log.detail(f"Created directory: {output_dir}")
            except OSError as e:
                raise RuntimeError(f"Creating directory {output_dir}: {e}")

        try:
            final_df = pd.DataFrame(final_results, columns=['Readable_Term', 'Frequency'])
            write_csv(final_df, OUTPUT_FILE_PATH)
            log.success(f"{len(final_df)} aggregated terms saved to '{OUTPUT_FILE_PATH}'")
        except Exception as e:
            log.error(f"Saving CSV '{OUTPUT_FILE_PATH}': {e}")
