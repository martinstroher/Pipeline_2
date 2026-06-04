import os

from src.utils import log
from src.utils.csv_io import read_csv, write_csv

INPUT_FILTERED_CSV= os.environ["AGGREGATOR_OUTPUT_FILE"]
OUTPUT_FILTERED_CSV = os.environ["FILTERED_TERMS_OUTPUT"]
MINIMUM_FREQUENCY = int(os.environ["MINIMUM_FREQUENCY_FILTER"])

def filter_top_terms():
    try:
        df_ranked = read_csv(INPUT_FILTERED_CSV)

        df_filtered = df_ranked[df_ranked['Frequency'] >= MINIMUM_FREQUENCY]

        write_csv(df_filtered, OUTPUT_FILTERED_CSV)

        log.info(f"Filtered terms (Frequency >= {MINIMUM_FREQUENCY}) saved to '{OUTPUT_FILTERED_CSV}'")
        log.detail(f"Reduced from {len(df_ranked)} to {len(df_filtered)} terms")

    except FileNotFoundError:
        log.error(f"Input file '{INPUT_FILTERED_CSV}' not found.")
    except KeyError:
        log.error(f"Column 'Frequency' not found in '{INPUT_FILTERED_CSV}'.")
    except Exception as e:
        log.error(f"{e}")
