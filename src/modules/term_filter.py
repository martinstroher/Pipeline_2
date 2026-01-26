import os

import pandas as pd

INPUT_FILTERED_CSV= os.environ["AGGREGATOR_OUTPUT_FILE"]
OUTPUT_FILTERED_CSV = os.environ["FILTERED_TERMS_OUTPUT"]
MINIMUM_FREQUENCY = int(os.environ["MINIMUM_FREQUENCY_FILTER"])

def filter_top_terms():
    try:
        df_ranked = pd.read_csv(INPUT_FILTERED_CSV)

        df_filtered = df_ranked[df_ranked['Frequency'] >= MINIMUM_FREQUENCY]

        df_filtered.to_csv(OUTPUT_FILTERED_CSV, index=False, encoding='utf-8-sig')

        print(f"Filtered terms (Frequency >= {MINIMUM_FREQUENCY}) saved to '{OUTPUT_FILTERED_CSV}'")
        print(f"Number of terms reduced from {len(df_ranked)} to {len(df_filtered)}")

    except FileNotFoundError:
        print(f"ERROR: Input file '{INPUT_FILTERED_CSV}' not found. Make sure the consolidation script ran successfully.")
    except KeyError:
        print(f"ERROR: Column 'Frequency' not found in '{INPUT_FILTERED_CSV}'. Check the header.")
    except Exception as e:
        print(f"An error occurred: {e}")