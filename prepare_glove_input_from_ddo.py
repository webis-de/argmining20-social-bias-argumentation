import argparse
import json
import logging
import pandas as pd

from os import path
from tqdm import tqdm

from sbeval.constants import LOGGING_CONFIG
from sbeval.utils import split_corpus


def main():
    # Read data from disk
    logging.info("Reading data from disk...")
    data = pd.read_csv(DATA_PATH)

    if GROUP_PROPERTIES_PATH:
        # Subgroups for which to export data
        with open(GROUP_PROPERTIES_PATH, "r") as f:
            group_properties = json.load(f)

        # Extract texts of interest from data
        logging.info("Extracting texts from groups of interests to separate files...")
        for prop, values in tqdm(group_properties.items(), position=1):
            for group in tqdm(values, position=2, leave=False):
                # Collect group texts into a single string, newline separated
                group_texts = data[data[prop] == group["column_value"]].argument_prepared.values

                # Remove nan values
                group_texts_clean = [text for text in group_texts if type(text) == str]

                # Write extracted text to file
                file_basename = f"debate_org-{prop}-{group['description']}-posts--glove-format"
                output_file = path.join(
                    OUTPUT_DIR, f"{file_basename}.txt")
                logging.info(f"Writing posts to disk at '{output_file}'...")
                with open(output_file, "w") as f:
                    f.write("\n".join(group_texts_clean).lower())

                # Generate random corpus splits and write them to disk
                logging.info("Generating random splits and writing them to disk...")
                split_corpus(group_texts_clean, args.splits, OUTPUT_DIR, file_basename)
    else:
        # Collect all posts
        logging.info("Extracting posts from given data...")
        texts = data.argument_prepared.values

        # Remove nan values
        texts_clean = [text for text in texts if type(text) == str]

        # Write extracted posts to file
        file_basename = "debate_org-all_posts--glove-format"
        output_file = path.join(OUTPUT_DIR, f"{file_basename}.txt")
        logging.info(f"Writing posts to disk at '{output_file}'...")
        with open(output_file, "w") as f:
            f.write("\n".join(texts_clean).lower())

        # Generate random corpus splits and write them to disk
        logging.info("Generating random splits and writing them to disk...")
        split_corpus(texts_clean, args.splits, OUTPUT_DIR, file_basename)


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to extract prepared textual data from the debate-org dataframe and export \
        them to files usable by the GloVe algorithm.")

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="The path to the debates data file (as .csv).",
        metavar="DEBATES_DATA")
    parser.add_argument(
        "--groups",
        "-g",
        default=None,
        type=str,
        help="Path to the file containing the groups of interest (as .json).",
        metavar="GROUP_PROPERTIES")
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        type=str,
        help="Path to directory where output files should be placed.",
        metavar="OUTPUT_DIR")
    parser.add_argument(
        "--splits",
        "-s",
        required=True,
        type=int,
        help="Number of random splits that should additionally be generated.",
        metavar="SPLITS")

    args = parser.parse_args()

    # Set data variables
    DATA_PATH = args.input
    GROUP_PROPERTIES_PATH = args.groups
    OUTPUT_DIR = args.output

    logging.basicConfig(**LOGGING_CONFIG)

    main()

    print("Done.")
