import argparse
import logging
import pandas as pd
import re
import spacy

from os import path
from tqdm import tqdm

from sbeval.constants import LOGGING_CONFIG


def main():
    # Read all data splits into a single pandas dataframe and extract only unqiue posts
    logging.info("Reading data from disk...")
    sbf_data = pd.concat([
        pd.read_csv(path.join(DATA_PATH, "SBFv2.trn.csv")),
        pd.read_csv(path.join(DATA_PATH, "SBFv2.dev.csv")),
        pd.read_csv(path.join(DATA_PATH, "SBFv2.tst.csv"))])
    sbf_posts = list(set(sbf_data.post.values))

    logging.info("Cleaning posts...")
    posts_cleaned = []
    for post in tqdm(sbf_posts):
        # Remove twitter mentions
        # post_no_mentions = re.sub(r"\@\w*", "", post)
        # Remove html entities
        post_no_html_entities = re.sub(r"(&.+?;)", "", post)
        # Remove linefeeds, twitter RTs, twitter mentions and replace ticks
        post_clean = post_no_html_entities.replace(
            "\n", " ").replace("RT", "").replace("@", "").replace("’", "'")
        # Tokenize and lowercase the post (and remove punctuation)
        # post_final = [t.text.lower() for t in nlp(post_clean) if not t.is_punct]
        post_final = [t.text.lower() for t in nlp(post_clean)]

        posts_cleaned.append(" ".join(post_final))

    # Write all cleaned posts to file, joined by a linefeed
    output_file = path.join(OUTPUT_DIR, "sbf_posts--glove_format.txt")
    logging.info(f"Exporting clean posts to disk at '{output_file}'...")
    with open(output_file, "w") as f:
        f.write("\n".join(posts_cleaned))


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to extract prepared textual data from the social bias frames data and export \
        them to files usable by the GloVe algorithm.")

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="The path to the directory containing the three data split sets.",
        metavar="DEBATES_DATA")
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Path to directory where output files should be placed.",
        metavar="OUTPUT_DIR")

    args = parser.parse_args()

    # Set data variables
    DATA_PATH = args.input
    OUTPUT_DIR = args.output

    logging.basicConfig(**LOGGING_CONFIG)

    # Initialize stanza pipeline
    nlp = spacy.load("en_core_web_sm")

    main()

    print("Done.")
