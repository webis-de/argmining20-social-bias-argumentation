import argparse
import logging
import spacy

from os import cpu_count, path
from tqdm import tqdm

from sbeval.constants import LOGGING_CONFIG
from sbeval.utils import split_corpus


def main():
    # Initialize spacy model
    nlp = spacy.load("en_core_web_sm")

    # Read data from disk
    logging.info("Reading data from disk...")
    with open(DATA_PATH, "r") as f:
        iac_posts = f.read().split("\n")

    # Extract texts of interest from data
    logging.info("Tokenizing posts...")
    posts_parsed = list(
        nlp.pipe(
            tqdm(iac_posts),
            disable=["tagger", "parser", "ner", "textcat"],
            n_process=cpu_count() - 1))
    logging.info("Joining tokens back together as space separated texts...")
    posts_tokenized = [
        " ".join([token.text.lower() for token in post]) for post in tqdm(posts_parsed)]

    # Write extracted text to file
    file_basename = "cmv-text_only--glove-format"
    output_file = path.join(OUTPUT_DIR, f"{file_basename}.txt")
    logging.info(f"Writing posts to disk at {output_file}.")
    with open(output_file, "w") as f:
        f.write("\n".join(posts_tokenized))

    # Generate random corpus splits and write them to disk
    logging.info("Generating random splits and writing them to disk...")
    split_corpus(posts_tokenized, args.splits, OUTPUT_DIR, file_basename)


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to extract prepared textual data from the webis CMV 2020 texts and export"
        "them to files usable by the GloVe algorithm.")

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="The path to the CMV data file.",
        metavar="CMV_DATA")
    parser.add_argument(
        "--output",
        "-o",
        default="output",
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
    OUTPUT_DIR = args.output

    logging.basicConfig(**LOGGING_CONFIG)

    main()
