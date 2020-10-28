import argparse
import json_lines
import logging

from os import path
from tqdm import tqdm

from sbeval.constants import LOGGING_CONFIG


def get_all_texts(comment: dict) -> list:
    """Recursive function to collect a comment's texts and all its children.

    Return a list containing all children texts

    Arguments:
    comment -- A dict containing all possible children.
    """
    comment_texts = [comment["body"]]

    # If comment has children...
    if "children" in comment.keys() and len(comment["children"]) > 0:
        # Collect all the childrens text in a flat list
        child_texts = [text for child in comment["children"] for text in get_all_texts(child)]
        comment_texts.extend(child_texts)

    # Return all texts of the given comment
    return comment_texts


def main():
    # List to store all texts and comments
    thread_texts = []

    # Read thread file from disk
    logging.info("Reading file from disk.")
    with open(args.input, "r") as f:
        threads = [item for item in json_lines.reader(f)]

    # For each thread...
    logging.info("Extracting texts from threads and comments.")
    for thread in tqdm(threads):
        # If this thread (still) has a selftext, extract it
        if "selftext" in thread.keys() and len(thread["selftext"]) > 0:
            thread_texts.append(thread["selftext"])

        # For each comment of this thread
        for comment in thread["comments"]:
            thread_texts.extend(get_all_texts(comment))

    # Remove all newlines inside the texts as they serve as separator in the final export
    logging.info("Removing newlines from post texts.")
    thread_texts = [text.replace("\n", " ") for text in tqdm(thread_texts)]

    # Write file of all included debate portal into a combined file
    output_file = path.join(args.output, "webis-cmv-20-texts_only.txt")
    logging.info(f"Writing extracted post texts to file at {output_file}.")
    with open(output_file, "w") as f:
        f.write("\n".join(thread_texts))


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to prepare the Webis CMV 2020 corpus to a plain text format.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input threads file.",
        metavar="INPUT_FILE")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to the directory the output should be written to.",
        metavar="OUTPUT_DIR")

    args = parser.parse_args()

    logging.basicConfig(**LOGGING_CONFIG)

    main()
    print("Done.")
