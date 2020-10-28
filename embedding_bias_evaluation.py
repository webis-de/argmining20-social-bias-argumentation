import argparse
import json
import logging

from datetime import datetime
from os import path

from sbeval.constants import LOGGING_CONFIG
from sbeval.weat_test import weat_score
from sbeval.word_vectors import CustomEmbeddings


def weat_evaluation(lexicons: dict, embeddings_model) -> dict:
    weat_results = {}

    # For each of the tests...
    for test_name, lexicon in lexicons.items():

        # Save lexicons into variables so they can be savely modified if necessary (e.g. lowercased)
        lexicon_x = lexicon["X"]
        lexicon_y = lexicon["Y"]
        lexicon_a = lexicon["A"]
        lexicon_b = lexicon["B"]

        # If specified, lowercase all lexicons
        if args.lowercase:
            lexicon_x = [token.lower() for token in lexicon_x]
            lexicon_y = [token.lower() for token in lexicon_y]
            lexicon_a = [token.lower() for token in lexicon_a]
            lexicon_b = [token.lower() for token in lexicon_b]

        # Try to caclulate score for the two different lexicons
        # Catch if none of the for at least one of the lexicons none of its words is in-vocabulary
        # Also lowercase all lexicon terms before testing
        try:
            test_result = weat_score(
                lexicon_x,
                lexicon_y,
                lexicon_a,
                lexicon_b,
                word_vector_getter=embeddings_model)

            weat_results[test_name] = {
                "score": test_result[0],
                "oov_tokens": test_result[1]}
        except AttributeError as e:
            weat_results[test_name] = f"No results possible: '{e}'"

    return weat_results


def main():
    # Load the given embeddings model from disk
    logging.info("Loading embedding model from disk.")
    embeddings_model = CustomEmbeddings(args.embedding_model)

    # Load metric test lexicons
    with open(path.join("sbeval", "tests", "weat_tests.json"), "r") as f:
        weat_lexicons = json.load(f)

    # Dict to store all test results
    results = {
        "embeddings_model": path.basename(args.embedding_model),
        "weat": {}}

    # Conduct all WEAT test evaluations
    logging.info("Evaluating WEAT tests.")
    results["weat"] = weat_evaluation(weat_lexicons, embeddings_model)

    # Export the results to disk
    dt = datetime.today().strftime("%Y%m%d%H%M%S")
    output_file = path.join(args.output, f"embedding_bias_evaluation_results-{dt}.json")
    logging.info(f"Exporting results to disk at {output_file}.")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to evaluate the bias of a given embdding model using different metrics.")

    parser.add_argument(
        "-e",
        "--embedding_model",
        required=True,
        type=str,
        help="Path to the embedding model. It needs to be in the word2vec format, binary or plain.",
        metavar="EMBEDDINGS")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to the directory where the result file should be written to.",
        metavar="OUTPUT_DIR")
    parser.add_argument(
        "-l",
        "--lowercase",
        action="store_true",
        help="Whether to lowercase all lexicons before testing or not. This is sometimes required "
             "when the embedding model was generated on solely lowercased tokens.")

    args = parser.parse_args()

    logging.basicConfig(**LOGGING_CONFIG)

    main()

    print("Done.")
