import argparse
import json
import logging
import spacy

from datetime import datetime
from itertools import product
from multiprocessing import Pool
from os import cpu_count, path
from tqdm import tqdm

from sbeval.constants import LOGGING_CONFIG


def _get_candidate_posts(pair: tuple) -> list:
    """Find all posts that contain both words in the given pair. Return them afterwards.

    If no candidate posts are found, return None.

    Arguments:
    pair -- A tuple of strings describing the pair to filter posts for.
    """
    global posts

    candidates = []
    for post in posts:
        # Split the post by whitespace into a list of tokens
        # Note: the input text is supposed to be whitespace tokenized already
        p_split = post.split(" ")

        # Check if both of the tokens are present in the resulting token list
        if pair[0] in p_split and pair[1] in p_split:
            candidates.append(post)

    # Only return a list if there are any candidate post to return
    if len(candidates) > 0:
        return candidates

    return None


def calculate_candidate_posts(pairs_to_test: list) -> list:
    """Find candidate posts that contain at least of the given pairs of words.

    This is mainly a wrapper function that distributes the search across all available cores.

    A potential performance improvement for the future would be to distribute the posts over
    multiple cores and check if any of the pairs is included in each post. This would avoid
    iterating over each post multiple times.

    Arguments:
    pairs_to_test -- A list of tuples, in which each tuple describes a word pair.
    """
    global args

    pool = Pool(processes=args.processing_cores)

    # For-loop to get a progress bar, even with different pools (small tqdm hack)
    candidates = []
    for _ in tqdm(pool.imap(_get_candidate_posts, pairs_to_test), total=len(pairs_to_test)):
        candidates.append(_)

    pool.close()
    pool.join()

    # Flatten and filter candidate list
    return [c for c_list in candidates if c_list is not None for c in c_list]


def get_sentences_from_posts(posts: list) -> list:
    """Split the given posts into sentences. Return a flat list of all sentences.

    Arguments:
    posts -- A list of all posts that should be split into sentences.
    """
    global args
    global nlp

    # Define a spacy processing pipe for all posts to disable unnecessary components
    posts_pipe = nlp.pipe(
        tqdm(posts),
        disable=["ner", "textcat"],
        n_process=args.processing_cores)

    all_sentences = []
    for post in posts_pipe:
        # Split post into sentences and extend the sentence list
        post_sents = [sent.text for sent in list(post.sents)]
        all_sentences.extend(post_sents)

    return all_sentences


def _get_cooccurrence_sentences(pair: tuple) -> tuple:
    """Collect all sentences that contain both words of the given pair.

    Return a tuple where the first element describes the pair and the second a list of sentences
    that contain both words. Return none if there are no sentences that contain both words.

    Arguments:
    pair -- A tuple describing the word pair.
    """
    global sentences

    # For each sentence, check if it contains both words of the pair
    cooccurrence_sentences = [
        s for s in sentences if pair[0] in s.split(" ") and pair[1] in s.split(" ")]

    # Only return a tuple if there are sentences containing both words
    if len(cooccurrence_sentences) > 0:
        return (pair, cooccurrence_sentences)

    return None


def calculate_cooccurrences(pairs_to_test: list) -> list:
    """For each of the given pair, collect all sentences that contain both words.

    Return a list of tuples where the first element is the pair and the second a list of sentences.

    Arguments:
    pairs_to_test -- A list of tuples, each describing a word pair.
    """
    global args

    pool = Pool(processes=args.processing_cores)

    # For-loop to get a progress bar, even with different pools (small tqdm hack)
    cooccurrences = []
    for _ in tqdm(pool.imap(_get_cooccurrence_sentences, pairs_to_test), total=len(pairs_to_test)):
        cooccurrences.append(_)

    pool.close()
    pool.join()

    # Filter the sentence list
    return [c for c in cooccurrences if c is not None]


def main():
    global args
    global nlp
    global posts
    global sentences

    # Read all posts from the given text file; assuming that posts are newline separated
    with open(args.data, "r") as f:
        posts = f.read().split("\n")

    # Read all target and associaton tests
    with open("sbeval/tests/weat_tests.json", "r") as f:
        weat_tests = json.load(f)

        # Some of the original WEAT (7 and 8) tests switch the lists of target and association words
        # For this analysis, we will switch them back again to make the code more streamlined
        tests_in_wrong_order = [7, 8]
        for test_number in tests_in_wrong_order:
            test_X = weat_tests[f"test{test_number}"]["X"]
            test_Y = weat_tests[f"test{test_number}"]["Y"]
            test_A = weat_tests[f"test{test_number}"]["A"]
            test_B = weat_tests[f"test{test_number}"]["B"]

            weat_tests[f"test{test_number}"] = {"X": test_A, "Y": test_B, "A": test_X, "B": test_Y}

        # Include only the specified tests, discard the rest
        weat_tests = {f"test{i}": weat_tests[f"test{i}"] for i in args.tests}

    # Generate pairs of all target/association word combinations that should be evaluated
    logging.info("Generating target-association test pairs...")
    pairs_to_test = []
    for test in weat_tests.values():
        associations_A = [a.lower() for a in test["A"]]
        associations_B = [a.lower() for a in test["B"]]
        target_x = [t.lower() for t in test["X"]]
        target_y = [t.lower() for t in test["Y"]]

        combinations_x = product(target_x, [*associations_A, *associations_B])
        combinations_y = product(target_y, [*associations_A, *associations_B])

        pairs_to_test = [*pairs_to_test, *combinations_x, *combinations_y]

    # Remove duplicate pairs (we only need each pair once)
    pairs_to_test = list(set(pairs_to_test))

    # Retrieve candidate posts to make following sentenization easier
    logging.info("Extracting candidate posts...")
    candidates = calculate_candidate_posts(pairs_to_test)

    logging.info("Splitting candidate posts into sentences...")
    # Some candidate posts might contain multiple pairs; thus we need to prune duplicates before
    # splitting them into sentences
    unique_candidates = set(candidates)
    sentences = get_sentences_from_posts(unique_candidates)

    # Get cooccurrence sentences for each pair
    logging.info("Calculating sentence-based cooccurrences...")
    cooccurrences_by_pair = calculate_cooccurrences(pairs_to_test)

    # Unpack cooccurrences
    logging.info("Unpacking cooccurrences into a more useful format...")
    cooccurrences_by_target = {}
    # For each pair with cooccurrence sentences...
    for cooccurrence in cooccurrences_by_pair:
        # Note: this assumes that the order of the word pairs did not change in previous processing
        # steps and that the first word of the tuple represents a target word
        target_word = cooccurrence[0][0]
        association_word = cooccurrence[0][1]

        # If the target word does not yet have an entry in the cooccurrence dict, add one
        if target_word not in cooccurrences_by_target.keys():
            cooccurrences_by_target[target_word] = {}

        # Count the number of sentences the target and association word cooccur with each other
        cooccurrences_by_target[target_word][association_word] = len(cooccurrence[1])

    # Sort co-occurrence counts by weat tests
    cooccurrences_by_test = {}
    for i in args.tests:
        test_data = weat_tests[f"test{i}"]
        test_associations = [assoc.lower() for assoc in [*test_data["A"], *test_data["B"]]]

        # Lowercase all target words
        test_data["X"] = [x.lower() for x in test_data["X"]]
        test_data["Y"] = [y.lower() for y in test_data["Y"]]

        target_x_associations = {}
        target_y_associations = {}

        # For each X target word in the current test...
        for x in test_data["X"]:
            # If current word has any sentence co-occurrences, add it to the final dictionary
            if x in cooccurrences_by_target.keys():
                target_associations = {
                    key: value
                    for key, value in cooccurrences_by_target[x].items()
                    if key in test_associations}

                if len(target_associations.keys()) > 0:
                    target_x_associations[x] = target_associations

        # For each Y target word in the current test...
        for y in test_data["Y"]:
            # If current word has any sentence co-occurrences, add it to the final dictionary
            if y in cooccurrences_by_target.keys():
                target_associations = {
                    key: value
                    for key, value in cooccurrences_by_target[y].items()
                    if key in test_associations}

                if len(target_associations.keys()) > 0:
                    target_y_associations[y] = target_associations

        cooccurrences_by_test[f"test{i}"] = {"X": target_x_associations, "Y": target_y_associations}

    # Export statistics to file
    dt = datetime.today().strftime("%Y%m%d%H%M%S")
    output_file = path.join(args.output, f"weat-cooccurrence-analysis_results-{dt}.json")

    logging.info(f"Exporting results to disk at {output_file}.")
    with open(output_file, "w") as f:
        json.dump({"corpus": path.basename(args.data), **cooccurrences_by_test}, f, indent=4)


if __name__ == "__main__":
    global nlp

    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to conduct a co-occurrence analysis of all WEAT test terms on a given corpus.")

    parser.add_argument(
        "-d",
        "--data",
        required=True,
        type=str,
        help="Path to the corpus. Expects one whitespace separated post per line.",
        metavar="DATA_PATH")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Path to the directory where the result file should be saved to.",
        metavar="OUTPUT_DIR")
    parser.add_argument(
        "-c",
        "--processing_cores",
        default=cpu_count() - 1,
        type=int,
        help="The number of processing cores to use for simultaneous computing of the scores.",
        metavar="PROCESSING_CORES")
    parser.add_argument(
        "-t",
        "--tests",
        required=True,
        nargs='+',
        help="A list of test numbers (from 1 to 10) that should be included in the analysis "
             "(whitespace separated).",
        type=int,
        metavar="TESTS_TO_INCLUDE")

    args = parser.parse_args()

    logging.basicConfig(**LOGGING_CONFIG)

    logging.info(
        "Please make sure that your input texts are whitespace separated tokens. The script might "
        "not work correctly otherwise.")

    nlp = spacy.load("en_core_web_sm")

    main()
    print("Done.")
