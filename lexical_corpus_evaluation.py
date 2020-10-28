import argparse
import codecs
import json
import logging
import re
import spacy

from collections import Counter
from datetime import datetime
from nltk import ngrams
from os import cpu_count, path
from spacy.tokenizer import Tokenizer
from tqdm import tqdm

from sbeval.constants import LOGGING_CONFIG


def tokenize_and_tag_posts(posts: list) -> list:
    """Tokenize the posts and assign PoS tags to the tokens. Return as list of tuples.

    Arguments:
    posts -- The list of posts that should be tokenized and tagged.
    """
    global nlp

    post_pipe = nlp.pipe(
        tqdm(posts),
        disable=["parser", "ner", "textcat"],
        n_process=args.processing_cores)

    posts_tokenized = []
    for post in post_pipe:
        post_tokens = []

        for token in tqdm(post, leave=False, desc="Post"):
            # Check if token fulfills all requirements to be kept
            keep_token = all([
                not token.is_punct,
                not token.pos_ == "PUNCT",
                not token.is_stop,
                not token.like_url,
                not token.like_email,
                not token.like_num,
                not token.pos_ == "NUM",
                not token.pos_ == "SYM",
                not token.is_space])

            if keep_token:
                post_tokens.append((token.lemma_.lower(), token.pos_))
        posts_tokenized.append(post_tokens)

    return posts_tokenized


def extract_context(word_of_interest: str, posts: list, context_size: int = 10) -> list:
    """Extract the context before and after a given word of interest from all given posts.

    Return a list of lists of tuples containing the context tokens.

    Arguments:
    word_of_interest -- The word of interest for which the context should be extracted.
    posts -- A list of posts from which the context should be extracted. The posts are assumed to
             already be tokenized, where each token is represented by a tuple of the
             form (LEMMA, POS).
    """
    # Extract WOI indices for each post
    woi_indices = [
        [i for i, token in enumerate(post) if token[0] == word_of_interest] for post in posts]

    context_before_woi = []
    context_after_woi = []

    # For each post's index list...
    for i, post_indices in enumerate(tqdm(woi_indices)):
        for woi_index in post_indices:
            # Context before the index
            if woi_index < context_size:
                context_before_woi.append(posts[i][0:woi_index])
            else:
                context_before_woi.append(posts[i][woi_index - context_size:woi_index])

            # Context after the index
            if woi_index + context_size > len(posts[i]):
                context_after_woi.append(posts[i][woi_index + 1:])
            else:
                context_after_woi.append(posts[i][woi_index + 1:woi_index + context_size])

    return [*context_before_woi, *context_after_woi]


def generate_statistics(contexts: list) -> dict:
    """Generate unigrams, bigrams and trigrams and generate statistics based on those.

    Return a dictionary containing the results of different tests.

    Arguments:
    contexts -- A list of lists of tuples containing the contexts for the word of interest. Contexts
                are expected to be lists of tuples, where each tuple defines a token and contains
                the literal token and its POS tag.
    """
    n_most_common = args.most_common

    # Read positive and negative word lists
    with codecs.open(args.positive_word_list, "r", encoding="latin1") as f:
        positive_words = f.read().split("\r\n")[31:]
    with codecs.open(args.negative_word_list, "r", encoding="latin1") as f:
        negative_words = f.read().split("\r\n")[31:]

    # Generate lists of unigrams, bigrams and trigrams of context tokens
    contexts_unigrams = [unigram for context in contexts for unigram in context]
    contexts_bigrams = [
        bigram for context in contexts for bigram in list(ngrams(context, 2))]
    contexts_trigrams = [
        bigram for context in contexts for bigram in list(ngrams(context, 3))]

    negative_context_unigrams = [u for u in contexts_unigrams if u[0] in negative_words]
    positive_context_unigrams = [u for u in contexts_unigrams if u[0] in positive_words]

    return {
        "unigrams": Counter([u[0] for u in contexts_unigrams]).most_common(n=n_most_common),
        "unigrams_adj": Counter(
            [u[0] for u in contexts_unigrams if u[1] == "ADJ"]).most_common(n=n_most_common),
        "unigrams_noun": Counter(
            [u[0] for u in contexts_unigrams if u[1] == "NOUN"]).most_common(n=n_most_common),
        "unigrams_negative_adj_total": len([u for u in negative_context_unigrams if u[1] == "ADJ"]),
        "unigrams_negative_adj": Counter(
            [u[0] for u in negative_context_unigrams if u[1] == "ADJ"]).most_common(
                n=n_most_common),
        "unigrams_negative_noun_total": len(
            [u for u in negative_context_unigrams if u[1] == "NOUN"]),
        "unigrams_negative_noun": Counter(
            [u[0] for u in negative_context_unigrams if u[1] == "NOUN"]).most_common(
                n=n_most_common),
        "unigrams_positive_adj_total": len([u for u in positive_context_unigrams if u[1] == "ADJ"]),
        "unigrams_positive_adj": Counter(
            [u[0] for u in positive_context_unigrams if u[1] == "ADJ"]).most_common(
                n=n_most_common),
        "unigrams_positive_noun_total": len(
            [u for u in positive_context_unigrams if u[1] == "NOUN"]),
        "unigrams_positive_noun": Counter(
            [u[0] for u in positive_context_unigrams if u[1] == "NOUN"]).most_common(
                n=n_most_common),
        "bigrams": Counter(
            [f"{b[0][0]}_{b[1][0]}" for b in contexts_bigrams]).most_common(n=n_most_common),
        "bigrams_adv_adj": Counter(
            [f"{b[0][0]}_{b[1][0]}" for b in contexts_bigrams
                if b[0][1] == "ADV" and b[1][1] == "ADJ"]).most_common(n=n_most_common),
        "bigrams_verb_noun": Counter(
            [f"{b[0][0]}_{b[1][0]}" for b in contexts_bigrams
                if b[0][1] == "VERB" and b[1][1] == "NOUN"]).most_common(n=n_most_common),
        "bigrams_noun_noun": Counter(
            [f"{b[0][0]}_{b[1][0]}" for b in contexts_bigrams
                if b[0][1] == "NOUN" and b[1][1] == "NOUN"]).most_common(n=n_most_common),
        "bigrams_adj_noun": Counter(
            [f"{b[0][0]}_{b[1][0]}" for b in contexts_bigrams
                if b[0][1] == "ADJ" and b[1][1] == "NOUN"]).most_common(n=n_most_common),
        "trigrams": Counter(
            [f"{t[0][0]}_{t[1][0]}_{t[2][0]}" for t in contexts_trigrams]).most_common(
                n=n_most_common)}


def main():
    global nlp

    logging.info("Reading data files...")
    # Read data file
    with open(args.data, "r") as f:
        posts = f.read().split("\n")

    # Read list of words of interest
    with open(args.words_of_interst, "r") as f:
        words_of_interest = json.load(f)

    # Initialize spacy language model and customize the tokenizer to not split the words of interest
    # (this is necessary for word combinations, such as 'african-american')
    logging.info("Loading spacy model...")
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer = Tokenizer(
        nlp.vocab,
        rules={token: [{"ORTH": token}] for token in words_of_interest.keys()})

    # For all words of interest...
    statistics_by_woi = {}
    for woi, woi_forms in words_of_interest.items():
        print("")
        # Filter posts by occurence of words of interest
        logging.info(f"Filtering on posts containing '{woi}' and variants...")
        posts_of_interest = list(
            filter(lambda x: any([t in x.split() for t in woi_forms]), tqdm(posts)))

        # Replace alternative writing forms
        logging.info(f"Replacing all variants with '{woi}'...")
        woi_pattern = re.compile(" | ".join(woi_forms))
        posts_of_interest = [
            re.sub(woi_pattern, f" {woi} ", post) for post in tqdm(posts_of_interest)]

        # Tokenize the texts, clean from unwanted tokens and extract PoS tags
        logging.info("Processing posts and extracting tokens...")
        poi_tokenized = tokenize_and_tag_posts(posts_of_interest)

        # Extracting contexts of WOI
        logging.info("Extracting contexts...")
        woi_contexts = extract_context(woi, poi_tokenized)

        # Generate n-gram counts
        logging.info("Generating n-gram statistics...")
        statistics_by_woi[woi] = {
            "total_posts": len(posts_of_interest),
            "total_occurrences": len(woi_contexts),
            **generate_statistics(woi_contexts)}

    # Export statistics to file
    dt = datetime.today().strftime("%Y%m%d%H%M%S")
    output_file = path.join(args.output, f"lexical_corpus_evaluation_results-{dt}.json")

    logging.info(f"Exporting results to disk at {output_file}.")
    with open(output_file, "w") as f:
        json.dump({"corpus": path.basename(args.data), **statistics_by_woi}, f, indent=4)


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to conduct a lexical evaluation of the debate portal corpora.")

    parser.add_argument(
        "-d",
        "--data",
        required=True,
        type=str,
        help="Path to the debate portal data. Expects one whitespace separated post per line.",
        metavar="DATA_PATH")
    parser.add_argument(
        "-w",
        "--words_of_interst",
        required=True,
        type=str,
        help="Path to the file specifying the words of interests and their alternative spellings.",
        metavar="WOI_DEFINITION_PATH")
    parser.add_argument(
        "-m",
        "--most_common",
        default=10,
        type=int,
        help="Number of most common n-grams to extract.")
    parser.add_argument(
        "-p",
        "--positive_word_list",
        required=True,
        type=str,
        help="Path to a list of positive words",
        metavar="POSITIVE_WORD_LIST_PATH")
    parser.add_argument(
        "-n",
        "--negative_word_list",
        required=True,
        type=str,
        help="Path to a list of negative words",
        metavar="NEGATIVE_WORD_LIST_PATH")
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

    args = parser.parse_args()

    logging.basicConfig(**LOGGING_CONFIG)

    logging.info(
        "Please make sure that your input texts are whitespace separated tokens. The regex "
        "substitution might not work correctly otherwise.")

    main()
    print("Done.")
