import argparse
import json
import logging

from sbeval.constants import LOGGING_CONFIG


def main():
    # Read evaluations file
    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Read WEAT lexicons
    with open(args.weat_lexicons, "r", encoding="utf-8") as f:
        weat_tests = json.load(f)

    logging.info(f"Printing accumulated results for corpus '{results['corpus']}'.")

    # For each test result...
    for test in results.keys():
        # If the current key is "corpus", we can safely skip it
        if test == "corpus":
            continue

        # Some of the lexicons have swapped attribute/target sets; in those cases, the tests might
        # need special treatment
        swapped_test_lexicons = ["test7", "test8"]

        print("\n\n" + "=" * 45)
        print(f"Current test: {test}")
        print("-" * 45)

        # Adopt the printed output to swapped tests
        if test in swapped_test_lexicons:
            target_descriptions = weat_tests[test]["attribute_words"].split(" vs. ")
        else:
            target_descriptions = weat_tests[test]["target_words"].split(" vs. ")

        # For each target lexicon
        for i, target in enumerate(results[test].keys()):
            print(target, target_descriptions[i])

            # For swapped lexicons, we also need to swap the X and Y groups before we lowercase all
            # words
            if test in swapped_test_lexicons:
                if target == "X":
                    all_target_words = [w.lower() for w in weat_tests[test]["A"] if w.lower()]
                elif target == "Y":
                    all_target_words = [w.lower() for w in weat_tests[test]["B"] if w.lower()]
            else:
                all_target_words = [w.lower() for w in weat_tests[test][target] if w.lower()]

            # Collect all target words for which to accumulate the results from
            relevant_target_keys = [
                p for p in results[test][target].keys() if p in all_target_words]

            target_total_pleasant = 0
            target_total_unpleasant = 0

            # Accumulate the counts for each of the target words identified above
            for word_key in relevant_target_keys:
                # Again, we need to treat the swapped lexicons as special case here
                if test in swapped_test_lexicons:
                    relevant_pleasant_words = [
                        w.lower()
                        for w in weat_tests[test]["X"]
                        if w.lower() in results[test][target][word_key]]
                    relevant_unpleasant_words = [
                        w.lower()
                        for w in weat_tests[test]["Y"]
                        if w.lower() in results[test][target][word_key]]
                else:
                    relevant_pleasant_words = [
                        w.lower()
                        for w in weat_tests[test]["A"]
                        if w.lower() in results[test][target][word_key]]
                    relevant_unpleasant_words = [
                        w.lower()
                        for w in weat_tests[test]["B"]
                        if w.lower() in results[test][target][word_key]]

                # Add up all counts
                pleasant_counts = sum([
                    results[test][target][word_key][p_word]
                    for p_word in relevant_pleasant_words])
                unpleasant_counts = sum([
                    results[test][target][word_key][u_word]
                    for u_word in relevant_unpleasant_words])

                target_total_pleasant += pleasant_counts
                target_total_unpleasant += unpleasant_counts

            # Print results to the console
            print("Association-A:Association-B")
            print(f"{target_total_pleasant}:{target_total_unpleasant}")


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to accumulate the weat co-occurrence counts.")
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the co-occurrence analysis results file.",
        metavar="INPUT_FILE")
    parser.add_argument(
        "--weat_lexicons",
        "-t",
        required=True,
        help="Path to the file that specifies the WEAT lexicons.",
        metavar="WEAT_LEXICONS")

    args = parser.parse_args()

    logging.basicConfig(**LOGGING_CONFIG)

    main()
    print("Done.")
