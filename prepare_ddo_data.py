import argparse
import json
import logging
import os
import pandas as pd
import spacy

from os import path
from tqdm import tqdm

from sbeval.constants import LOGGING_CONFIG


def calculate_age_group(data: pd.DataFrame):
    """Calculate the age group of each user and add it is a column to the given DataFrame.

    Return the new DataFrame.

    Arguments:
    data -- The data for which the age groups should be calculated. Is expected to have a 'birthday'
            column.
    """
    # Year of the data collection of the ddo data sets
    data_collection_year = 2017

    # Limits of the age groups; we start at age 1, as 0 indicates a user that didn't specify a date
    # Also, 118 is the maximum age in the dataset
    age_groups = [1, 23, 118]
    age_group_labels = ["<23", "23+"]

    def _calculate_age(birthday: str):
        if birthday == "- Private -":
            return 0

        return data_collection_year - int(birthday.split(",")[1])

    # Calculate the age and bin the users into the pre-defined age groups
    data["tmp_age"] = data["birthday"].apply(_calculate_age)
    data["age_groups"] = pd.cut(data["tmp_age"], bins=age_groups, labels=age_group_labels)

    # Return the new DataFrame without the temporary age column
    return data.drop(labels=["tmp_age"], axis=1)


def extract_data(debates_data: dict, users_data: dict) -> pd.DataFrame:
    """Extract and combines debates and user data into a single dataframe. Return the dataframe.

    Currently, only the birthday, education, gender and political orientation are extracted and
    returned as user-defining features.

    Arguments:
    debates_data -- Dictionary containing the debates data.
    users_data -- Dictionary containing the users and their properties.
    """
    extracted_data = []
    properties_of_interest = ["birthday", "ethnicity", "gender", "political_ideology"]

    for key, debate in tqdm(debates_data.items()):
        # Sometimes, the users of the debate didn't exist anymore at the time
        # the data was collected.
        try:
            user1 = users_data[debate["participant_1_name"]]
        except KeyError:
            user1 = None

        try:
            user2 = users_data[debate["participant_2_name"]]
        except KeyError:
            user2 = None

        # If both users do not exist, skip this debate
        if not user1 and not user2:
            logging.debug("Both users are absent from debate data. Skipping.")
            continue

        # For each round in this debate...
        for debate_round in debate["rounds"]:
            # For each argument in this round...
            for argument in debate_round:
                arguing_user = (
                    user1 if argument["side"] == debate["participant_1_position"] else user2)

                # Skip this argument if arguing user does not exist in the dta
                if not arguing_user:
                    continue

                # Filtering for relevant properties
                properties = {
                    key: value
                    for key, value in arguing_user.items() if key in properties_of_interest}

                # Save the text and find the political ideology of the user.
                extracted_data.append({
                    "argument": argument["text"],
                    **properties})

    return pd.DataFrame(columns=["argument", *properties_of_interest], data=extracted_data)


def prepare_data(source_text: str) -> str:
    """Prepare the given text using spacy. Return the prepared text.

    The preparation currently manifests in the tokenization of the text and the removal of
    whitespace and url-like tokens. The returned text will be in whitespace tokenized form.

    Arguments:
    source_text -- The text that should be prepard.
    """
    doc = nlp(source_text)
    tokens = []
    for token in doc:
        if not token.is_space and not token.like_url:
            tokens.append(token.text)

    return " ".join(tokens)


def main():
    # Read data from disk
    logging.info("Reading data from disk...")
    with open(DEBATES_DATA_PATH, "r") as f:
        debates_data = json.load(f)

    with open(USERS_DATA_PATH, "r") as f:
        users_data = json.load(f)

    # Extract and combine data of interest
    logging.info("Collecting debate data...")
    combined_data = extract_data(debates_data, users_data)

    # Prepare textual data
    logging.info("Preparing debate data...")
    # Modin does not yet support `progress_apply`
    if args.multicore:
        combined_data["argument_prepared"] = combined_data["argument"].apply(prepare_data)
    else:
        combined_data["argument_prepared"] = combined_data["argument"].progress_apply(prepare_data)

    # Calculate age groups
    logging.info("Calculating age groups...")
    combined_data = calculate_age_group(combined_data)

    # Write final data to disk
    logging.info(f"Writing final dataframe to {OUTPUT_PATH}...")
    # Sort columns by labels before saving
    combined_data[sorted(combined_data.columns.values)].to_csv(OUTPUT_PATH, index=False)

    logging.info("Done.")


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to prepare the debate-org data to the necessary format.")
    parser.add_argument(
        "--debates",
        "-d",
        required=True,
        help="The path to the debates data file.",
        metavar="DEBATES_DATA_PATH")
    parser.add_argument(
        "--users",
        "-u",
        required=True,
        help="The path to the users data file.",
        metavar="USERS_DATA_PATH")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Directory to save the output to.",
        metavar="OUTPUT_DIR")
    parser.add_argument(
        "--multicore",
        "-m",
        default=False,
        action="store_true",
        help="Whether to use multiple cores to speed up the preprocessing process.")
    parser.add_argument(
        "--processing_cores",
        "-p",
        default=2,
        type=int,
        help="The number of processing cores to use. Only relevant if multicore flag is set.",
        metavar="PROCESSING_CORES")

    args = parser.parse_args()

    # Set data variables
    DEBATES_DATA_PATH = args.debates
    USERS_DATA_PATH = args.users
    OUTPUT_PATH = path.join(args.output, "debates_data-prepared.csv")

    logging.basicConfig(**LOGGING_CONFIG)

    # Import pandas if multicore is disabled, otherwise import modin
    if args.multicore:
        logging.info("Multicore enabled; using modin instead of pandas with {0} cores.".format(
            args.processing_cores))
        os.environ["MODIN_CPUS"] = str(args.processing_cores)
        import modin.pandas as pd
    else:
        # Enable tqdm pandas integration
        tqdm.pandas()

    # Initialize spacy model
    nlp = spacy.load("en_core_web_sm")

    main()
