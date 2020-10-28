import argparse
import logging
import pandas as pd

from os import path
from sqlalchemy import create_engine

from sbeval.constants import LOGGING_CONFIG


def main():
    # List to save all posts from all databases
    all_posts = []
    debate_portal_db_names = [
        args.convinceme_db_name, args.createdebate_db_name, args.fourforums_db_name]

    for debate_portal in debate_portal_db_names:
        if debate_portal == "":
            continue

        logging.info(f"Working on debate portal '{debate_portal}'...")
        db_address = "mysql://" + \
            f"{args.mysql_user}:{args.mysql_password}@{args.mysql_address}/{debate_portal}"

        # Initialize db connection
        logging.info("Connecting to database.")
        engine = create_engine(db_address)
        db_connection = engine.connect()

        # Read textual post data from database
        logging.info("Reading post data from database...")
        posts_df = pd.read_sql("SELECT * FROM post_view", db_connection)
        debate_portal_posts = [text.replace("\n", " ").lower() for text in posts_df.text.values]
        all_posts.extend(debate_portal_posts)

        # Write data of single debate portal to file
        output_file = path.join(args.output, f"iacv2-{debate_portal}-texts.txt")
        logging.info(f"Writing posts to text file at {output_file}.")
        with open(output_file, "w") as f:
            f.write("\n".join(debate_portal_posts))

        logging.info(f"Debate portal '{debate_portal}' done.\n")

    # Write file of all included debate portal into a combined file
    output_file = path.join(args.output, "iacv2-combined-texts.txt")
    logging.info(f"Writing combined posts (all portals) to text file at {output_file}.")
    with open(output_file, "w") as f:
        f.write("\n".join(all_posts))


if __name__ == "__main__":
    # Add cli parameters
    parser = argparse.ArgumentParser(
        "A script to prepare the Internet Argument Corpus v2 data to a plain text format.")
    parser.add_argument(
        "--mysql_address",
        "-a",
        required=True,
        help="Address of the MySQL server that provides read access to the IAC databases.",
        metavar="MYSQL_ADDRESS")
    parser.add_argument(
        "--mysql_user",
        "-u",
        required=True,
        help="User that is able to access the database at the specified MySQL address.",
        metavar="MYSQL_USER")
    parser.add_argument(
        "--mysql_password",
        "-p",
        required=True,
        help="Password for the given MySQL user.",
        metavar="MYSQL_PASSWORD")
    parser.add_argument(
        "--convinceme_db_name",
        "-c",
        default="",
        help="Database name containing the data for the 'convinceme' debate portal.",
        metavar="CONVINCEME_DB_NAME")
    parser.add_argument(
        "--createdebate_db_name",
        "-d",
        default="",
        help="Database name containing the data for the 'createdebate' debate portal.",
        metavar="CREATEDEBATE_DB_NAME")
    parser.add_argument(
        "--fourforums_db_name",
        "-f",
        default="",
        help="Database name containing the data for the 'fourforums' debate portal.",
        metavar="FOURFORUMS_DB_NAME")
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
