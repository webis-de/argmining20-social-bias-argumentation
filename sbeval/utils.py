from os import mkdir, path
from random import shuffle


def split_corpus(corpus: list, n: int, output_path: str, filename: str, random_state: int = 42):
    """Split the given corpus into $n$ random splits and saves them to the given location.

    Arguments:
    corpus -- A list of texts that should be split.
    n -- The number of splits to output.
    output_path -- The path to save the output files to.
    filename -- The base filename of the output split files.
    random_state -- The seed to be used for generating the random splits.
    """
    # Shuffle texts in the corpus
    shuffle(corpus)

    # Select every n-th item in the shuffled list, shifting the start at each iteration
    # https://stackoverflow.com/a/51838144
    shuffled_splits = [corpus[i::n] for i in range(0, n)]

    # Write splits to files
    for i, split in enumerate(shuffled_splits):
        split_dir = path.join(output_path, "splits")
        # Check if 'splits' sub-directory exists; create it if it doesn't
        if not path.isdir(split_dir):
            mkdir(split_dir)

        # Acutally do the write operation
        output_file = path.join(split_dir, f"{filename}__split{i}.txt")
        with open(output_file, "w") as f:
            f.write("\n".join(split).lower())
