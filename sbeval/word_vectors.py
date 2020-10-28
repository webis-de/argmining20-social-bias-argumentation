import logging
import numpy as np

from abc import ABC, abstractmethod
from gensim.models import KeyedVectors
from os import path

from sbeval.constants import LOGGING_CONFIG, WORD_VECTOR_DIR

logging.basicConfig(**LOGGING_CONFIG)


class WordVectors:
    """A basic class to load and retrieve vectors for words using different methods.

    Current available methods are 'word2vec' (default), 'glove', 'conceptnet'.
    """

    def __init__(self, method: str = "word2vec"):
        embedding_classes = {
            "word2vec": Word2VecEmbeddings,
            "glove": GloVeEmbeddings,
            "conceptnet": ConceptNetNumberbatchEmbeddings}

        self.embeddings = embedding_classes[method]()

    def __getitem__(self, token: str) -> np.ndarray:
        """Get a vector for the corresponding token string."""
        return self.embeddings[token]


class BaseEmbeddings(ABC):
    """The base class for all embedding classes."""

    @abstractmethod
    def _load_embeddings(self, path: str = None) -> None:
        """Abstract class for loading pre-trained word vectors, possibly from a file."""
        pass

    @abstractmethod
    def __getitem__(self, token: str) -> np.ndarray:
        """Abstract class for retrieving a vector for a given token string."""
        pass


class Word2VecEmbeddings(BaseEmbeddings):
    """Class that provides easy access to the 300-dimensional GoogleNews word2vec embeddings."""

    def __init__(self):
        logging.debug("Initialized word2vec embeddings.")
        self.embeddings = self._load_embeddings(
            path.join(WORD_VECTOR_DIR, "GoogleNews-vectors-negative300.bin"))

    def _load_embeddings(self, path: str) -> None:
        """Load the pretrained word2vec embeddings from the given path using gensim.

        Return the embeddings objects that is able to handle calls to __getitem__.

        Arguments:
        path -- The path to the word vector file.
        """
        logging.debug("Loading word2vec embeddings.")
        return KeyedVectors.load_word2vec_format(path, binary=True)

    def __getitem__(self, token: str) -> np.ndarray:
        """Get the vector for the given token string.

        Return the vector as numpy array.

        Arguments:
        token -- The token for which a vector should be returned.
        """
        logging.debug(f"Retrieving and returning word2vec vector for '{token}'.")
        try:
            return self.embeddings[token]
        except KeyError as e:
            logging.debug("Couldn't find token. Trying to split it by hyphen or space.")
            if " " in token:
                tokens = token.split(" ")
            elif "-" in token:
                tokens = token.split("-")
            else:
                raise e

            # If token as either hyphen or space separated,
            # return the mean vector of both embeddings
            logging.debug("Successfully split token by hyphen or space. \
                Will return mean of both vectors, if possible.")
            token_embeds = [self.embeddings[t] for t in tokens]
            return np.mean(token_embeds, axis=0)


class GloVeEmbeddings(BaseEmbeddings):
    """Class that provides easy access to the 840B, 300-dimensional GloVe embeddings.

    On the webpage [1], they are specified as:
    "Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)".

    [1]: https://nlp.stanford.edu/projects/glove/
    """

    def __init__(self):
        logging.debug("Initialized GloVe embeddings.")
        self.embeddings = self._load_embeddings(
            path.join(WORD_VECTOR_DIR, "glove.840B.300d_word2vec-format.txt"))

    def _load_embeddings(self, path: str) -> None:
        """Load the pretrained GloVe embeddings from the given path using gensim.

        Return the embeddings objects that is able to handle calls to __getitem__.

        Arguments:
        path -- The path to the word vector file.
        """
        logging.debug("Loading GloVe embeddings.")
        return KeyedVectors.load_word2vec_format(path)

    def __getitem__(self, token: str) -> np.ndarray:
        """Get the vector for the given token string.

        Return the vector as numpy array.

        Arguments:
        token -- The token for which a vector should be returned.
        """
        logging.debug(f"Retrieving and returning glove vector for '{token}'.")
        try:
            return self.embeddings[token]
        except KeyError as e:
            logging.debug("Couldn't find token. Trying to split it by hyphen or space.")
            if " " in token:
                tokens = token.split(" ")
            elif "-" in token:
                tokens = token.split("-")
            else:
                raise e

            # If token as either hyphen or space separated,
            # return the mean vector of both embeddings
            logging.debug("Successfully split token by hyphen or space. \
                Will return mean of both vectors, if possible.")
            token_embeds = [self.embeddings[t] for t in tokens]
            return np.mean(token_embeds, axis=0)


class ConceptNetNumberbatchEmbeddings(BaseEmbeddings):
    """Class that provides easy access to the 300-dimensional ConceptNet Numberbatch embeddings."""

    def __init__(self):
        logging.debug("Initialized conceptnet embeddings.")
        self.embeddings = self._load_embeddings(
            path.join(WORD_VECTOR_DIR, "numberbatch-en.txt"))

    def _load_embeddings(self, path: str) -> None:
        """Load the pretrained conceptnet embeddings from the given path using gensim.

        Return the embeddings objects that is able to handle calls to __getitem__.

        Arguments:
        path -- The path to the word vector file.
        """
        logging.debug("Loading conceptnet embeddings.")
        return KeyedVectors.load_word2vec_format(path, binary=False)

    def __getitem__(self, token: str) -> np.ndarray:
        """Get the vector for the given token string.

        Return the vector as numpy array.

        Arguments:
        token -- The token for which a vector should be returned.
        """
        logging.debug(f"Retrieving and returning conceptnet vector for '{token}'.")
        try:
            return self.embeddings[token]
        except KeyError as e:
            logging.debug("Couldn't find token. Trying to split it by hyphen or space.")
            if " " in token:
                tokens = token.split(" ")
            elif "-" in token:
                tokens = token.split("-")
            else:
                raise e

            # If token as either hyphen or space separated,
            # return the mean vector of both embeddings
            logging.debug("Successfully split token by hyphen or space. \
                Will return mean of both vectors, if possible.")
            token_embeds = [self.embeddings[t] for t in tokens]
            return np.mean(token_embeds, axis=0)


class CustomEmbeddings(BaseEmbeddings):
    """Class that provides easy access to loading custom embeddings in word2vec text format.

    Arguments:
    embeddings_path -- Path to the embeddings file. Vectors are expected to be present in word2vec
                       format and in non-binary text format.
    """

    def __init__(self, embeddings_path: str):
        logging.debug("Initialized custom embeddings.")
        self.embeddings = self._load_embeddings(embeddings_path)

    def _load_embeddings(self, embeddings_path: str) -> None:
        """Load the pretrained custom embeddings from the given path using gensim.

        Return the embeddings objects that is able to handle calls to __getitem__.

        Arguments:
        embeddings_path -- The path to the word vector file.
        """
        logging.debug("Loading custom embeddings.")

        # Determine if the format is bianry or not, based on the file extension
        binary_format = path.splitext(embeddings_path)[1] == ".bin"

        return KeyedVectors.load_word2vec_format(embeddings_path, binary=binary_format)

    def __getitem__(self, token: str) -> np.ndarray:
        """Get the vector for the given token string.

        Return the vector as numpy array.

        Arguments:
        token -- The token for which a vector should be returned.
        """
        logging.debug(f"Retrieving and returning custom vector for '{token}'.")
        try:
            return self.embeddings[token]
        except KeyError as e:
            logging.debug("Couldn't find token. Trying to split it by hyphen or space.")
            if " " in token:
                tokens = token.split(" ")
            elif "-" in token:
                tokens = token.split("-")
            else:
                raise e

            # If token as either hyphen or space separated,
            # return the mean vector of both embeddings
            logging.debug("Successfully split token by hyphen or space. \
                Will return mean of both vectors, if possible.")
            token_embeds = [self.embeddings[t] for t in tokens]
            return np.mean(token_embeds, axis=0)
