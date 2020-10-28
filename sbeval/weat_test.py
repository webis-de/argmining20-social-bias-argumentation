# ############################################################################################### #
#                                                                                                 #
# Implements the a more efficient variant of the WEAT score calculation presented in [1].         #
#                                                                                                 #
# [1] https://doi.org/10.1126/science.aal4230                                                     #
#                                                                                                 #
# ############################################################################################### #


import logging
import numpy as np

from scipy.spatial.distance import cdist
from sys import intern

from sbeval.constants import LOGGING_CONFIG
from sbeval.word_vectors import WordVectors

logging.basicConfig(**LOGGING_CONFIG)


def _association_test(
        word_vectors: np.ndarray,
        attributes_a: np.ndarray,
        attributes_b: np.ndarray) -> float:
    """Calculate the association of a given word vector to the attribute matrices $A$ and $B$.

    Return the association value that resembles the relative similarity between the word and the
    two attribute matrices.

    In the original WEAT paper [1], the calculation is formulated as:
    $s(w, A, B)
        = mean_{a\in A} cos(\vec{w}, \vec{a}) - mean_{b\in B} cos(\vec{w}, \vec{b})$


    [1] https://doi.org/10.1126/science.aal4230

    Arguments:
    word_vector -- The word vectors for which the association should be calculated.
    attributes_a -- Matrix of word vectors for all attribute words in $A$.
    attributes_b -- Matrix of word vectors for all attribute words in $B$.
    """
    association_values_a = np.mean(cdist(word_vectors, attributes_a, metric="cosine"), axis=1)
    association_values_b = np.mean(cdist(word_vectors, attributes_b, metric="cosine"), axis=1)

    return np.subtract(association_values_a, association_values_b) * -1


def _differential_association_test(
        word_vectors_X: np.ndarray,
        word_vectors_Y: np.ndarray,
        attributes_a: np.ndarray,
        attributes_b: np.ndarray) -> float:
    """Calculate the difference between the associations of $X$ and $Y$ with $A$ and $B$.

    Return the differential association value that resembles the difference in relative similarity
    between the two target matrices to the two attribute matrices.

    A positive value denotes a closer association between $X$ and $A$, while a negative value
    denotes a closer association between $Y$ and $A$.

    In the original WEAT paper [1], the calculation is formulated as:
    $s(X, Y, A, B) = \sum_{x\in X} s(x, A, B) - \sum_{y\in Y} s(y, A, B)$, where the function $s()$
    is the association test between a word and two lists of attributes.


    [1] https://doi.org/10.1126/science.aal4230

    Arguments:
    word_vectors_X -- Matrix of word vectors for all target words in $X$.
    word_vectors_Y -- Matrix of word vectors for all target words in $Y$.
    attributes_a -- Matrix of word vectors for all attribute words in $A$.
    attributes_b -- Matrix of word vectors for all attribute words in $B$.
    """
    associations_sum_x = sum(
        _association_test(word_vectors_X, attributes_a, attributes_b))
    associations_sum_y = sum(
        _association_test(word_vectors_Y, attributes_a, attributes_b))

    return associations_sum_x - associations_sum_y


def _embed_token_list(token_list: list, word_vector_getter) -> tuple:
    """Transform a list of tokens to a list of word vectors. Return the list.

    If a token is found to be out-of-vocabulary, it will be added to a separate list that is
    returned alongside the list of vectors; the token will be excluded from the latter.

    Arguments:
    token_list -- A list of tokens that should be transformed.
    word_vector_getter -- An object that returns a vector given a word as parameter to the
                          `__getitem__()` function.
    """
    vector_list = []
    oov = []
    for token in token_list:
        try:
            vector_list.append(word_vector_getter[intern(token)])
        except KeyError:
            logging.debug(f"Token '{token}' is OOV. Ignoring.")
            oov.append(token)

    return (vector_list, oov)


def weat_score(
        target_words_X: list,
        target_words_Y: list,
        attribute_words_a: list,
        attribute_words_b: list,
        word_vector_getter=None) -> tuple:
    """Calculates the effect size of the differential association tests.

    Returns a tuple containing the result of the calculation and a list of OOV terms. The score
    simultaniously represents the WEAT score metric and can have values in the range between $-2$
    and $+2$.

    A positive value denotes a closer association between $X$ and $A$, while a negative value
    denotes a closer association between $Y$ and $A$.

    In the original WEAT paper [1], the calculation of the effect size if formulated as:
    $\frac{mean_{x\in X} s(x, A, B) - mean_{y\in Y} s(y, A, B)}{std\_dev_{w\in X\cup Y} s(w, A, B)}$


    [1] https://doi.org/10.1126/science.aal4230

    Arguments:
    target_words_X -- List of target words in $X$.
    target_words_Y -- List of target words in $Y$.
    attribute_words_a -- List of all attribute words in $A$.
    attribute_words_b -- List of all attribute words in $B$.
    word_vector_getter -- An object that returns a vector given a word as parameter to the
                          `__getitem__()` function. If `None`, the default is to use word2vec
                          embeddings, as loaded by the `WordVectors` class.
    """
    if not word_vector_getter:
        word_vector_getter = WordVectors("word2vec")

    # Retrieve all vectors for words in X, Y, A and B
    Xv, oov_x = _embed_token_list(target_words_X, word_vector_getter)
    Yv, oov_y = _embed_token_list(target_words_Y, word_vector_getter)
    Av, oov_a = _embed_token_list(attribute_words_a, word_vector_getter)
    Bv, oov_b = _embed_token_list(attribute_words_b, word_vector_getter)

    if len(Xv) == 0 or len(Yv) == 0 or len(Av) == 0 or len(Bv) == 0:
        raise AttributeError("For at least one of the given lexicons all tokens are OOV.")

    # Calculate effect size numerator
    association_X = _association_test(Xv, Av, Bv)
    association_Y = _association_test(Yv, Av, Bv)
    numerator = np.mean(association_X) - np.mean(association_Y)

    # Calculate effect size denominator
    denominator = np.std(np.concatenate((association_X, association_Y), axis=0))

    return (numerator / denominator, [*oov_x, *oov_y, *oov_a, *oov_b])
