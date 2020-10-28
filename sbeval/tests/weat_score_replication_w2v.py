import json
import unittest

from os import path

from ..constants import WEAT_TEST_TOLERANCE
from ..word_vectors import WordVectors
from ..weat_test import weat_score


class TestWeatWord2vec(unittest.TestCase):
    # Load test data from file
    with open(path.join("ddo", "tests", "weat_tests.json"), "r") as f:
        weat_tests = json.load(f)

    # Load word vectors
    vectors = WordVectors("word2vec")

    def _calculate_weat_score(self, test_data):
        """Simple test helper function to calculate and return the WEAT score for given data."""
        # Make the out-of-vocabulary tokens easier to access
        oovs = test_data["out_of_vocabularies"]["word2vec"]

        # Generate and return test scores
        return weat_score(
            target_words_X=[x for x in test_data["X"] if x not in oovs["X"]],
            target_words_Y=[y for y in test_data["Y"] if y not in oovs["Y"]],
            attribute_words_a=[a for a in test_data["A"] if a not in oovs["A"]],
            attribute_words_b=[b for b in test_data["B"] if b not in oovs["B"]],
            word_vector_getter=self.__class__.vectors)[0]

    def test_one(self):
        # Retrieving test data
        test_data = self.__class__.weat_tests["test1"]

        self.assertAlmostEqual(
            self._calculate_weat_score(test_data),
            test_data["word2vec_result"],
            delta=WEAT_TEST_TOLERANCE)

    def test_two(self):
        # Retrieving test data
        test_data = self.__class__.weat_tests["test2"]

        self.assertAlmostEqual(
            self._calculate_weat_score(test_data),
            test_data["word2vec_result"],
            delta=WEAT_TEST_TOLERANCE)

    def test_three(self):
        # Retrieving test data
        test_data = self.__class__.weat_tests["test3"]

        self.assertAlmostEqual(
            self._calculate_weat_score(test_data),
            test_data["word2vec_result"],
            delta=WEAT_TEST_TOLERANCE)

    def test_four(self):
        # Retrieving test data
        test_data = self.__class__.weat_tests["test4"]

        self.assertAlmostEqual(
            self._calculate_weat_score(test_data),
            test_data["word2vec_result"],
            delta=WEAT_TEST_TOLERANCE)

    def test_five(self):
        # Retrieving test data
        test_data = self.__class__.weat_tests["test5"]

        self.assertAlmostEqual(
            self._calculate_weat_score(test_data),
            test_data["word2vec_result"],
            delta=WEAT_TEST_TOLERANCE)

    def test_six(self):
        # Retrieving test data
        test_data = self.__class__.weat_tests["test6"]

        self.assertAlmostEqual(
            self._calculate_weat_score(test_data),
            test_data["word2vec_result"],
            delta=WEAT_TEST_TOLERANCE)

    def test_seven(self):
        # Retrieving test data
        test_data = self.__class__.weat_tests["test7"]

        self.assertAlmostEqual(
            self._calculate_weat_score(test_data),
            test_data["word2vec_result"],
            delta=WEAT_TEST_TOLERANCE)

    def test_eight(self):
        # Retrieving test data
        test_data = self.__class__.weat_tests["test8"]

        self.assertAlmostEqual(
            self._calculate_weat_score(test_data),
            test_data["word2vec_result"],
            delta=WEAT_TEST_TOLERANCE)

    def test_nine(self):
        # Retrieving test data
        test_data = self.__class__.weat_tests["test9"]

        self.assertAlmostEqual(
            self._calculate_weat_score(test_data),
            test_data["word2vec_result"],
            delta=WEAT_TEST_TOLERANCE)

    def test_ten(self):
        # Retrieving test data
        test_data = self.__class__.weat_tests["test10"]

        self.assertAlmostEqual(
            self._calculate_weat_score(test_data),
            test_data["word2vec_result"],
            delta=WEAT_TEST_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
