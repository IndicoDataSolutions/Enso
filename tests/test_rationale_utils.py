import unittest
from enso.rationale_utils import get_rationales_by_sentences


class TestRationaleUtils(unittest.TestCase):
    def test_get_rationales_by_sentences(self):
        rationales = [
            {'start': 1},
            {'start': 2},
            {'start': 14},
            {'start': 18},
            {'start': 22},
            {'start': 30},
            {'start': 40}
        ]
        sentences = [
            'a',  # 0 - 2
            'bbb',  # 3 - 7
            'ccc',  # 8 - 12
            'ddd',  # 13 - 17
            'eee',  # 18 - 22
            'fff',  # 23 - 27
            'ggg',  # 28 - 32
            'hhh',  # 33 - 37
            'iii',  # 38 - 42
            'jjj',  # 43 - 48
        ]
        expected = [
            [{'start': 1}, {'start': 2}],
            [],
            [],
            [{'start': 14}],
            [{'start': 18}, {'start': 22}],
            [],
            [{'start': 30}],
            [],
            [{'start': 40}],
            []
        ]
        results = get_rationales_by_sentences(sentences, rationales)
        self.assertEqual(results, expected)