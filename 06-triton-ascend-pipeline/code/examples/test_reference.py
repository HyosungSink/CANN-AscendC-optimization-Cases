"""Small independent checks for boundary, tail and batch-offset semantics."""
import unittest

from part2_reference import full_page_reference


class FullPageReferenceTest(unittest.TestCase):
    def test_non_divisible_sequence_tail_is_untouched(self):
        actual = full_page_reference([0], [35], [7, 4, 9], 16)
        expected = list(range(112, 128)) + list(range(64, 80)) + [-1] * 3
        self.assertEqual(actual, expected)

    def test_partial_first_page_and_batch_offsets(self):
        actual = full_page_reference([5, 32], [36, 65], [7, 4, 9, 2, 8], 16)
        expected = (
            list(range(112, 128)) + [-1] * 15
            + list(range(144, 160)) + list(range(32, 48)) + [-1]
        )
        self.assertEqual(actual, expected)

    def test_no_new_full_page(self):
        self.assertEqual(full_page_reference([5], [15], [], 16), [-1] * 10)

    def test_exact_page(self):
        self.assertEqual(full_page_reference([16], [32], [3], 16), list(range(48, 64)))

    def test_empty_sequence(self):
        self.assertEqual(full_page_reference([16], [16], [], 16), [])

    def test_tail_beyond_one_1024_tile(self):
        actual = full_page_reference([0], [1091], list(range(70)), 16)
        self.assertEqual(actual[:1088], list(range(1088)))
        self.assertEqual(actual[1088:], [-1] * 3)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            full_page_reference([16], [8], [], 16)
        with self.assertRaises(ValueError):
            full_page_reference([0], [16], [], 16)


if __name__ == "__main__":
    unittest.main()
