"""Unit tests for the post evaluation of user performance in annotating Wikipedia


Created on May, 2021

@author: JRCasanova
"""
import unittest
from benchmarkers.account_old_annotations import account_for_older_annotations


class TestUserPerformance(unittest.TestCase):
    """Tests the functions for evaluating user performance for annotating wikipedia"""

    def test_account_for_older_annotations(self):
        """Tests the account_for_older_annotations method"""

        acro = "mbit/s"
        first_part_acro, second_part_acro = acro.split("/")
        predicted_expansions = {"mbit": "megabit", "s": "second"}
        actual_expansion = "megabit per second"

        predicted_expansions, tp, fp, fn = account_for_older_annotations(
            first_part_acro, second_part_acro, predicted_expansions, actual_expansion
        )

        self.assertEqual(predicted_expansions, {})
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        acro = "gbit/s"
        first_part_acro, second_part_acro = acro.split("/")
        predicted_expansions = {"gbit": "gigabit", "s": "second"}
        actual_expansion = "gigabit per second"

        predicted_expansions, tp, fp, fn = account_for_older_annotations(
            first_part_acro, second_part_acro, predicted_expansions, actual_expansion
        )

        self.assertEqual(predicted_expansions, {})
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        acro = "gbit/s"
        first_part_acro, second_part_acro = acro.split("/")
        predicted_expansions = {"gbit": "gigabit", "s": "second"}
        actual_expansion = "gigabit per bla second"

        predicted_expansions, tp, fp, fn = account_for_older_annotations(
            first_part_acro, second_part_acro, predicted_expansions, actual_expansion
        )

        self.assertEqual(predicted_expansions, {})
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        acro = "gbit/s"
        first_part_acro, second_part_acro = acro.split("/")
        predicted_expansions = {"gbit": "gigabit", "s": "second"}
        actual_expansion = "gigabit per bla bla second"

        predicted_expansions, tp, fp, fn = account_for_older_annotations(
            first_part_acro, second_part_acro, predicted_expansions, actual_expansion
        )

        self.assertEqual(predicted_expansions, {})
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

        acro = "gbit/s"
        first_part_acro, second_part_acro = acro.split("/")
        predicted_expansions = {"gbit": "gigabit"}
        actual_expansion = "gigabit per second"

        predicted_expansions, tp, fp, fn = account_for_older_annotations(
            first_part_acro, second_part_acro, predicted_expansions, actual_expansion
        )

        self.assertEqual(predicted_expansions, {})
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 1)

        acro = "os/2"
        first_part_acro, second_part_acro = acro.split("/")
        predicted_expansions = {"os": "operating system"}
        actual_expansion = "operating system 2"

        predicted_expansions, tp, fp, fn = account_for_older_annotations(
            first_part_acro, second_part_acro, predicted_expansions, actual_expansion
        )

        self.assertEqual(predicted_expansions, {})
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        acro = "4-d"
        first_part_acro, second_part_acro = acro.split("-")
        predicted_expansions = {"d": "dimensions"}
        actual_expansion = "4-dimensions"

        predicted_expansions, tp, fp, fn = account_for_older_annotations(
            first_part_acro, second_part_acro, predicted_expansions, actual_expansion
        )

        self.assertEqual(predicted_expansions, {})
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        acro = "4-d"
        first_part_acro, second_part_acro = acro.split("-")
        predicted_expansions = {"d": "dimension"}
        actual_expansion = "4-dimensions"

        predicted_expansions, tp, fp, fn = account_for_older_annotations(
            first_part_acro, second_part_acro, predicted_expansions, actual_expansion
        )

        self.assertEqual(predicted_expansions, {})
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        acro = "4-d"
        first_part_acro, second_part_acro = acro.split("-")
        predicted_expansions = {}
        actual_expansion = "4-dimensions"

        predicted_expansions, tp, fp, fn = account_for_older_annotations(
            first_part_acro, second_part_acro, predicted_expansions, actual_expansion
        )

        self.assertEqual(predicted_expansions, {})
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 1)


if __name__ == "__main__":
    unittest.main()