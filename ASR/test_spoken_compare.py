"""Regression tests for spoken ASR comparison false-positive policy."""

from __future__ import annotations

import unittest

from ASR.spoken_compare import compare_spoken, normalize_spoken, spoken_token_equivalent


class SpokenCompareRegressionTests(unittest.TestCase):
    """Keep audited sound-alike, split-word, and stutter behavior narrow."""

    def test_audited_sound_alikes_and_split_words_pass(self):
        """Accept exact spoken equivalents that differ only in ASR spelling or spacing."""
        cases = [
            (
                "On my left Herr Wieland, the German master, thirty-odd years old.",
                "On my left her Weiland, the German master, thirty-odd years old.",
            ),
            (
                "I got into conversation with Herr Wieland.",
                "I got into conversation with Herwylund.",
            ),
            (
                "Go to Maiwand. Trouble at Maiwand?",
                "Go to my wand. Trouble at my wand?",
            ),
            (
                "I fumbled around on the bedside table until I found a match.",
                "E fumbled around on the bedside table until I found a match.",
            ),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertTrue(compare_spoken(reference, hypothesis, threshold=0.65)["passed"])

    def test_missing_negation_stays_failed_without_consuming_a_neighbor(self):
        """Keep a missing No visible instead of misreporting an exact hand as absent."""
        result = compare_spoken(
            "an iron hand. No easy task, I gather.",
            "an iron hand. Azy task, I gather.",
            threshold=0.65,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_tokens"], ["no"])
        self.assertNotIn("hand", result["missing_tokens"])

    def test_one_adjacent_noncritical_word_stutter_passes(self):
        """Ignore exactly one direct duplicate while preserving the rest of a sentence."""
        cases = [
            (
                "Graves frowned. He has occasionally had a slight lapse of memory.",
                "Graves frowned, frowned. He has occasionally had a slight lapse of memory.",
            ),
            (
                "We shall have to think about getting ready for dinner. Indeed.",
                "We shall have to think about getting ready for dinner. Indeed, indeed.",
            ),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertTrue(compare_spoken(reference, hypothesis, threshold=0.65)["passed"])

    def test_phrase_repeats_and_arbitrary_negation_substitutions_still_fail(self):
        """Keep multiword repeats and unrelated replacements visible as failures."""
        repeated_phrase = compare_spoken(
            "The house was not built directly upon the old foundations, then? I asked.",
            "The house was not built directly upon the old foundations, then? I asked. I asked, oh!",
            threshold=0.65,
        )
        self.assertFalse(repeated_phrase["passed"])
        self.assertEqual(spoken_token_equivalent("no", "hand"), "none")

    def test_contraction_sentence_boundary_does_not_become_initialism(self):
        """Keep an apostrophe tail plus sentence-start I out of acronym joining."""
        reference = "I just couldn't— I know."
        hypothesis = "I just couldn't. I know."
        result = compare_spoken(reference, hypothesis, threshold=0.99)

        self.assertEqual(normalize_spoken(hypothesis), "i just could not i know")
        self.assertEqual(normalize_spoken("D. C. office."), "dc office")
        self.assertEqual(normalize_spoken("N. I. S systems."), "nis systems")
        self.assertTrue(result["passed"])

    def test_exact_surface_word_fusions_are_limited_to_literal_surface_matches(self):
        """Accept literal fusions and never label altered or numeric spans as such."""
        cases = [
            ("Tran's face paled.", "Transface paled."),
            ("Black goop covered it.", "Blackgoop covered it."),
            ("She's ready now.", "Shesready now."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis, threshold=0.99)
                self.assertTrue(result["passed"])
                self.assertTrue(any(
                    item.get("kind") == "exact_surface_word_fusion"
                    for item in result["accepted_phrase_equivalences"]
                ))

        changed = compare_spoken(
            "Black goop covered it.",
            "Blackcoop covered it.",
            threshold=0.99,
        )
        numeric = compare_spoken(
            "Pioneer Metro Cube three eight.",
            "Pioneer Metro Cubethree eight.",
            threshold=0.99,
        )
        for result in (changed, numeric):
            self.assertFalse(any(
                item.get("kind") == "exact_surface_word_fusion"
                for item in result["accepted_phrase_equivalences"]
            ))

    def test_compact_asr_identifier_sequences_pass(self):
        """Accept digit-by-digit identifiers rendered compactly by ASR."""
        cases = [
            (
                "Pioneer Metro 04 10 twenty-one fifty-five. eleven twenty hours.",
                "Pioneer Metro 0410-2155. 1120 hours.",
            ),
            (
                "8 1 2 3 2 3 4 is offline - 4 0 2 0 8 0 2. What does it mean? Niko asked.",
                "8123234 is offline, 4SO2-08SO2. What does it mean? Nico asked.",
            ),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"])
                self.assertIn("<IDSEQ>", result["ref_normalized"])

    def test_spoken_identifier_separators_are_contextual(self):
        """Treat spoken punctuation bridges as structural only between ID slots."""
        cases = [
            (
                "The route is 3 8 dash 1 dash 4.",
                "The route is 38-1-4.",
            ),
            (
                "The route is 3 8 dot 1 dot 1.",
                "The route is 38.11.",
            ),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertTrue(compare_spoken(reference, hypothesis)["passed"])

        # A normal conjunction remains spoken prose, not an identifier bridge.
        self.assertFalse(compare_spoken(
            "The route is 3 and 4.",
            "The route is 34.",
        )["passed"])

    def test_listener_confirmed_short_context_fuzzies_skip_regeneration(self):
        """Classify only approved one-token context repairs as fuzzy accepts."""
        for reference, hypothesis in [
            ("I would bet my life on it.", "I would bet my life on him."),
            ("I know that answer.", "I know them answer."),
        ]:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis)
                self.assertTrue(result["passed"])
                self.assertEqual(result["classification"], "FUZZY_ACCEPT")
                self.assertTrue(any(
                    item.get("kind") == "context_repaired_short_token"
                    for item in result["accepted_equivalences"]
                ))
        self.assertTrue(compare_spoken(
            "She had nearly gotten knocked out of the fighting once.",
            "She had nearly gotten knocked out of the fighting ones.",
        )["passed"])

    def test_unapproved_short_substitutions_and_negations_stay_failed(self):
        """Keep arbitrary short-word swaps and negation changes as real failures."""
        cases = [
            ("I saw him leave.", "I saw her leave."),
            ("I will not go.", "I will now go."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertFalse(compare_spoken(reference, hypothesis)["passed"])

    def test_high_coverage_listener_fuzzy_is_bounded(self):
        """Accept a close long resegmentation but retain added-number failures."""
        fuzzy = compare_spoken(
            "It was easy to contact someone from badge to badge, all he had to do was say their name. Why hadn't they made the general comms as simple?",
            "It was easy to contact someone from batch to batch. All he had to do was say their name. Why hadn't they made the general comms as simple?",
        )
        self.assertTrue(fuzzy["passed"])
        self.assertEqual(fuzzy["classification"], "FUZZY_ACCEPT")

        added_number = compare_spoken(
            "This is Oslo. Oslo, it's Grant. I'm in the galley on deck eight. We're having a power issue down here. All of the lights are out.",
            "This is Oslo. Oslo. It's grand. I'm in the galley on deck 8. 8. We're having a power issue down here. All of the lights around.",
        )
        self.assertFalse(added_number["passed"])

    def test_by_what_question_fuzzy_does_not_weaken_other_questions(self):
        """Accept only the listener-confirmed question rendering around ``by``."""
        fuzzy = compare_spoken(
            "Sometimes I feel trapped. By what? I don't know. Myself?",
            "Sometimes I feel trapped, by one, I don't know, myself?",
        )
        self.assertTrue(fuzzy["passed"])
        self.assertEqual(fuzzy["classification"], "FUZZY_ACCEPT")
        self.assertFalse(compare_spoken("From what?", "From one.")["passed"])


if __name__ == "__main__":
    unittest.main()
