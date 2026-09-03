"""Regression tests for spoken ASR comparison false-positive policy."""

from __future__ import annotations

import unittest

from ASR.spoken_compare import (
    _HOMOGRAPH_WHITELIST_PATH,
    _boundary_resegmentation_mode,
    compare_spoken,
    normalize_spoken,
    spoken_token_equivalent,
)


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

    def test_full_initialism_equivalence_skips_later_boundary_rewrites(self):
        """Keep C-I-C and CIC equal without manufacturing a second CI token."""
        result = compare_spoken(
            "Joseph didn't answer, crossing through the former C-I-C to the armory.",
            "Joseph didn't answer, crossing through the former CIC to the armory.",
            threshold=0.65,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["ref_normalized"], result["hyp_normalized"])
        self.assertNotIn("raw_compact_boundary_surface", str(result["accepted_phrase_equivalences"]))

    def test_apostrophe_boundary_matching_preserves_adjacent_speech(self):
        """Reject fuzzy contraction boundaries that would delete a subject or object."""
        leading_subject = compare_spoken(
            "We aren't up to that yet, Preslan said.",
            "We aren't up to that yet, Preslin said.",
            threshold=0.65,
        )
        directional_object = compare_spoken(
            "Preslan turned t'ward her.",
            "Preslin turned toward her.",
            threshold=0.65,
        )

        self.assertTrue(leading_subject["passed"])
        self.assertTrue(directional_object["passed"])
        self.assertNotIn("we", leading_subject["extra_tokens"])
        self.assertNotIn("her", directional_object["extra_tokens"])

    def test_listener_confirmed_extra_filler_remains_visible(self):
        """Do not suppress audible internal or trailing filler insertions."""
        result = compare_spoken(
            "Your username has the second highest score in history. Seriously?",
            "Your username has the second highest score in history. Um, seriously?",
            threshold=0.65,
        )

        self.assertFalse(result["passed"])
        self.assertIn("uh", result["extra_tokens"])

    def test_spelled_reduction_expands_before_alignment(self):
        """Treat an ordinary written reduction as its fully spoken phrase."""
        result = compare_spoken(
            "Give me the cartridge.",
            "Gimme the cartridge.",
            threshold=0.65,
        )

        self.assertTrue(result["passed"])

    def test_listener_verified_regeneration_renderings_pass(self):
        """Keep all listener-verified Folder 1 regeneration transcripts out of retry."""
        cases = [
            (
                "You know how many people there are waiting for you down there? There's my four boys plus four locals. None of em are what I'd call peaceful citizens.",
                "You know how many people there are waiting for you down there. There's my four boys plus four locals. None of them are what I'd call peaceful citizens.",
                "raw_normalized_equivalence",
            ),
            (
                "The circles in the center represent a marksman's target, a bull's-eye. It's a marksman's medal. The girl's eyes danced and her cheeks puffed with air.",
                "The circles in the center represent a marksman's target, a bullseye. It's a marksman's medal. The girl's eyes danced and her cheeks puffed with air.",
                "raw_compact_boundary_surface",
            ),
            (
                "Bol'un/Peterson chuckled and said, You're sure of that, eh? As sure as anyone can ever be. Well I'm interested in the Expo Seventy-Four angle. The guy sighed.",
                "Bolan Peterson chuckled and said, you're sure of that, eh? As sure as anyone can ever be. Well, I'm interested in the Expo's 74 angle. The guy sighed.",
                "slash_joined_name_equivalence",
            ),
            (
                "Seriously. I'm worried sick. This country could topple. It's that serious? It is. I wish I could tell you, no I don't. I wouldn't burden anyone with that.",
                "Seriously, I'm worried sick. This country could topple. It's that serious. It is. I wish I could tell you. No, I don't. I wouldn't burden anyone with that.",
                "raw_formatting_equivalence",
            ),
            (
                "Likes em for lunch, likes em for dinner, and now and then for a midnight snack. I think you've been servicing the guy. I'm trying to locate him.",
                "Likes him for lunch, likes him for dinner, and now and then for a midnight snack. I think you've been servicing the guy. I'm trying to locate him.",
                "reduced_pronoun_equivalence",
            ),
            (
                "You're Low Boy. I'm High Boy. Right. Radio silence, though, unless you get lost. Right. How soon, Jack? Let's see, what will I need? Guts and skill.",
                "You're a low boy, I'm high boy, right. Radio silence though, unless you get lost, right. How soon, Jack? Let's see, what will I need? Guts and skill.",
                "call_sign_article_equivalence",
            ),
            (
                "Head weapon for the mission was Bol'un's favorite heavy piece, the M-Sixteen/M-Seventy-Nine over n under combo.",
                "Head weapon for the mission was Boland's favorite heavy piece, the M16-M79 over and under combo.",
                "spoken_letter_number_code_equivalence",
            ),
            (
                "Grimaldi punched the channel selector and gave Bol'un a visual go-ahead. Go ahead, Low Boy, Bol'un replied. Okay, they're sprung and scrambling.",
                "Grimaldi punched the channel selector and gave Bolin a visual go ahead. Go ahead, low boy, Bolin replied. Okay, they're sprung and scrambling.",
                None,
            ),
        ]
        for reference, hypothesis, rule in cases:
            with self.subTest(rule=rule):
                result = compare_spoken(reference, hypothesis, threshold=0.65)
                self.assertTrue(result["passed"])
                self.assertEqual(result["score"], 1.0)
                if rule is not None:
                    self.assertTrue(any(
                        item.get("kind") == rule
                        for item in result["accepted_phrase_equivalences"]
                    ))

    def test_special_apostrophe_fusions_and_generic_boundaries_remain_distinct(self):
        """Keep apostrophe recovery raw while exposing ordinary joins in alignment."""
        cases = [
            ("Tran's face paled.", "Transface paled."),
            ("She's ready now.", "Shesready now."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis, threshold=0.99)
                self.assertTrue(result["passed"])
                self.assertTrue(any(
                    item.get("kind") in {"exact_surface_word_fusion", "exact_surface_word_split"}
                    and item.get("specialized_rule") == "apostrophe_surface"
                    for item in result["accepted_phrase_equivalences"]
                ))

        ordinary = compare_spoken(
            "Black goop covered it.",
            "Blackgoop covered it.",
            threshold=0.99,
        )
        self.assertTrue(ordinary["passed"])
        self.assertTrue(any(
            item["op"] == "boundary_resegmentation"
            and item.get("boundary_method") == "exact_surface"
            for item in ordinary["alignment_operations"]
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
        self.assertFalse(changed["passed"])
        for result in (changed, numeric):
            self.assertFalse(any(
                item["op"] == "boundary_resegmentation"
                for item in result["alignment_operations"]
            ))

    def test_configured_homograph_whitelist_restores_record_respellings(self):
        """Accept configured record pronunciation spellings in either direction."""
        self.assertEqual(_HOMOGRAPH_WHITELIST_PATH.parent.name, "config")
        cases = [
            (
                "That's my rec urd so far. The timer hit zero and vanished.",
                "That's my record so far. The timer hit zero and vanished.",
                "ref_2_to_hyp_1",
            ),
            (
                "That's my record so far. The timer hit zero and vanished.",
                "That's my rec urd so far. The timer hit zero and vanished.",
                "ref_1_to_hyp_2",
            ),
            (
                "That's my record so far.",
                "That's my re cord so far.",
                "ref_1_to_hyp_2",
            ),
        ]
        for reference, hypothesis, direction in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis, threshold=0.65)
                self.assertTrue(result["passed"])
                self.assertEqual(result["score"], 1.0)
                self.assertEqual(result["missing_tokens"], [])
                self.assertEqual(result["substitutions"], [])
                boundaries = [
                    item
                    for item in result["alignment_operations"]
                    if item["op"] == "boundary_resegmentation"
                ]
                self.assertEqual(len(boundaries), 1)
                self.assertEqual(boundaries[0]["direction"], direction)
                self.assertEqual(boundaries[0]["boundary_method"], "homograph_whitelist")

        one_token = compare_spoken("They read the note.", "They red the note.")
        self.assertTrue(one_token["passed"])
        self.assertTrue(any(
            item.get("kind") == "homograph_whitelist"
            for item in one_token["accepted_equivalences"]
        ))

    def test_homograph_whitelist_stays_exact_and_protects_numeric_content(self):
        """Reject altered whitelist surfaces and preserve numeric boundary guards."""
        altered = compare_spoken("That's my record so far.", "That's my rec art so far.")
        self.assertFalse(altered["passed"])
        self.assertIsNone(_boundary_resegmentation_mode(["estimate"], ["estim", "8"]))
        self.assertFalse(compare_spoken("I will not go.", "I will now go.")["passed"])

    def test_generic_boundary_resegmentation_excludes_protected_tokens(self):
        """Keep numeric slots, typed IDs, and strict acronyms out of generic joins."""
        cases = [
            (["eight", "rail"], ["eightrail"], None),
            (["<NUM>", "rail"], ["numrail"], None),
            (["<ID0>", "rail"], ["idrail"], None),
            (["dc", "line"], ["dcline"], {"dc"}),
        ]
        for reference, hypothesis, strict_tokens in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                self.assertIsNone(_boundary_resegmentation_mode(
                    reference,
                    hypothesis,
                    strict_ref_tokens=strict_tokens,
                ))

    def test_boundary_resegmentation_is_bidirectional_and_local(self):
        """Match exact compound boundaries independently of another ASR spelling change."""
        cases = [
            (
                "Nevermind that, Preslan.",
                "Never mind that, Preslin.",
                "ref_1_to_hyp_2",
            ),
            (
                "Never mind that, Preslan.",
                "Nevermind that, Preslin.",
                "ref_2_to_hyp_1",
            ),
            (
                "The highspeedrail passed Preslan.",
                "The high speed rail passed Preslin.",
                "ref_1_to_hyp_3",
            ),
            (
                "The high speed rail passed Preslan.",
                "The highspeedrail passed Preslin.",
                "ref_3_to_hyp_1",
            ),
            (
                "No where was safe.",
                "Nowhere was safe.",
                "ref_2_to_hyp_1",
            ),
            (
                "Not able to proceed.",
                "Notable to proceed.",
                "ref_2_to_hyp_1",
            ),
            (
                "Therapist arrived.",
                "The rapist arrived.",
                "ref_1_to_hyp_2",
            ),
        ]
        for reference, hypothesis, direction in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis, threshold=0.99)
                self.assertTrue(result["passed"])
                self.assertEqual(result["score"], 1.0)
                self.assertEqual(result["comparison_policy_version"], "boundary-resegmentation-v2")
                self.assertEqual(result["extra_tokens"], [])
                self.assertEqual(result["missing_tokens"], [])
                self.assertEqual(result["substitutions"], [])
                self.assertFalse(result["requires_second_stage_confirmation"])
                boundaries = [
                    item
                    for item in result["alignment_operations"]
                    if item["op"] == "boundary_resegmentation"
                ]
                self.assertEqual(len(boundaries), 1)
                self.assertEqual(boundaries[0]["direction"], direction)
                self.assertEqual(boundaries[0]["boundary_method"], "exact_surface")
                self.assertTrue(any(
                    item.get("kind") == "boundary_resegmentation"
                    and item.get("direction") == direction
                    for item in result["accepted_phrase_equivalences"]
                ))

    def test_boundary_resegmentation_rejects_missing_or_altered_content(self):
        """Reject negation loss, changed compact surfaces, and partial matches."""
        cases = [
            ("Never mind was fine.", "Mind was fine."),
            ("Not able to proceed.", "Able to proceed."),
            ("No one arrived.", "One arrived."),
            ("Black goop near Preslan.", "Black elephant near Preslin."),
            ("Black goop arrived.", "Blackcoop arrived."),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis, threshold=0.99)
                self.assertFalse(result["passed"])
                self.assertFalse(any(
                    item["op"] == "boundary_resegmentation"
                    for item in result["alignment_operations"]
                ))

    def test_named_boundary_compatibility_rules_and_albeit_remain_available(self):
        """Keep historical exact and Albeit compatibility rules explicit and bounded."""
        for reference, hypothesis in [
            ("A round object rolled.", "Around object rolled."),
            ("We walked in to town.", "We walked into town."),
        ]:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis, threshold=0.99)
                self.assertTrue(result["passed"])
                self.assertTrue(any(
                    item["op"] == "boundary_resegmentation"
                    and item.get("boundary_method") == "special_exact_surface"
                    for item in result["alignment_operations"]
                ))

        reduced_auxiliary = compare_spoken(
            "What d'you think?",
            "What do you think?",
            threshold=0.99,
        )
        self.assertTrue(reduced_auxiliary["passed"])
        self.assertTrue(any(
            item["op"] == "boundary_resegmentation"
            and item.get("boundary_method") == "guarded_reduced_auxiliary"
            for item in reduced_auxiliary["alignment_operations"]
        ))

        for reference, hypothesis in [
            ("Albeit, we continued.", "I'll be at, we continued."),
            ("I'll be at, we continued.", "Albeit, we continued."),
        ]:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis, threshold=0.99)
                self.assertTrue(result["passed"])
                self.assertTrue(any(
                    item.get("kind") == "raw_phrase_resegmentation"
                    and item.get("specialized_rule") == "albeit"
                    for item in result["accepted_phrase_equivalences"]
                ))

    def test_boundary_alignment_is_deterministic_and_truncation_aware(self):
        """Prefer nonoverlapping exact boundaries and restore their word coverage."""
        reference = "Never mind near high speed rail."
        hypothesis = "Nevermind near highspeedrail."
        results = [compare_spoken(reference, hypothesis, threshold=0.99) for _ in range(3)]
        expected_operations = results[0]["alignment_operations"]
        for result in results:
            self.assertTrue(result["passed"])
            self.assertEqual(result["alignment_operations"], expected_operations)
            boundaries = [
                item for item in result["alignment_operations"]
                if item["op"] == "boundary_resegmentation"
            ]
            self.assertEqual(len(boundaries), 2)
            self.assertEqual(result["truncation_warning"], "")

        reverse = compare_spoken(
            "High speed rail.",
            "Highspeedrail.",
            threshold=0.99,
        )
        self.assertTrue(reverse["passed"])
        self.assertEqual(reverse["truncation_warning"], "")

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

    def test_written_and_digit_numeric_structures_align(self):
        """Align written and digit forms without merging lists or ranges."""
        cases = [
            ("sixteen-year-old", "16-year-old"),
            ("two and three", "2 and 3"),
            ("thirty-two and thirty-six", "32 and 36"),
            ("one hundred and five", "105"),
            ("twenty to twelve hundred", "20 to 1200"),
        ]
        for reference, hypothesis in cases:
            with self.subTest(reference=reference, hypothesis=hypothesis):
                result = compare_spoken(reference, hypothesis, threshold=0.65)
                self.assertTrue(result["passed"])
                self.assertEqual(result["ref_normalized"], result["hyp_normalized"])

    def test_numeric_range_connector_change_stays_failed(self):
        """Reject a list when ASR changes its numeric relationship to a range."""
        result = compare_spoken(
            "twenty to thirty",
            "twenty and thirty",
            threshold=0.65,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_type"], "changed_numeric_structure")

    def test_spoken_code_and_time_numeric_variants_pass(self):
        """Accept compact code letters and guarded time two-to substitutions."""
        code = compare_spoken(
            "They were at point eight cee.",
            "They were at point 8C.",
            threshold=0.65,
        )
        self.assertTrue(code["passed"])
        self.assertEqual(code["ref_normalized"], code["hyp_normalized"])

        time_variant = compare_spoken(
            "zero two thirty hours.",
            "0 to 30 hours.",
            threshold=0.65,
        )
        self.assertTrue(time_variant["passed"])
        self.assertTrue(any(
            item.get("kind") == "time_two_to_surface_variation"
            for item in time_variant["accepted_phrase_equivalences"]
        ))

    def test_roman_acronym_context_does_not_force_numeric_conversion(self):
        """Keep bare uppercase acronyms distinct from explicitly framed numerals."""
        self.assertEqual(normalize_spoken("DV"), "dv")
        self.assertEqual(normalize_spoken("Chapter IV"), "chapter <ID0>")
        self.assertFalse(compare_spoken("DV", "505", threshold=0.65)["passed"])

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

    def test_multiple_raw_surface_variations_preserve_content_failures(self):
        """Accept unlimited anchored spelling surfaces without weakening content checks."""
        accepted = compare_spoken(
            "Next he used a sixpack of Hamm's and Mack Bol'un left with xaxkluth.",
            "Next he used a six-pack of hams and MacBolan left with X-Ax Cluth.",
            threshold=0.65,
        )
        self.assertTrue(accepted["passed"])
        surfaces = [
            item for item in accepted["accepted_phrase_equivalences"]
            if item.get("kind") in {
                "raw_compact_boundary_surface",
                "apostrophe_s_surface_variation",
            }
        ]
        self.assertGreaterEqual(len(surfaces), 3)

        terminal_diagnostic = compare_spoken(
            "Correct text here.",
            "Correct text here. [BLANK_AUDIO]",
            threshold=0.65,
        )
        self.assertTrue(terminal_diagnostic["passed"])

        self.assertFalse(compare_spoken(
            "A sixpack arrived.",
            "A five-pack arrived.",
            threshold=0.65,
        )["passed"])
        self.assertFalse(compare_spoken(
            "It wasn't safe.",
            "It was safe.",
            threshold=0.65,
        )["passed"])


if __name__ == "__main__":
    unittest.main()
