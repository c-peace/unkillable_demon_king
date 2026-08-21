from __future__ import annotations

import unittest

from app.coverage import extract_contract


class ContractExtractionTests(unittest.TestCase):
    def test_a_single_question_carries_no_contract(self) -> None:
        for text in ("Metformin이 뭐예요?", "What is metformin?", "이 약 부작용이 뭔가요"):
            with self.subTest(text=text):
                self.assertFalse(extract_contract(text).is_multipart)

    def test_two_korean_asks_in_one_message_are_both_kept(self) -> None:
        contract = extract_contract(
            "아이가 3일째 설사를 해요. 병원에 가야 할까요? 그리고 집에서 뭘 해줄 수 있나요?"
        )
        self.assertTrue(contract.is_multipart)
        self.assertEqual(len(contract.requirements), 2)

    def test_an_english_sentence_joined_by_and_splits_on_the_question_word(self) -> None:
        contract = extract_contract(
            "What painkillers are safe, and how would I know if he is bleeding?"
        )
        self.assertEqual(len(contract.requirements), 2)

    def test_an_ordinary_and_inside_a_phrase_does_not_split(self) -> None:
        contract = extract_contract("Mix salt and water for the rinse — is that right?")
        self.assertFalse(contract.is_multipart)

    def test_nothing_is_invented_on_the_user_s_behalf(self) -> None:
        """No emergency, referral, or monitoring item appears unless the user asked."""
        contract = extract_contract(
            "와파린 먹는데 이부프로펜 먹어도 되나요? 위험한 이유도 알려주세요"
        )
        joined = " ".join(contract.requirements)
        for invented in ("응급", "119", "진료를 받으세요", "monitoring", "emergency"):
            self.assertNotIn(invented, joined)

    def test_the_prompt_lists_every_requirement(self) -> None:
        contract = extract_contract(
            "이 약 언제 먹나요? 그리고 부작용은 어떤 게 있나요?"
        )
        prompt = contract.as_prompt()
        for requirement in contract.requirements:
            self.assertIn(requirement, prompt)

    def test_empty_input_is_handled(self) -> None:
        for text in ("", "   ", "\n"):
            with self.subTest(text=repr(text)):
                self.assertEqual(extract_contract(text).requirements, ())

    def test_the_list_is_capped_so_a_long_message_cannot_flood_the_prompt(self) -> None:
        text = " ".join(f"{i}번째로 이건 어떻게 하나요?" for i in range(20))
        self.assertLessEqual(len(extract_contract(text).requirements), 6)


if __name__ == "__main__":
    unittest.main()
