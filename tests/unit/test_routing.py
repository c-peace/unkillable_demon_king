from __future__ import annotations

import unittest

from app.evidence.routing import ROUTE_LABELS, ROUTING_RULES, SourceRouter


class RouteLabelTests(unittest.TestCase):
    def test_every_rule_has_a_label(self) -> None:
        self.assertEqual(len(ROUTE_LABELS), len(ROUTING_RULES))


class PatientPhrasingTests(unittest.TestCase):
    """Patients ask whether two medicines can be taken together, not about 상호작용."""

    def setUp(self) -> None:
        self.router = SourceRouter("family")

    def test_korean_asking_whether_two_medicines_can_be_taken_together(self) -> None:
        for query in (
            "와파린이랑 이부프로펜 같이 먹어도 되나요",
            "혈압약이랑 감기약 함께 복용해도 괜찮을까요",
            "이 두 개를 동시에 먹어도 안전한가요",
        ):
            with self.subTest(query=query):
                self.assertIn("drug_safety", self.router.routes(query))

    def test_korean_asking_whether_a_medicine_is_safe_at_all(self) -> None:
        for query in (
            "임신 중인데 이 약 먹어도 안전한가요",
            "이 약 먹으면 안 되는 사람이 있나요",
            "이거 복용해도 되나요",
        ):
            with self.subTest(query=query):
                self.assertIn("drug_safety", self.router.routes(query))

    def test_english_phrasings_route_the_same_way(self) -> None:
        for query in (
            "Is it safe to take ibuprofen with warfarin?",
            "Can I take these together?",
        ):
            with self.subTest(query=query):
                self.assertIn("drug_safety", self.router.routes(query))

    def test_other_domains_are_unaffected(self) -> None:
        cases = {
            "CKD 환자의 목표 혈압 가이드라인": "guideline",
            "트라스투주맙 위암 급여기준": "hira",
            "제2형 당뇨병 KCD 코드": "coding",
            "의료법상 진료기록부 보존연한": "law",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertIn(expected, self.router.routes(query))


if __name__ == "__main__":
    unittest.main()
