import unittest
from app.orchestration.retrieval import _semantic_call_key

class SemanticKeyTests(unittest.TestCase):
    def test_rewordings_of_one_search_collapse_to_one_key(self) -> None:
        keys = {
            _semantic_call_key("rag_vector_query", {"query": q})
            for q in (
                "warfarin ibuprofen interaction",
                "ibuprofen interaction with warfarin",
                "Warfarin and ibuprofen interaction!",
            )
        }
        self.assertEqual(len(keys), 1)

    def test_a_genuinely_different_search_keeps_its_own_key(self) -> None:
        a = _semantic_call_key("rag_vector_query", {"query": "warfarin ibuprofen interaction"})
        b = _semantic_call_key("rag_vector_query", {"query": "warfarin dosing in renal impairment"})
        self.assertNotEqual(a, b)

    def test_the_same_words_to_a_different_tool_are_a_different_call(self) -> None:
        a = _semantic_call_key("rag_vector_query", {"query": "metformin lactic acidosis"})
        b = _semantic_call_key("adr_retrieve_drug_info", {"query": "metformin lactic acidosis"})
        self.assertNotEqual(a, b)

    def test_korean_rewordings_collapse_too(self) -> None:
        keys = {
            _semantic_call_key("hira_updates_search", {"query": q})
            for q in ("위암 트라스투주맙 급여기준", "트라스투주맙의 위암 급여기준은?")
        }
        self.assertEqual(len(keys), 1)

if __name__ == "__main__":
    unittest.main()
