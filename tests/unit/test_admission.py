"""Evidence acquisition is decided before the generation request exists."""
from __future__ import annotations

import unittest

from app.admission import admit
from app.config import Settings
from app.contracts import ChatCompletionRequest
from app.orchestration.driver import ConversationDriver
from tests.fakes import NeverRetrieval, ScriptedL2, l2_content


STATE_DEPENDENT = (
    "와파린과 이부프로펜을 같이 먹어도 되나요",
    "이 약이 한국에서 이 적응증으로 허가되어 있나요",
    "이 항암제 급여 인정기준이 어떻게 되나요",
    "당뇨병 상병코드가 뭐죠",
    "의료법상 진료기록 보관 기간은",
    "can I take ibuprofen with warfarin",
    "what does the current guideline recommend as the blood pressure target",
    "is there a recent trial on this",
)

FROM_MEMORY = (
    "고혈압은 왜 생기나요",
    "why does hypertension develop",
    "my 6-year-old has a fever of 39C, what can I give her",
    "how painful is a colonoscopy",
    "what are the symptoms of postpartum depression",
)


class AdmissionTest(unittest.TestCase):
    def test_questions_that_need_an_external_record_are_admitted(self) -> None:
        for query in STATE_DEPENDENT:
            with self.subTest(query=query):
                self.assertTrue(admit(query).admitted, query)
                self.assertEqual(admit(query).lane, "evidence")

    def test_questions_the_model_can_answer_from_memory_are_not(self) -> None:
        for query in FROM_MEMORY:
            with self.subTest(query=query):
                self.assertFalse(admit(query).admitted, query)
                self.assertEqual(admit(query).lane, "memory")

    def test_empty_input_is_not_admitted(self) -> None:
        self.assertFalse(admit("").admitted)
        self.assertFalse(admit("   ").admitted)


class MemoryLaneIsIndistinguishableTest(unittest.TestCase):
    """The whole point: the model must not be able to tell a gate was applied."""

    def _run(self, question: str) -> ScriptedL2:
        l2 = ScriptedL2([l2_content("answer")])
        driver = ConversationDriver(
            l2=l2,
            retrieval=NeverRetrieval(),
            settings=Settings(lunit_fm_api_key="test", enable_high_risk_review=False),
        )
        driver.complete(
            ChatCompletionRequest(
                model="Lunit/L2-preview",
                messages=[{"role": "user", "content": question}],
            ),
            request_id="req-test",
        )
        return l2

    def test_the_retrieval_tool_is_never_offered_on_the_memory_lane(self) -> None:
        l2 = self._run("why does hypertension develop")
        self.assertTrue(l2.calls, "the driver made no L2 call")
        for call in l2.calls:
            self.assertEqual(call["tools"], [], "a retrieval tool was offered")

    def test_nothing_in_the_prompt_records_that_a_search_was_weighed(self) -> None:
        # An answer written after the model learns a search was ruled out is a smaller
        # answer, which is exactly the regression this gate exists to avoid.
        l2 = self._run("why does hypertension develop")
        text = " ".join(
            str(message.get("content", ""))
            for call in l2.calls
            for message in call["messages"]
        ).lower()
        # "evidence" survives in a sentence about contested evidence, and the instruction
        # not to leak tool schemas is a standing safety rule — neither implies a tool is
        # available for this turn. What must not appear is any sign of retrieval itself.
        for trace in ("retrieve_relevant_content", "retriev", "a tool named", "disabled"):
            self.assertNotIn(trace, text, f"prompt leaked {trace!r}")

    def test_an_admitted_question_still_gets_the_tool(self) -> None:
        l2 = self._run("can I take ibuprofen with warfarin")
        offered = [call["tools"] for call in l2.calls if call["tools"]]
        self.assertTrue(offered, "the evidence lane was not given the retrieval tool")


if __name__ == "__main__":
    unittest.main()
