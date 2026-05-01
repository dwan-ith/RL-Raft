import os
import unittest

from rlraft.rl.llm_advisor import AdvisorGuidance, deterministic_guidance, get_default_guidance


class LlmAdvisorTests(unittest.TestCase):
    def test_deterministic_guidance_reacts_to_bad_metrics(self) -> None:
        guidance = deterministic_guidance({"split_vote_rate": 0.5, "best_node_win_rate": 0.1})
        self.assertGreater(guidance.split_penalty, AdvisorGuidance().split_penalty)
        self.assertGreater(guidance.best_leader_bonus, AdvisorGuidance().best_leader_bonus)

    def test_llm_default_falls_back_without_api_key(self) -> None:
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            guidance = get_default_guidance(advisor="llm")
            self.assertEqual(guidance.source, "deterministic_fallback")
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key


if __name__ == "__main__":
    unittest.main()
