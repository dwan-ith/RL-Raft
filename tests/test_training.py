import json
import tempfile
import unittest
from pathlib import Path

from rlraft.rl.training import train_policy


class TrainingTests(unittest.TestCase):
    def test_training_writes_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.json"
            result = train_policy(episodes=200, nodes=20, output_path=str(path), seed=3)
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["policy_type"], "tabular_marl_timeout_q_policy")
            self.assertEqual(data["algorithm"], "independent-q-learning-with-parameter-sharing")
            self.assertIn("q_values", data)
            self.assertIn("greedy_policy", data)
            self.assertIn("evaluation_metrics", data)
            self.assertGreaterEqual(result.success_rate, 0.0)
            self.assertLessEqual(result.split_vote_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
