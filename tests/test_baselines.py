import unittest
from guandan.agents import GreedyAgent, RandomAgent, TeamHeuristicAgent
from guandan.types import Action, PlayerObservation, PASS


class BaselineTests(unittest.TestCase):
    def obs(self, last_player=2):
        return PlayerObservation(0, (1, 2, 3), 2, 0, (3, 7, 5, 7), (),
                                 Action("single", (0,), 2), last_player, (), (), 1, False, None)

    def test_random_is_reproducible_and_legal(self):
        actions = [PASS, Action("single", (1,), 3)]
        a, b = RandomAgent(123), RandomAgent(123)
        seq_a = [a.act(self.obs(), actions) for _ in range(50)]
        self.assertEqual(seq_a, [b.act(self.obs(), actions) for _ in range(50)])
        self.assertTrue(all(x in actions for x in seq_a))

    def test_teammate_preference_is_soft(self):
        agent = TeamHeuristicAgent()
        single = Action("single", (1,), 3)
        self.assertEqual(agent.act(self.obs(), [PASS, single]), PASS)
        finish = Action("triple", (1, 2, 3), 3)
        self.assertEqual(agent.act(self.obs(), [PASS, finish]), finish)

    def test_greedy_finishes(self):
        finish = Action("triple", (1, 2, 3), 3)
        self.assertEqual(GreedyAgent().act(self.obs(), [PASS, finish]), finish)


if __name__ == "__main__":
    unittest.main()
