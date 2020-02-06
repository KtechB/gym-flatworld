import unittest
import gym_flatworld.envs.flatworld as flatworld

class TestFlatworld(unittest.TestCase):
    def test_flatworld(self):
        done = flatworld.test_run()
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main(verbosity = 2)