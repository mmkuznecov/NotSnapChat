"""Tests for game registration and creation"""

import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import to register games
import snapflow.games
from snapflow.core.registry import get_game, GAMES, register_game
from snapflow.core.store import Store


class TestGameRegistry(unittest.TestCase):
    """Test game registration system"""

    def test_games_registered(self):
        """Test that games are properly registered"""
        expected_games = [
            "blink_counter",
            "blink_counter_hard",
            "eye_enlarger",
            "eye_enlarger_extreme",
            "cannon_game",
            "cannon_game_easy",
            "cannon_game_hard",
            "ghost_faces",
            "ghost_faces_dense",
            "ghost_faces_sparse",
            "ghost_faces_outline",
        ]

        for game in expected_games:
            self.assertIn(game, GAMES, f"Game '{game}' not registered")

    def test_get_game(self):
        """Test getting game components"""
        processors, store = get_game("blink_counter")

        self.assertIsInstance(processors, list)
        self.assertGreater(len(processors), 0)
        self.assertIsInstance(store, Store)

    def test_invalid_game(self):
        """Test error handling for invalid game name"""
        with self.assertRaises(ValueError):
            get_game("nonexistent_game")

    def test_custom_game_registration(self):
        """Test registering a custom game"""
        from snapflow.core.processor import Processor
        from snapflow.core.store import NoOpStore

        class TestProcessor(Processor):
            def process(self, frame, state):
                return {"image": frame["image"]}

        @register_game("test_game")
        def make_test_game():
            return [TestProcessor()], NoOpStore()

        # Check it was registered
        self.assertIn("test_game", GAMES)

        # Check we can create it
        processors, store = get_game("test_game")
        self.assertEqual(len(processors), 1)
        self.assertIsInstance(processors[0], TestProcessor)
        self.assertIsInstance(store, NoOpStore)


if __name__ == "__main__":
    # Run all tests
    unittest.main()
