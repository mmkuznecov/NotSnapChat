"""Tests for processors"""

import unittest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snapflow.core.processor import Processor
from snapflow.games.blink_counter.processor import BlinkGameProcessor
from snapflow.games.eye_enlarger.processor import EyeEnlargerProcessor


class MockProcessor(Processor):
    """Mock processor for testing"""

    def process(self, frame, state):
        return {
            "image": frame["image"],
            "events": ["test_event"],
            "metrics": {"test_metric": 1.0},
            "state": {"test_state": True},
        }


class TestProcessors(unittest.TestCase):
    """Test processor implementations"""

    def setUp(self):
        """Set up test data"""
        self.test_frame = {
            "image": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "frame_id": 1,
            "timestamp": 1234567890.0,
        }
        self.test_state = {}

    def test_mock_processor(self):
        """Test mock processor"""
        processor = MockProcessor()
        result = processor.process(self.test_frame, self.test_state)

        self.assertIn("image", result)
        self.assertIn("events", result)
        self.assertIn("metrics", result)
        self.assertIn("state", result)
        self.assertEqual(result["events"], ["test_event"])
        self.assertEqual(result["metrics"]["test_metric"], 1.0)
        self.assertTrue(result["state"]["test_state"])

    def test_blink_processor_init(self):
        """Test blink processor initialization"""
        processor = BlinkGameProcessor(ear_threshold=0.25, consecutive_frames=3)
        self.assertEqual(processor.ear_threshold, 0.25)
        self.assertEqual(processor.consecutive_frames, 3)
        self.assertEqual(processor.total_blinks, 0)

    def test_eye_enlarger_init(self):
        """Test eye enlarger initialization"""
        processor = EyeEnlargerProcessor(scale_factor=1.5)
        self.assertEqual(processor.scale_factor, 1.5)

    def test_processor_lifecycle(self):
        """Test processor open/close lifecycle"""
        processor = MockProcessor()

        # Should not raise exceptions
        processor.open()
        result = processor.process(self.test_frame, self.test_state)
        processor.close()

        self.assertIsInstance(result, dict)
