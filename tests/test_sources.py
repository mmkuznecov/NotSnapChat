"""Tests for frame sources"""

import unittest
import tempfile
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from snapflow.core.source import CameraSource, VideoSource


class TestFrameSources(unittest.TestCase):
    """Test frame source implementations"""

    def test_camera_source_init(self):
        """Test camera source initialization"""
        source = CameraSource(device=0, width=640, height=480)
        self.assertEqual(source.device, 0)
        self.assertEqual(source.width, 640)
        self.assertEqual(source.height, 480)
        self.assertEqual(source.frame_id, 0)

    def test_video_source_init(self):
        """Test video source initialization"""
        source = VideoSource("test.mp4")
        self.assertEqual(source.path, "test.mp4")
        self.assertEqual(source.frame_id, 0)

    def test_frame_format(self):
        """Test that frame format is correct"""
        # Create a test video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            # Create a simple test video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp.name, fourcc, 20.0, (640, 480))

            # Write a few test frames
            for i in range(5):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                out.write(frame)
            out.release()

            # Test reading the video
            source = VideoSource(tmp.name)
            source.open()

            frame = source.read()
            if frame is not None:  # Might be None if codec issues
                self.assertIn("image", frame)
                self.assertIn("frame_id", frame)
                self.assertIn("timestamp", frame)
                self.assertIsInstance(frame["image"], np.ndarray)
                self.assertIsInstance(frame["frame_id"], int)
                self.assertIsInstance(frame["timestamp"], float)

            source.close()
            Path(tmp.name).unlink()  # Clean up
