"""Frame sources for capturing video input"""

import cv2
import time
from abc import ABC, abstractmethod


class FrameSource(ABC):
    """Base class for frame sources"""

    @abstractmethod
    def open(self):
        """Initialize the frame source"""
        pass

    @abstractmethod
    def read(self):
        """Read next frame. Returns frame dict or None for end-of-stream"""
        pass

    @abstractmethod
    def close(self):
        """Clean up resources"""
        pass


class CameraSource(FrameSource):
    """Camera frame source"""

    def __init__(self, device=0, width=640, height=480):
        self.device = device
        self.width = width
        self.height = height
        self.cap = None
        self.frame_id = 0

    def open(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        """Read frame from camera"""
        if not self.cap or not self.cap.isOpened():
            return None
        ret, img = self.cap.read()
        if not ret:
            return None
        self.frame_id += 1
        return {"image": img, "frame_id": self.frame_id, "timestamp": time.time()}

    def close(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()


class VideoSource(FrameSource):
    """Video file frame source"""

    def __init__(self, path):
        self.path = path
        self.cap = None
        self.frame_id = 0

    def open(self):
        """Initialize video file capture"""
        self.cap = cv2.VideoCapture(self.path)

    def read(self):
        """Read frame from video file"""
        if not self.cap or not self.cap.isOpened():
            return None
        ret, img = self.cap.read()
        if not ret:
            return None
        self.frame_id += 1
        return {"image": img, "frame_id": self.frame_id, "timestamp": time.time()}

    def close(self):
        """Release video file resources"""
        if self.cap:
            self.cap.release()
