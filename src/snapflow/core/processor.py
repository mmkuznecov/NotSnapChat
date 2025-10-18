"""Base processor class for effects and games"""

from abc import ABC, abstractmethod


class Processor(ABC):
    """Base class for frame processors"""

    def open(self):
        """Initialize processor resources (optional)"""
        pass

    @abstractmethod
    def process(self, frame, state):
        """
        Process a frame and return result

        Args:
            frame: Frame dict with keys 'image', 'frame_id', 'timestamp'
            state: Shared state dict between processors

        Returns:
            Result dict with optional keys:
            - 'image': Processed image (or None if headless)
            - 'events': List of event names
            - 'metrics': Dict of measurements
            - 'state': State updates to pass to next processor
        """
        pass

    def close(self):
        """Clean up processor resources (optional)"""
        pass
