"""SnapFlow - Real-time Vision Pipeline for Effects and Games"""

__version__ = "0.1.0"
__author__ = "SnapFlow Team"

# src/snapflow/core/__init__.py
"""Core framework components for SnapFlow"""

from .source import FrameSource, CameraSource, VideoSource
from .processor import Processor
from .store import Store, NoOpStore, SimpleStore
from .registry import register_game, get_game, GAMES

__all__ = [
    "FrameSource",
    "CameraSource",
    "VideoSource",
    "Processor",
    "Store",
    "NoOpStore",
    "SimpleStore",
    "register_game",
    "get_game",
    "GAMES",
]
