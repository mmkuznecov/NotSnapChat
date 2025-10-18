"""Storage classes for events and metrics"""

import time
from abc import ABC, abstractmethod


class Store(ABC):
    """Base class for data storage"""

    def open(self):
        """Initialize storage (optional)"""
        pass

    @abstractmethod
    def record(self, result):
        """Record processing result"""
        pass

    @abstractmethod
    def get_data(self):
        """Get stored data"""
        pass

    def close(self):
        """Clean up storage resources (optional)"""
        pass


class NoOpStore(Store):
    """No-operation storage - for effects that don't need metrics"""

    def record(self, result):
        """Do nothing"""
        pass

    def get_data(self):
        """Return empty data"""
        return {}


class SimpleStore(Store):
    """In-memory storage for events and metrics"""

    def __init__(self):
        self.events = []
        self.metrics = []

    def record(self, result):
        """Record events and metrics from result"""
        if "events" in result:
            self.events.extend(result["events"])
        if "metrics" in result:
            for name, value in result["metrics"].items():
                self.metrics.append({"name": name, "value": value, "time": time.time()})

    def get_data(self):
        """Get all stored events and metrics"""
        return {"events": self.events, "metrics": self.metrics}
