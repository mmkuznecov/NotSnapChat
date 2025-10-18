# Adding a New Game to SnapFlow

This guide walks you through creating a new game or effect for the SnapFlow vision pipeline framework.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Step-by-Step Game Creation](#step-by-step-game-creation)
4. [Complete Example: Mouth Open Detector](#complete-example-mouth-open-detector)
5. [Testing Your Game](#testing-your-game)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)

---

## Quick Start

**Minimum steps to add a new game:**

1. Create a new directory under `src/snapflow/games/your_game_name/`
2. Add `__init__.py` and `processor.py` files
3. Implement a `Processor` subclass
4. Register your game with `@register_game("your_game_name")`
5. Import your game in `src/snapflow/games/__init__.py`
6. Run with `python run_game.py your_game_name`

---

## Understanding the Architecture

### The Four Core Components

Every SnapFlow game consists of four components:

```
┌─────────────────┐
│  Frame Source   │  Captures video input (camera/file)
└────────┬────────┘
         │ frame dict
         ▼
┌─────────────────┐
│   Processor(s)  │  Your game logic lives here
└────────┬────────┘
         │ result dict
         ▼
┌─────────────────┐
│     Store       │  Collects events and metrics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Main Loop     │  Orchestrates everything
└─────────────────┘
```

### What You Need to Implement

**Minimum (for effects):**
- A `Processor` class with `process(frame, state)` method
- A factory function decorated with `@register_game()`

**Optional (for games with scoring):**
- Custom `Store` class (or use `SimpleStore`)
- Additional `open()` and `close()` methods for resource management

---

## Step-by-Step Game Creation

### Step 1: Create Game Directory Structure

```bash
cd src/snapflow/games/
mkdir your_game_name
cd your_game_name
touch __init__.py processor.py
```

Your structure should look like:
```
src/snapflow/games/
├── your_game_name/
│   ├── __init__.py
│   └── processor.py
```

### Step 2: Define Your Processor Class

In `processor.py`:

```python
"""Your game description"""

import cv2
import numpy as np
import mediapipe as mp
from ...core.processor import Processor
from ...core.registry import register_game
from ...core.store import SimpleStore, NoOpStore


class YourGameProcessor(Processor):
    """Your game processor that does something cool"""
    
    def __init__(self, param1=default_value):
        """
        Initialize your game
        
        Args:
            param1: Description of parameter
        """
        self.param1 = param1
        # Initialize any game state variables here
        self.score = 0
        
        # MediaPipe (if you need face detection)
        self.mp_face = None
    
    def open(self):
        """Initialize resources (called once at startup)"""
        # Initialize MediaPipe if needed
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    
    def process(self, frame, state):
        """
        Process a single frame
        
        Args:
            frame: Dict with keys:
                - 'image': numpy array (BGR format)
                - 'frame_id': int
                - 'timestamp': float
            state: Dict shared between processors
        
        Returns:
            Dict with keys:
                - 'image': processed image (or original if no changes)
                - 'events': list of event names (optional)
                - 'metrics': dict of measurements (optional)
                - 'state': dict of state updates (optional)
        """
        img = frame["image"]
        h, w = img.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)
        
        events = []
        metrics = {}
        img_out = img.copy()
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # YOUR GAME LOGIC HERE
            # Example: detect something, update score, draw visuals
            
            # Draw something on the image
            cv2.putText(
                img_out,
                f"Score: {self.score}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
            )
        
        return {
            "image": img_out,
            "events": events,
            "metrics": metrics,
            "state": {"score": self.score},
        }
    
    def close(self):
        """Clean up resources (called at shutdown)"""
        if self.mp_face:
            self.mp_face.close()


# Register your game
@register_game("your_game_name")
def make_your_game():
    """Factory function to create your game components"""
    processors = [YourGameProcessor()]
    store = SimpleStore()  # or NoOpStore() for effects
    return processors, store
```

### Step 3: Export Your Processor

In `__init__.py`:

```python
"""Your game name"""

from .processor import YourGameProcessor

__all__ = ["YourGameProcessor"]
```

### Step 4: Register in Main Games Module

Add your import to `src/snapflow/games/__init__.py`:

```python
"""SnapFlow games and effects"""

# Import game modules to register them
from . import blink_counter
from . import eye_enlarger
from . import cannon_game
from . import ghost_faces
from . import your_game_name  # ADD THIS LINE
```

### Step 5: Test Your Game

```bash
# Run your game
python run_game.py your_game_name

# Verify it appears in the list
python run_game.py --list-games
```

---

## Complete Example: Mouth Open Detector

Let's create a complete game that detects when your mouth is open and counts how long you keep it open.

### File: `src/snapflow/games/mouth_open_detector/__init__.py`

```python
"""Mouth Open Detector Game"""

from .processor import MouthOpenDetectorProcessor

__all__ = ["MouthOpenDetectorProcessor"]
```

### File: `src/snapflow/games/mouth_open_detector/processor.py`

```python
"""Mouth open detector game processor"""

import cv2
import numpy as np
import mediapipe as mp
import math
from ...core.processor import Processor
from ...core.registry import register_game
from ...core.store import SimpleStore


class MouthOpenDetectorProcessor(Processor):
    """Processor that detects and scores mouth opening"""
    
    # MediaPipe landmark indices for mouth
    UPPER_LIP = 13   # Top of upper lip
    LOWER_LIP = 14   # Bottom of lower lip
    MOUTH_LEFT = 78  # Left corner
    MOUTH_RIGHT = 308  # Right corner
    
    def __init__(self, open_threshold=0.15, consecutive_frames=3):
        """
        Initialize mouth open detector
        
        Args:
            open_threshold: Mouth aspect ratio threshold for "open" detection
            consecutive_frames: Frames needed to confirm mouth is open
        """
        self.open_threshold = open_threshold
        self.consecutive_frames = consecutive_frames
        
        # Game state
        self.total_open_time = 0.0  # Total seconds mouth was open
        self.current_open_frames = 0
        self.is_currently_open = False
        self.longest_streak = 0
        self.current_streak = 0
        
        # MediaPipe
        self.mp_face = None
    
    def open(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    
    def _calculate_mouth_aspect_ratio(self, landmarks, w, h):
        """
        Calculate Mouth Aspect Ratio (MAR)
        
        Returns:
            float: MAR value (higher = more open)
        """
        def get_point(idx):
            return np.array([
                landmarks[idx].x * w,
                landmarks[idx].y * h
            ])
        
        # Get mouth landmark points
        upper_lip = get_point(self.UPPER_LIP)
        lower_lip = get_point(self.LOWER_LIP)
        left_corner = get_point(self.MOUTH_LEFT)
        right_corner = get_point(self.MOUTH_RIGHT)
        
        # Calculate vertical and horizontal distances
        vertical = np.linalg.norm(upper_lip - lower_lip)
        horizontal = np.linalg.norm(left_corner - right_corner)
        
        if horizontal == 0:
            return 0.0
        
        # MAR = vertical / horizontal
        mar = vertical / horizontal
        return mar
    
    def process(self, frame, state):
        """Process frame to detect mouth opening"""
        if not self.mp_face:
            return {"image": frame["image"]}
        
        img = frame["image"]
        h, w = img.shape[:2]
        
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.mp_face.process(rgb)
        
        events = []
        metrics = {}
        img_out = img.copy()
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate mouth aspect ratio
            mar = self._calculate_mouth_aspect_ratio(landmarks, w, h)
            metrics["mouth_aspect_ratio"] = mar
            
            # Detect mouth opening
            if mar > self.open_threshold:
                self.current_open_frames += 1
                
                # Confirm mouth is open
                if self.current_open_frames >= self.consecutive_frames:
                    if not self.is_currently_open:
                        events.append("mouth_opened")
                        self.is_currently_open = True
                    
                    # Track open time (assuming 30 fps)
                    self.total_open_time += 1/30.0
                    self.current_streak += 1
                    
                    if self.current_streak > self.longest_streak:
                        self.longest_streak = self.current_streak
            else:
                # Mouth closed
                if self.is_currently_open:
                    events.append("mouth_closed")
                    self.is_currently_open = False
                    self.current_streak = 0
                
                self.current_open_frames = 0
            
            # Draw mouth landmarks
            def draw_point(idx, color):
                pt = landmarks[idx]
                x, y = int(pt.x * w), int(pt.y * h)
                cv2.circle(img_out, (x, y), 3, color, -1)
            
            color = (0, 255, 0) if self.is_currently_open else (0, 0, 255)
            draw_point(self.UPPER_LIP, color)
            draw_point(self.LOWER_LIP, color)
            draw_point(self.MOUTH_LEFT, color)
            draw_point(self.MOUTH_RIGHT, color)
        
        # Draw UI
        self._draw_ui(img_out, metrics.get("mouth_aspect_ratio", 0))
        
        return {
            "image": img_out,
            "events": events,
            "metrics": metrics,
            "state": {
                "total_open_time": self.total_open_time,
                "longest_streak": self.longest_streak,
                "is_open": self.is_currently_open,
            },
        }
    
    def _draw_ui(self, image, mar):
        """Draw game UI"""
        # Title
        cv2.putText(
            image,
            "Mouth Open Challenge",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        
        # Total open time
        cv2.putText(
            image,
            f"Total Time: {self.total_open_time:.1f}s",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        
        # Longest streak
        cv2.putText(
            image,
            f"Longest: {self.longest_streak} frames",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 200, 0),
            2,
        )
        
        # Current status
        status = "OPEN" if self.is_currently_open else "CLOSED"
        color = (0, 255, 0) if self.is_currently_open else (0, 0, 255)
        cv2.putText(
            image,
            f"Mouth: {status}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        
        # MAR value
        cv2.putText(
            image,
            f"MAR: {mar:.3f}",
            (10, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        
        # Instructions
        cv2.putText(
            image,
            "Keep your mouth open as long as possible!",
            (10, image.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    
    def close(self):
        """Clean up MediaPipe resources"""
        if self.mp_face:
            self.mp_face.close()


# Register the game with multiple variants
@register_game("mouth_open_detector")
def make_mouth_open_game():
    """Standard difficulty mouth open detector"""
    processors = [MouthOpenDetectorProcessor(open_threshold=0.15)]
    store = SimpleStore()
    return processors, store


@register_game("mouth_open_detector_hard")
def make_mouth_open_game_hard():
    """Harder variant - requires wider mouth opening"""
    processors = [MouthOpenDetectorProcessor(open_threshold=0.25)]
    store = SimpleStore()
    return processors, store


@register_game("mouth_open_detector_easy")
def make_mouth_open_game_easy():
    """Easier variant - detects smaller mouth openings"""
    processors = [MouthOpenDetectorProcessor(open_threshold=0.10)]
    store = SimpleStore()
    return processors, store
```

### Add Import

In `src/snapflow/games/__init__.py`:

```python
"""SnapFlow games and effects"""

from . import blink_counter
from . import eye_enlarger
from . import cannon_game
from . import ghost_faces
from . import mouth_open_detector  # ADD THIS
```

### Test It

```bash
python run_game.py mouth_open_detector
python run_game.py mouth_open_detector_easy
python run_game.py mouth_open_detector_hard
```

---

## Testing Your Game

### Manual Testing Checklist

- [ ] Game appears in `--list-games`
- [ ] Runs without errors
- [ ] Face detection works reliably
- [ ] Game logic responds to inputs correctly
- [ ] UI elements display properly
- [ ] Performance is smooth (30+ fps)
- [ ] Clean exit with 'q' or ESC

### Automated Testing

Create `tests/test_your_game.py`:

```python
"""Tests for your game"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import snapflow.games.your_game_name
from snapflow.core.registry import get_game, GAMES


class TestYourGame(unittest.TestCase):
    """Test your game registration and creation"""
    
    def test_game_registered(self):
        """Test that game is properly registered"""
        self.assertIn("your_game_name", GAMES)
    
    def test_get_game(self):
        """Test getting game components"""
        processors, store = get_game("your_game_name")
        
        self.assertIsInstance(processors, list)
        self.assertGreater(len(processors), 0)
        self.assertIsNotNone(store)
    
    def test_processor_init(self):
        """Test processor initialization"""
        from snapflow.games.your_game_name.processor import YourGameProcessor
        
        processor = YourGameProcessor()
        # Test initialization values
        self.assertEqual(processor.score, 0)


if __name__ == "__main__":
    unittest.main()
```

Run tests:
```bash
python -m pytest tests/test_your_game.py
# or
python tests/test_your_game.py
```

---

## Advanced Features

### 1. Multiple Processors Chain

Combine multiple processors for complex effects:

```python
@register_game("combined_effect")
def make_combined_effect():
    """Chain multiple processors together"""
    processors = [
        EyeEnlargerProcessor(scale_factor=1.5),
        BlinkGameProcessor(),
        # Output from first processor becomes input to second
    ]
    store = SimpleStore()
    return processors, store
```

### 2. Custom Store for Data Persistence

Save game data to files:

```python
from ...core.store import Store
import json

class FileStore(Store):
    """Store that saves data to JSON file"""
    
    def __init__(self, filepath="game_data.json"):
        self.filepath = filepath
        self.data = {"events": [], "metrics": []}
    
    def record(self, result):
        """Record events and metrics"""
        if "events" in result:
            self.data["events"].extend(result["events"])
        if "metrics" in result:
            self.data["metrics"].append(result["metrics"])
    
    def get_data(self):
        """Return recorded data"""
        return self.data
    
    def close(self):
        """Save data to file on close"""
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=2)


@register_game("your_game_with_save")
def make_game_with_save():
    processors = [YourGameProcessor()]
    store = FileStore("your_game_scores.json")
    return processors, store
```

### 3. Using MediaPipe Utilities

SnapFlow provides helper functions in `utils/`:

```python
from ...utils.mediapipe_helpers import (
    extract_eye_landmarks,
    calculate_ear,
    extract_eye_anchors,
)
from ...utils.cv_helpers import (
    bgr_to_rgb,
    rgb_to_bgr,
    draw_landmarks,
    create_eye_mask,
)

# In your processor:
left_eye, right_eye = extract_eye_landmarks(landmarks, w, h)
left_ear = calculate_ear(left_eye)
```

### 4. Game State Sharing

Share state between processors:

```python
def process(self, frame, state):
    # Read state from previous processor
    previous_score = state.get("score", 0)
    
    # Your processing logic
    new_score = previous_score + 10
    
    return {
        "image": processed_img,
        "state": {"score": new_score},  # Pass to next processor
    }
```

### 5. Performance Optimization

```python
def __init__(self):
    # Cache expensive computations
    self._cached_data = None
    self._last_cache_frame = -1

def process(self, frame, state):
    # Only recompute if needed
    if frame["frame_id"] != self._last_cache_frame:
        self._cached_data = expensive_computation()
        self._last_cache_frame = frame["frame_id"]
    
    # Use cached data
    result = use_cached_data(self._cached_data)
```

---

## Best Practices

### 1. Code Organization

```
✅ DO:
- One game per directory
- Keep processor logic in processor.py
- Use descriptive variable names
- Add docstrings to all classes and methods

❌ DON'T:
- Put multiple unrelated games in one file
- Mix game logic with framework code
- Use cryptic variable names like 'x1, x2, tmp'
```

### 2. MediaPipe Face Landmarks

Common landmark indices you'll use:

```python
# Eyes
LEFT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, ...]
RIGHT_EYE_LANDMARKS = [362, 398, 384, 385, 386, 387, 388, 466, 263, ...]

# Eye corners for EAR
LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Mouth
MOUTH_UPPER = 13
MOUTH_LOWER = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# Nose
NOSE_TIP = 1

# Face outline
FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, ...]
```

Full landmark map: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

### 3. UI Design Guidelines

```python
# Clear hierarchy
cv2.putText(img, "Main Title", (10, 30), 
           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

cv2.putText(img, "Score: 100", (10, 70),
           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

cv2.putText(img, "Instructions here", (10, 110),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

# Use color meaningfully
GREEN = (0, 255, 0)    # Success, active
RED = (0, 0, 255)      # Danger, inactive
BLUE = (255, 0, 0)     # Info
YELLOW = (0, 255, 255) # Warning
WHITE = (255, 255, 255)# Neutral
```

### 4. Error Handling

```python
def process(self, frame, state):
    try:
        # Your processing
        if not self.mp_face:
            return {"image": frame["image"]}
        
        # Safe landmark access
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Validate indices
            if landmark_idx < len(landmarks):
                point = landmarks[landmark_idx]
        
    except Exception as e:
        print(f"Error in processing: {e}")
        return {"image": frame["image"]}  # Return original on error
```

### 5. Configuration

Make games configurable:

```python
@register_game("your_game_easy")
def make_easy():
    return [YourGameProcessor(difficulty=0.5)], SimpleStore()

@register_game("your_game_normal")
def make_normal():
    return [YourGameProcessor(difficulty=1.0)], SimpleStore()

@register_game("your_game_hard")
def make_hard():
    return [YourGameProcessor(difficulty=2.0)], SimpleStore()
```

---

## Troubleshooting

### Common Issues

**Issue: Game doesn't appear in --list-games**
- Solution: Make sure you imported it in `src/snapflow/games/__init__.py`

**Issue: MediaPipe landmarks not detected**
- Solution: Check lighting, face angle, camera quality
- Try adjusting `min_detection_confidence` parameter

**Issue: Poor performance / Low FPS**
- Solution: Reduce processing per frame, use caching, lower resolution
- Profile with `--verbose` flag

**Issue: Landmarks jumping around**
- Solution: Increase `min_tracking_confidence`
- Smooth values over multiple frames

**Issue: Effect looks wrong**
- Solution: Check BGR vs RGB color space conversions
- Verify landmark indices are correct

---

## Example Game Ideas to Try

1. **Eyebrow Raise Counter** - Count eyebrow raises
2. **Smile Detector** - Detect and score smiles
3. **Head Shake Game** - Shake head to control something
4. **Nose Touch Game** - Move nose to touch targets
5. **Expression Mirror** - Copy AI-shown expressions
6. **Wink Counter** - Detect left vs right winks separately
7. **Face Tilt Maze** - Navigate maze by tilting face
8. **Tongue Out Detector** - Detect when tongue is visible
9. **Multi-face Game** - Multiplayer games with 2+ faces

---

## Resources

- **MediaPipe Documentation**: https://google.github.io/mediapipe/
- **OpenCV Tutorials**: https://docs.opencv.org/4.x/d9/df8/tutorial_root.html
- **Face Mesh Landmarks**: https://github.com/google/mediapipe/blob/master/docs/solutions/face_mesh.md
- **Existing Games**: Look at `src/snapflow/games/` for examples

---

## Getting Help

1. Check existing games for examples
2. Run with `--verbose` for debugging info
3. Test with `python run_game.py --list-games`
4. Review the architecture section in README.md
5. Examine test files in `tests/` directory

---

## Summary Checklist

New game creation checklist:

- [ ] Create directory `src/snapflow/games/your_game_name/`
- [ ] Create `__init__.py` with exports
- [ ] Create `processor.py` with `Processor` subclass
- [ ] Implement `process(frame, state)` method
- [ ] Register with `@register_game("name")`
- [ ] Import in `src/snapflow/games/__init__.py`
- [ ] Test with `python run_game.py your_game_name`
- [ ] Add tests in `tests/test_your_game.py` if you want to test carefully
- [ ] Document in README.md (optional)

**You're now ready to create amazing SnapFlow games! 🎮**