# SnapFlow - Real-time Vision Pipeline

A pluggable real-time vision pipeline for creating interactive games and face effects similar to ones in SnapcChat. Built with MediaPipe and OpenCV, featuring a modular architecture for developing computer vision applications.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/mmkuznecov/NotSnapChat
cd NotSnapChat
pip install -r requirements.txt

# Run your first game
python run_game.py blink_counter

# List all available games
python run_game.py --list-games
```

## Key Features

- **Pluggable Architecture** - Game registration with `@register_game()` decorator
- **Interactive Games** - Head movement and facial expression controls
- **Visual Effects** - Real-time face transformations and overlays
- **Flexible Input** - Camera or video file support
- **Modular Design** - Clean separation of concerns with minimal interfaces
- **Configurable** - Adjustable parameters for effects and games

## Available Games & Effects

<table>
<thead>
<tr>
<th>Game/Effect</th>
<th>Description</th>
<th>Controls</th>
</tr>
</thead>
<tbody>

<tr>
<td><strong>blink_counter</strong></td>
<td>Count eye blinks using Eye Aspect Ratio detection</td>
<td>Blink to score</td>
</tr>

<tr>
<td><strong>blink_counter_hard</strong></td>
<td>Harder blink detection with stricter thresholds</td>
<td>Precise blinks required</td>
</tr>

<tr>
<td><strong>eye_enlarger</strong></td>
<td>Real-time eye enlargement effect (1.4x scaling)</td>
<td>Automatic</td>
</tr>

<tr>
<td><strong>eye_enlarger_extreme</strong></td>
<td>Extreme eye enlargement effect (2.0x scaling)</td>
<td>Automatic</td>
</tr>

<tr>
<td><strong>cannon_game</strong></td>
<td>Head-tilt controlled cannon shoots falling targets</td>
<td>Head tilt + Blink to shoot</td>
</tr>

<tr>
<td><strong>cannon_game_easy</strong></td>
<td>Easy mode: 5 lives, slower enemy spawn rate</td>
<td>Head tilt + Blink to shoot</td>
</tr>

<tr>
<td><strong>cannon_game_hard</strong></td>
<td>Hard mode: 1 life, fast enemy spawn rate</td>
<td>Head tilt + Blink to shoot</td>
</tr>

<tr>
<td><strong>ghost_faces</strong></td>
<td>Shows past facial expressions as colored ghost overlays</td>
<td>Any facial expressions</td>
</tr>

<tr>
<td><strong>ghost_faces_dense</strong></td>
<td>Dense ghost effect: 100 ghosts, 5-second lifetime</td>
<td>Any facial expressions</td>
</tr>

<tr>
<td><strong>ghost_faces_sparse</strong></td>
<td>Minimal ghost effect: 20 ghosts, 2-second lifetime</td>
<td>Any facial expressions</td>
</tr>

<tr>
<td><strong>ghost_faces_outline</strong></td>
<td>Clean outline-only ghost faces with no current landmarks</td>
<td>Any facial expressions</td>
</tr>

</tbody>
</table>

### Color Legend for Ghost Faces Effects

| Time Ago | Color | Description |
|----------|-------|-------------|
| 0.0-0.5s | 🟢 **Green** | Very recent expressions |
| 0.5-1.0s | 🔵 **Blue** | Recent expressions |
| 1.0-1.5s | 🟣 **Purple** | Medium-age expressions |
| 1.5-2.0s | 🔴 **Red** | Older expressions |
| 2.0-3.0s | 🟡 **Yellow** | Ancient expressions (fading) |

## Installation

### Setup
```bash
# Clone the repository
git clone https://github.com/mmkuznecov/NotSnapChat.git
cd NotSnapChat

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python run_game.py --list-games
```

### Dependencies
```
opencv-python>=4.8.0    # Computer vision operations
mediapipe>=0.10.0       # Face detection and landmarks
numpy>=1.24.0           # Numerical computations
```

## Usage Examples

### Basic Usage
```bash
# Run games with default camera
python run_game.py blink_counter
python run_game.py eye_enlarger
python run_game.py cannon_game
python run_game.py ghost_faces
```

### Advanced Usage
```bash
# Use video file as input
python run_game.py blink_counter --source path/to/video.mp4

# Custom camera resolution
python run_game.py cannon_game --width 1280 --height 720

# Verbose mode for debugging
python run_game.py ghost_faces --verbose

# List all available games
python run_game.py --list-games
```

### Game-Specific Tips

#### Blink Counter Games
- Good lighting is essential for accurate blink detection
- Look directly at camera for best results
- Natural blinks work better than forced ones
- Try both normal and hard modes to test your blinking consistency

#### Cannon Game
- Tilt head left/right to move the blue cannon
- Blink quickly to fire green bullets at red targets
- Don't let red circles reach the bottom or you lose a life
- Start with easy mode, progress to hard mode for challenge

#### Ghost Faces Effects
- Move slowly to see beautiful motion trails
- Change expressions to create dynamic overlays  
- Hold still then move to build intensity then create trails
- Try different variants for various visual styles

#### Eye Enlarger Effects
- Works automatically once face is detected
- Good lighting ensures stable effect
- Face the camera directly for best results
- Compare normal vs extreme modes for different intensities

## Architecture Overview

SnapFlow uses a modular architecture designed for extensibility:

### Core Components (The Four Pillars)

```
SnapFlow Architecture
├── Frame Sources          # Camera, Video File input
├── Processors            # Game logic, Effects processing  
├── Stores               # Data collection, Metrics storage
├── Game Registry         # Plugin system for games
└── Main Loop            # Processing pipeline coordination
```

#### Pillar 1: Frame Sources
- **Purpose**: Capture and standardize input from different sources
- **Components**: `CameraSource` for live webcam feed, `VideoSource` for video files
- **Output**: Standardized frame dictionaries with image data, frame ID, and timestamp
- **Extensibility**: Easy to add new input sources (network streams, image sequences, etc.)

#### Pillar 2: Processors
- **Purpose**: Core game logic and effects processing
- **Components**: Individual processor classes for each game/effect
- **Input**: Frame dict and shared state from previous processors
- **Output**: Result dict with processed image, events, metrics, and state updates
- **Examples**: `BlinkGameProcessor`, `EyeEnlargerProcessor`, `HeadTiltCannonGameProcessor`

#### Pillar 3: Stores
- **Purpose**: Collect and manage data generated during processing
- **Components**: `NoOpStore` for effects that don't need data storage, `SimpleStore` for in-memory event/metric collection
- **Data**: Events (like 'blink', 'hit_target'), metrics (scores, measurements), and timestamped records
- **Extensibility**: Can be extended to save to files, databases, or external APIs

#### Pillar 4: Game Registry
- **Purpose**: Plugin system for registering and managing games
- **Components**: `@register_game()` decorator and `GAMES` dictionary
- **Functionality**: Automatic game discovery, factory pattern for game creation
- **Benefits**: Clean separation between framework and game implementations

### Add Your Own Game

Check out `NEW_GAME.md` for a step-by-step guide on creating and integrating your own games into SnapFlow.