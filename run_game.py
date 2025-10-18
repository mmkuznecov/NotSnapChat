"""
SnapFlow - Real-time Vision Pipeline Entry Point

Usage:
    python run_game.py blink_counter
    python run_game.py eye_enlarger --source video.mp4
    python run_game.py blink_counter --width 1280 --height 720 --verbose
"""

import sys
import argparse
from pathlib import Path

# Add src to path so we can import snapflow
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import games to register them
import snapflow.games  # This will register all games
from snapflow.core.main import run_game
from snapflow.core.registry import GAMES


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SnapFlow - Real-time Vision Pipeline for Effects and Games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available games:
{chr(10).join(f'  {name}' for name in sorted(GAMES.keys()))}

Examples:
  python run_game.py blink_counter
  python run_game.py eye_enlarger --source camera
  python run_game.py blink_counter --source video.mp4 --verbose
  python run_game.py eye_enlarger_extreme --width 1280 --height 720
  python run_game.py --list-games
        """,
    )

    parser.add_argument("game", nargs="?", help="Game/effect to run")
    parser.add_argument(
        "--source",
        default="camera",
        help='Input source: "camera" or path to video file (default: camera)',
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Camera width in pixels (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Camera height in pixels (default: 480)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--list-games", action="store_true", help="List available games and exit"
    )

    args = parser.parse_args()

    # List games if requested
    if args.list_games:
        print("Available games:")
        for name in sorted(GAMES.keys()):
            print(f"  {name}")
        return

    # Check if game was provided
    if not args.game:
        print("Error: Game name is required")
        print(f"Available games: {', '.join(sorted(GAMES.keys()))}")
        print("Use --list-games to see all available games")
        return

    # Validate game name
    if args.game not in GAMES:
        print(f"Error: Unknown game '{args.game}'")
        print(f"Available games: {', '.join(sorted(GAMES.keys()))}")
        print("Use --list-games to see all available games")
        return

    # Run the game
    try:
        run_game(
            game_name=args.game,
            source_type=args.source,
            width=args.width,
            height=args.height,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
