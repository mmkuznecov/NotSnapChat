"""Main processing loop for SnapFlow"""

import cv2
import time
from .source import CameraSource, VideoSource
from .registry import get_game


def run_game(game_name, source_type="camera", width=640, height=480, verbose=False):
    """
    Main processing loop for running games

    Args:
        game_name: Name of registered game to run
        source_type: 'camera' or path to video file
        width: Camera width (ignored for video files)
        height: Camera height (ignored for video files)
        verbose: Print debug information
    """
    print(f"Starting SnapFlow game: {game_name}")

    # Get game components
    try:
        processors, store = get_game(game_name)
        print(f"Loaded {len(processors)} processor(s)")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Create source
    if source_type == "camera":
        source = CameraSource(width=width, height=height)
        print(f"Using camera source ({width}x{height})")
    else:
        source = VideoSource(source_type)
        print(f"Using video source: {source_type}")

    # Initialize everything
    print("Initializing components...")
    source.open()
    store.open()
    for i, processor in enumerate(processors):
        processor.open()
        if verbose:
            print(f"  Processor {i+1}: {processor.__class__.__name__}")

    # Performance tracking
    frame_count = 0
    start_time = time.time()
    last_fps_time = start_time
    fps = 0

    try:
        state = {}
        print("Starting main loop (Press 'q' or ESC to quit)")

        while True:
            loop_start = time.time()

            # Read frame
            frame = source.read()
            if frame is None:
                print("End of stream reached")
                break

            # Process through chain
            current_frame = frame
            all_events = []
            all_metrics = {}

            for processor in processors:
                result = processor.process(current_frame, state)

                # Update frame if processor returned one
                if "image" in result and result["image"] is not None:
                    current_frame = {**current_frame, "image": result["image"]}

                # Collect events and metrics
                if "events" in result:
                    all_events.extend(result["events"])

                if "metrics" in result:
                    all_metrics.update(result["metrics"])

                # Update shared state
                if "state" in result:
                    state.update(result["state"])

            # Store results
            store.record({"events": all_events, "metrics": all_metrics})

            # Display frame with FPS counter
            img_display = current_frame["image"].copy()
            cv2.putText(
                img_display,
                f"FPS: {fps:.1f}",
                (img_display.shape[1] - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("SnapFlow", img_display)

            # Performance tracking
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:  # Update FPS every second
                fps = frame_count / (current_time - start_time)
                last_fps_time = current_time
                if verbose:
                    print(
                        f"FPS: {fps:.1f}, Events: {all_events}, "
                        f"State keys: {list(state.keys())}"
                    )

            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord("q")]:  # ESC or Q
                print("Exit requested by user")
                break

            # Maintain target framerate (optional)
            loop_time = time.time() - loop_start
            target_fps = 30
            target_time = 1.0 / target_fps
            if loop_time < target_time:
                time.sleep(target_time - loop_time)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean shutdown
        print("Cleaning up...")
        for processor in processors:
            processor.close()
        store.close()
        source.close()
        cv2.destroyAllWindows()

        # Print final stats
        total_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {total_time:.1f}s")
        print(f"Average FPS: {frame_count / total_time:.1f}")

        # Print stored data summary
        data = store.get_data()
        if "events" in data and data["events"]:
            print(f"Total events: {len(data['events'])}")
        if "metrics" in data and data["metrics"]:
            print(f"Total metrics recorded: {len(data['metrics'])}")
