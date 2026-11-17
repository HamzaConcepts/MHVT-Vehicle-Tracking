"""Manual single-vehicle selection and CSRT tracking demonstration."""

import argparse
from pathlib import Path
from typing import Callable

import cv2


WINDOW_NAME = "MHVT - Single Vehicle Tracking"
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = PROJECT_DIR / "traffic.mp4"
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "tracked_traffic.mp4"
MAX_WORKING_DIMENSION = 1920
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
        help="Input video path (default: traffic.mp4).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Annotated output path (default: output/tracked_traffic.mp4).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print video metadata and tracker availability, then exit.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        metavar="N",
        help="Stop after N processed frames (useful for a short manual test).",
    )
    return parser.parse_args()


def get_tracker_factory() -> tuple[str, Callable[[], object]] | None:
    """Return the preferred available tracker constructor."""
    constructors = (
        ("CSRT", "TrackerCSRT_create"),
        ("CSRT (legacy)", "legacy.TrackerCSRT_create"),
        ("KCF", "TrackerKCF_create"),
        ("KCF (legacy)", "legacy.TrackerKCF_create"),
        ("MIL", "TrackerMIL_create"),
        ("MIL (legacy)", "legacy.TrackerMIL_create"),
    )
    for name, path in constructors:
        owner: object = cv2
        available = True
        for part in path.split("."):
            if not hasattr(owner, part):
                available = False
                break
            owner = getattr(owner, part)
        if available and callable(owner):
            return name, owner
    return None


def video_metadata(capture: cv2.VideoCapture) -> tuple[int, int, float, int]:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps if fps > 0 else 30.0, frame_count


def fit_size(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    scale = min(max_width / width, max_height / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_for_display(frame: object) -> object:
    height, width = frame.shape[:2]
    display_size = fit_size(width, height, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)
    if display_size == (width, height):
        return frame.copy()
    return cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)


def display_roi_to_working(
    roi: tuple[int, int, int, int],
    display_size: tuple[int, int],
    working_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    display_width, display_height = display_size
    working_width, working_height = working_size
    x, y, width, height = roi
    scale_x = working_width / display_width
    scale_y = working_height / display_height
    left = max(0, min(working_width - 1, round(x * scale_x)))
    top = max(0, min(working_height - 1, round(y * scale_y)))
    right = max(left + 1, min(working_width, round((x + width) * scale_x)))
    bottom = max(top + 1, min(working_height, round((y + height) * scale_y)))
    return left, top, right - left, bottom - top


def working_frame(frame: object) -> object:
    height, width = frame.shape[:2]
    if max(width, height) <= MAX_WORKING_DIMENSION:
        return frame
    size = fit_size(width, height, MAX_WORKING_DIMENSION, MAX_WORKING_DIMENSION)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def select_target(frame: object) -> tuple[int, int, int, int] | None:
    display = resize_for_display(frame)
    display_height, display_width = display.shape[:2]
    working_height, working_width = frame.shape[:2]
    roi = cv2.selectROI(WINDOW_NAME, display, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(WINDOW_NAME)
    if roi[2] <= 0 or roi[3] <= 0:
        return None
    return display_roi_to_working(
        tuple(int(value) for value in roi),
        (display_width, display_height),
        (working_width, working_height),
    )


def draw_status(
    frame: object,
    frame_number: int,
    status: str,
    bbox: tuple[int, int, int, int] | None = None,
) -> object:
    output = frame.copy()
    if bbox is not None:
        x, y, width, height = bbox
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 3)
        cv2.putText(
            output,
            "Target ID: 1",
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"Status: {status}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255) if status == "TARGET LOST" else (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Frame: {frame_number}",
        (15, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        "SPACE: pause/resume  R: reselect  Q: quit",
        (15, output.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def run_tracking(video_path: Path, output_path: Path, max_frames: int | None = None) -> None:
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be greater than zero.")
    tracker_info = get_tracker_factory()
    if tracker_info is None:
        raise RuntimeError(
            "No OpenCV object tracker is available. Install an OpenCV build with "
            "tracking support (opencv-contrib-python) if permitted."
        )
    tracker_name, tracker_factory = tracker_info
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width, height, fps, frame_count = video_metadata(capture)
    print(f"Video: {video_path}")
    print(f"Metadata: {width}x{height}, {fps:.2f} FPS, {frame_count} frames")
    print(f"Tracker: {tracker_name}")
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError("The video opened but did not provide a readable first frame.")

    first_working = working_frame(first_frame)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    bbox = select_target(first_working)
    if bbox is None:
        capture.release()
        cv2.destroyAllWindows()
        print("Selection cancelled; no output was written.")
        return

    output_height, output_width = first_working.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        cv2.destroyAllWindows()
        raise RuntimeError(f"Could not create output video: {output_path}")

    paused = False
    frame_number = 1
    tracker: object | None = None
    tracking_ok = False
    current_frame = first_working

    try:
        while True:
            if tracker is None:
                tracker = tracker_factory()
                tracking_ok = bool(tracker.init(current_frame, bbox))

            if not paused and frame_number > 1:
                read_ok, source_frame = capture.read()
                if not read_ok:
                    break
                current_frame = working_frame(source_frame)
                tracking_ok, tracked_bbox = tracker.update(current_frame)
                bbox = tuple(int(round(value)) for value in tracked_bbox) if tracking_ok else None

            status = "TRACKING" if tracking_ok else "TARGET LOST"
            annotated = draw_status(current_frame, frame_number, status, bbox if tracking_ok else None)
            writer.write(annotated)
            cv2.imshow(WINDOW_NAME, resize_for_display(annotated))

            delay = max(1, round(1000 / fps)) if not paused else 50
            key = cv2.waitKey(delay) & 0xFF
            if max_frames is not None and frame_number >= max_frames:
                break
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                paused = True
                new_bbox = select_target(current_frame)
                if new_bbox is not None:
                    bbox = new_bbox
                    tracker = None
                    tracking_ok = False
                else:
                    print("Selection cancelled; keeping the current tracking state.")
                paused = False
            elif not paused:
                frame_number += 1
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"Saved processed tracking video to: {output_path}")


def run_check(video_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width, height, fps, frame_count = video_metadata(capture)
    capture.release()
    print(f"Video opens successfully: {video_path}")
    print(f"Metadata: {width}x{height}, {fps:.2f} FPS, {frame_count} frames")
    tracker_info = get_tracker_factory()
    print(f"Tracker API: {tracker_info[0] if tracker_info else 'unavailable'}")
    if tracker_info is None:
        raise RuntimeError("No OpenCV tracker API is available.")
    display_size = fit_size(width, height, MAX_DISPLAY_WIDTH, MAX_DISPLAY_HEIGHT)
    test_roi = display_roi_to_working((10, 10, 100, 80), display_size, (width, height))
    print(f"Coordinate mapping check: display ROI (10, 10, 100, 80) -> {test_roi}")


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Input video was not found: {args.video}")
    if args.check:
        run_check(args.video)
    else:
        run_tracking(args.video, args.output, args.max_frames)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
