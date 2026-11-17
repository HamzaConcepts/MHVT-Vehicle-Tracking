"""BASELINE (kept for the evaluation write-up): automatic multi-vehicle tracking demo for the MHVT project."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_VIDEO = PROJECT_DIR / "traffic.mp4"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_VIDEO = OUTPUT_DIR / "tracked_traffic.mp4"
MODEL_NAME = "yolov8n.pt"
TRACKER_CONFIG = PROJECT_DIR / "bytetrack_mhvt.yaml"
DISPLAY_SIZE = (960, 540)
INFERENCE_SIZE = 960
DETECTION_CONFIDENCE = 0.20
MIN_TRACK_CONFIRMATIONS = 2

# COCO class ID 2 is car.
CAR_CLASS_ID = 2
CLASS_NAME = "car"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Use webcam 0 instead of traffic.mp4.",
    )
    parser.add_argument(
        "--test-frames",
        type=int,
        metavar="N",
        help="Process N video frames without a window or output file.",
    )
    return parser.parse_args()


def box_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def main() -> None:
    """Run automatic vehicle tracking on traffic.mp4 or webcam 0."""
    args = parse_args()
    source: str | int = 0 if args.camera else str(INPUT_VIDEO)
    display = args.test_frames is None
    save_output = not args.camera and args.test_frames is None

    if not args.camera and not INPUT_VIDEO.is_file():
        print(f"Input video was not found: {INPUT_VIDEO}")
        print("Place a prerecorded video named traffic.mp4 in the project folder.")
        return

    if not TRACKER_CONFIG.is_file():
        print(f"Tracker configuration was not found: {TRACKER_CONFIG}")
        return

    if args.test_frames is not None and args.test_frames <= 0:
        print("--test-frames must be greater than zero.")
        return

    if save_output:
        OUTPUT_DIR.mkdir(exist_ok=True)

    try:
        print(f"Loading {MODEL_NAME} for CPU tracking...")
        model = YOLO(MODEL_NAME)
    except Exception as error:
        print(f"Could not load the YOLO model: {error}")
        return

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"Could not open tracking source: {source}")
        return

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = source_fps if source_fps > 0 else 30.0
    writer = None
    frame_number = 0
    all_track_ids: set[int] = set()
    previous_track_ids: set[int] = set()
    previous_track_boxes: dict[int, tuple[int, int, int, int]] = {}
    track_confirmations: dict[int, int] = {}
    missing_track_frames = 0
    identity_switch_candidates = 0
    frames_with_tracks = 0
    track_observations = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                if frame_number == 0:
                    print("The video could not provide a readable frame.")
                else:
                    print("Reached the end of the video.")
                break

            frame_number += 1
            source_height, source_width = frame.shape[:2]
            display_width, display_height = DISPLAY_SIZE
            x_scale = display_width / source_width
            y_scale = display_height / source_height

            # One sequential model.track call with persist=True keeps ByteTrack state.
            results = model.track(
                frame,
                persist=True,
                tracker=str(TRACKER_CONFIG),
                classes=[2],
                conf=DETECTION_CONFIDENCE,
                imgsz=INFERENCE_SIZE,
                device="cpu",
                verbose=False,
            )
            result = results[0]
            annotated_frame = cv2.resize(
                frame, DISPLAY_SIZE, interpolation=cv2.INTER_AREA
            )
            current_track_ids: set[int] = set()
            current_track_boxes: dict[int, tuple[int, int, int, int]] = {}
            confirmed_track_ids: set[int] = set()
            tracked_count = len(result.boxes) if result.boxes is not None else 0
            detection_count = 0

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                track_ids = (
                    result.boxes.id.int().cpu().tolist()
                    if result.boxes.id is not None
                    else [None] * tracked_count
                )

                for box, class_id, confidence, track_id in zip(
                    boxes, class_ids, confidences, track_ids
                ):
                    if track_id is None:
                        continue
                    track_id = int(track_id)
                    current_track_ids.add(track_id)
                    all_track_ids.add(track_id)
                    detection_count += 1
                    track_confirmations[track_id] = (
                        track_confirmations.get(track_id, 0) + 1
                    )
                    source_x1, source_y1, source_x2, source_y2 = box
                    x1 = int(source_x1 * x_scale)
                    y1 = int(source_y1 * y_scale)
                    x2 = int(source_x2 * x_scale)
                    y2 = int(source_y2 * y_scale)
                    current_track_boxes[track_id] = (x1, y1, x2, y2)
                    if track_confirmations[track_id] < MIN_TRACK_CONFIRMATIONS:
                        continue
                    confirmed_track_ids.add(track_id)
                    if class_id != CAR_CLASS_ID:
                        continue
                    label = f"{CLASS_NAME} ID {track_id} {confidence:.2f}"

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )
                    label_top = max(y1 - 8, 20)
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, label_top),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            disappeared_ids = previous_track_ids - current_track_ids
            new_ids = current_track_ids - previous_track_ids
            missing_track_frames += len(disappeared_ids)
            for new_id in new_ids:
                for disappeared_id in disappeared_ids:
                    if box_iou(
                        current_track_boxes[new_id],
                        previous_track_boxes[disappeared_id],
                    ) >= 0.5:
                        identity_switch_candidates += 1
            if current_track_ids:
                frames_with_tracks += 1
                track_observations += len(current_track_ids)
            tracked_count = len(confirmed_track_ids)
            previous_track_ids = current_track_ids
            previous_track_boxes = current_track_boxes
            cv2.putText(
                annotated_frame,
                f"Confirmed tracks: {tracked_count}  Detection-backed IDs: {detection_count}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                "Green boxes are detection-backed tracks; no guessed boxes are drawn",
                (10, 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if save_output and writer is None:
                height, width = annotated_frame.shape[:2]
                codec = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(OUTPUT_VIDEO), codec, fps, (width, height)
                )
                if not writer.isOpened():
                    print(f"Could not create output video: {OUTPUT_VIDEO}")
                    return

            if writer is not None:
                writer.write(annotated_frame)
            if display:
                cv2.imshow("MHVT - Automatic Vehicle Tracking", annotated_frame)
            if display and cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break
            if args.test_frames is not None and frame_number >= args.test_frames:
                break
    except Exception as error:
        print(f"Video processing stopped because of an error: {error}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if args.test_frames is not None:
        print(f"Test frames processed: {frame_number}")
        print(f"Unique track IDs observed: {len(all_track_ids)}")
        print(f"Track IDs still active on final frame: {len(previous_track_ids)}")
        print(f"Frames with active track IDs: {frames_with_tracks}")
        print(f"Track observations: {track_observations}")
        print(f"Temporary missing-track observations: {missing_track_frames}")
        print(
            "Identity-switch candidates: "
            f"{identity_switch_candidates} (IoU heuristic; no ground-truth labels)"
        )
    elif OUTPUT_VIDEO.is_file():
        print(f"Annotated video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
