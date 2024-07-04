import argparse
import cv2
import numpy as np
import logging
import pathlib
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory

from overlay_tracks import (
    overlay_tracking_data,
    add_txt_to_frame,
    TRACKING_DIR,
    SRC_DIR,
)

logger = logging.getLogger("Tracking")
logging.basicConfig(
    format="%(asctime)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

MIN_BEE_CONTOUR = 100
MAX_BEE_CONTOUR = 5000
DIST_TO_ROI_THRESHOLD = 300
BLUR = (35, 35)

ROI_WINDOW = "Select ROI"
TRACK_WINDOW = "Tracking"
FG_WINDOW = "Foreground mask"
MAN_WINDOW = "Manual tracking"

CONTINUE_KEY = ord("n")
QUIT_KEY = ord("q")
MANUAL_KEY = ord("m")


def filter_lines(lines, img_H):
    """Filter detected lines to be horizontal and within middle
    band on image."""
    filtered_lines = []
    for line in lines:
        _, y1, _, y2 = line[0]
        if abs(y1 - y2) < 5:  # Horizontal lines
            # Select only in centre band
            center_band_top = img_H // 3
            center_band_bottom = 2 * img_H // 3
            if (
                center_band_top <= max(y1, y2) <= center_band_bottom
                and center_band_top <= min(y1, y2) <= center_band_bottom
            ):
                filtered_lines.append(line)
    return filtered_lines


def find_nearest_line(lines, clicked_point):
    min_distance = np.inf
    nearest_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        distance = np.abs(
            (y2 - y1) * clicked_point[0]
            - (x2 - x1) * clicked_point[1]
            + x2 * y1
            - y2 * x1
        ) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_line = line
    return nearest_line


def compute_angle_degrees(line):
    """Computes the angle of a line relative to 0 degrees East using negative angles."""
    x1, y1, x2, y2 = line[0]
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    normalized_angle_rad = (angle_rad + np.pi) % (2 * np.pi) - np.pi
    return np.rad2deg(normalized_angle_rad)


def determine_video_straightness(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    labelled_frame = frame.copy()
    angle = None
    video_path = pathlib.Path(str(video_path).replace(SRC_DIR, TRACKING_DIR))
    video_path.parent.mkdir(parents=True, exist_ok=True)

    def mouse_callback(event, x, y, flags, params):
        nonlocal frame, video_path, angle
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_point = (x, y)
            nearest_line = find_nearest_line(lines, clicked_point)
            if nearest_line is not None:
                angle = compute_angle_degrees(nearest_line)
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(
                    f"{video_path.with_suffix('')}-tunnelAlignment{angle:.2f}deg.png",
                    frame,
                )
                logger.info(f"Line selected: {angle:.2f} degrees")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Edge detection
    lines = cv2.HoughLinesP(  # Detect lines
        edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=15
    )
    if lines is not None:
        lines = filter_lines(lines, frame.shape[0])
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(labelled_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.namedWindow("AlignTunnel")
    cv2.setMouseCallback("AlignTunnel", mouse_callback)
    cv2.imshow("AlignTunnel", labelled_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()
    if angle == None:
        angle = float(input("Enter tunnel alignment (degrees):"))
    return angle


def select_init_bee_ROI(video_path):
    """Advance through first frames of video until bee is visible.
    Click to select bee and press 'n' to continue."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    add_txt_to_frame(frame, "0")
    cv2.imshow(ROI_WINDOW, frame)
    roi = []

    def mouse_callback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info("ROI %d, %d", x, y)
            roi.clear()
            roi.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(ROI_WINDOW, frame)

    cv2.setMouseCallback(ROI_WINDOW, mouse_callback)
    key = cv2.waitKey(0)

    start_frame = 0
    while key != QUIT_KEY and key != CONTINUE_KEY:
        ret, frame = cap.read()
        start_frame += 1
        logger.info("Advanced to frame %d", start_frame)
        add_txt_to_frame(frame, f"{start_frame}")
        cv2.imshow(ROI_WINDOW, frame)
        key = cv2.waitKey(0)

    cap.release()
    cv2.destroyWindow(ROI_WINDOW)
    if key == QUIT_KEY:
        logger.info("Exiting... Video skipped.")
        exit()
    return start_frame, roi[0]


def step_through_and_select_bee(video_path, frame_n, bkg_subtractor):
    """Continue advancing through a video from a particular point until
    bee is seen again. Either reselect bee and press 'n' to continue
    tracking or 'q' if bee is not seen again."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, BLUR, 0)
    bkg_subtractor.apply(frame)
    add_txt_to_frame(frame, f"{frame_n}")
    cv2.imshow(TRACK_WINDOW, frame)
    roi = []

    def mouse_callback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info("ROI %d, %d", x, y)
            roi.clear()
            roi.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(TRACK_WINDOW, frame)

    cv2.setMouseCallback(TRACK_WINDOW, mouse_callback)
    key = cv2.waitKey(0)

    start_frame = 0
    while key != QUIT_KEY and key != CONTINUE_KEY:
        ret, frame = cap.read()
        start_frame += 1
        if ret == False:
            key = QUIT_KEY
            break
        add_txt_to_frame(frame, f"{start_frame + frame_n}")
        frame = cv2.GaussianBlur(frame, BLUR, 0)
        fg_mask = bkg_subtractor.apply(frame)
        cv2.imshow(FG_WINDOW, fg_mask)
        cv2.imshow(TRACK_WINDOW, frame)
        logger.info("Advanced (%d) to frame: %d", start_frame, start_frame + frame_n)
        key = cv2.waitKey(0)

    if key == QUIT_KEY:  # Bee not found
        return frame_n, False, bkg_subtractor

    cap.release()
    return start_frame + frame_n, roi[0], bkg_subtractor


def manual_tracking(video_path):
    """Manually track insect through video."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    add_txt_to_frame(frame, "0")
    cv2.imshow(MAN_WINDOW, frame)
    roi = []
    start_frame = 0
    video_data = {}

    def mouse_callback(event, x, y, flags, params):
        nonlocal frame, start_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info("ROI %d, %d", x, y)
            roi.clear()
            roi.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(MAN_WINDOW, frame)
            video_data[start_frame] = {
                "x": x,
                "y": y,
                "area": np.nan,
                "boundingRec": [np.nan, np.nan, np.nan, np.nan],
            }
            print("Wrote data for frame", start_frame)
            start_frame += 1
            ret, frame = cap.read()
            cv2.imshow(MAN_WINDOW, frame)

    cv2.setMouseCallback(MAN_WINDOW, mouse_callback)
    key = cv2.waitKey(0)

    while key != CONTINUE_KEY and key != QUIT_KEY:
        ret, frame = cap.read()
        start_frame += 1
        logger.info("Advanced to frame %d", start_frame)
        add_txt_to_frame(frame, f"{start_frame}")
        cv2.imshow(MAN_WINDOW, frame)
        key = cv2.waitKey(0)

    cap.release()
    cv2.destroyWindow(MAN_WINDOW)
    return video_data


def track_bee(video_path):
    logger.info("Selected video: %s", video_path)
    # Get and save alignment of tunnel (optional)
    # angle = determine_video_straightness(pathlib.Path(video_path))
    # logger.info(f"Tunnel alignment angle: {angle:.2f} degrees")

    # Define object tracking variables and select ROI
    x_center, y_center = 0, 0
    tracking_bee = True
    start_n, (roi_x_center, roi_y_center) = select_init_bee_ROI(video_path)
    x_center, y_center, w, h = roi_x_center, roi_y_center, 40, 40
    x, y = int(x_center - w / 2), int(y_center - h / 2)
    video_data = {
        start_n: {
            "x": roi_x_center,
            "y": roi_y_center,
            "area": np.nan,
            "boundingRec": [np.nan, np.nan, np.nan, np.nan],
        }
    }

    # Play video from first frame where bee is visible
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_n)
    bkg_subtractor = cv2.createBackgroundSubtractorMOG2()
    ret, frame = cap.read()
    frame = cv2.GaussianBlur(frame, BLUR, 0)  # Blur to remove noise
    bkg_subtractor.apply(frame)  # Update bkg subtractor
    frame_n = start_n
    while True:
        if tracking_bee is False:  # Advance until you find bee again
            logger.info("Advance and reselect bee or 'q' to quit...")
            skip_to, roi, bkg_subtractor = step_through_and_select_bee(
                video_path, frame_n, bkg_subtractor
            )
            frame_n = skip_to
            if roi is False:  # No bee found
                logger.info("No bee in remaining video.")
                break
            logger.info("Bee selected. Continuing tracking...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n + 1)  # Else, continue tracking
            roi_x_center, roi_y_center = roi
            tracking_bee = True
            video_data[frame_n] = {
                "x": roi_x_center,
                "y": roi_y_center,
                "area": np.nan,
                "boundingRec": [np.nan, np.nan, np.nan, np.nan],
            }
        ret, frame = cap.read()
        frame_n += 1
        if ret == False:
            logger.info("End of video.")
            break

        # Track bee by finding the closet contour to ROI (the bee)
        frame = cv2.GaussianBlur(frame, BLUR, 0)
        foreground_mask = bkg_subtractor.apply(frame)
        contours, _ = cv2.findContours(
            foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [
            c
            for c in contours
            if cv2.contourArea(c) > MIN_BEE_CONTOUR
            and cv2.contourArea(c) < MAX_BEE_CONTOUR
        ]  # Filter those of relevant 'bee' size

        def roi_distance(contour):
            x, y, w, h = cv2.boundingRect(contour)
            x_center = x + w / 2
            y_center = y + h / 2
            return np.sqrt(
                (x_center - roi_x_center) ** 2 + (y_center - roi_y_center) ** 2
            )

        if len(contours) > 0:
            contours = sorted(contours, key=roi_distance)  # Candidate objects
            largest_contour = contours[0]
            x, y, w, h = cv2.boundingRect(largest_contour)
            x_center = int(x + w / 2)
            y_center = int(y + h / 2)
            dist_to_roi = np.sqrt(
                (x_center - roi_x_center) ** 2 + (y_center - roi_y_center) ** 2
            )  # Dist of contour bbox (candidate bee) to previous ROI (bee)
            logger.info("Frame %d. Dist to ROI %.2f", frame_n, dist_to_roi)
            roi_x_center = x_center  # Update ROI for next loop
            roi_y_center = y_center

            if dist_to_roi > DIST_TO_ROI_THRESHOLD:
                tracking_bee = False  # Bee out of frame
            else:
                tracking_bee = True

            if tracking_bee:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                foreground_mask = cv2.cvtColor(foreground_mask, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(foreground_mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)
                video_data[frame_n] = {
                    "x": roi_x_center,
                    "y": roi_y_center,
                    "area": cv2.contourArea(largest_contour),
                    "boundingRec": [x, y, w, h],
                }
        else:
            if frame_n > 1:  # Due to initial bk subtraction
                tracking_bee = False
        add_txt_to_frame(frame, f"{frame_n}")
        cv2.imshow(FG_WINDOW, foreground_mask)
        cv2.imshow(TRACK_WINDOW, frame)
        key = cv2.waitKey(1)
        if key & 0xFF == MANUAL_KEY:
            logger.info("Launching manual mode...")
            video_data = manual_tracking(video_path)
            break
        if cv2.waitKey(1) & 0xFF == QUIT_KEY:
            logger.info("Tracking aborted.")
            break

    cap.release()
    cv2.destroyAllWindows()
    tracking_vid_path = pathlib.Path(video_path.replace(SRC_DIR, TRACKING_DIR))
    tracking_vid_path.parent.mkdir(parents=True, exist_ok=True)
    # filename = tracking_vid_path.with_stem(
    #     tracking_vid_path.stem + f"-TunnelAlignment{angle:.2f}deg.pkl"
    # ).with_suffix("")
    filename = tracking_vid_path.with_stem(
        tracking_vid_path.stem + f".pkl"
    ).with_suffix("")    
    with open(f"{filename}", "wb") as f:
        pickle.dump(video_data, f)
    overlay_tracking_data(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--select_dir",
        help="Select directory of videos rather than an individual file.",
        action="store_true",
    )
    args = parser.parse_args()
    Tk().withdraw()
    root = pathlib.Path.cwd()
    if args.select_dir:
        data_path = pathlib.Path(askdirectory(initialdir=root))
        videos = list(data_path.rglob("*.mov")) + list(data_path.rglob("*.MOV"))
        videos = sorted(videos, key=lambda x: x.name)
        tracking_vid = list(pathlib.Path(TRACKING_DIR).rglob("*.pkl"))
        i = 1
        for video in videos:
            logger.info("%d / %d", i, len(videos))
            i += 1
            pkl_file = pathlib.Path(
                str(video.with_suffix("")).replace(SRC_DIR, TRACKING_DIR)
            )
            if len(list(pkl_file.parents[0].glob("*" + pkl_file.name + "*.pkl"))) > 0:
                logger.info("Skipping %s...", str(video))
                continue
            track_bee(str(video))
    else:
        data_path = pathlib.Path(askopenfilename(initialdir=root))
        track_bee(str(data_path))
