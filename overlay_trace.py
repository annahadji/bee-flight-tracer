import argparse
import cv2
import logging
import numpy as np
import pathlib
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory


SRC_DIR = "data"
TRACKING_DIR = "tracking_data"


logger = logging.getLogger("Overlaying")
logging.basicConfig(
    format="%(asctime)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def add_txt_to_frame(frame, txt: str):
    """Add text to frame in default format."""
    cv2.putText(
        frame,
        txt,
        (100, 200),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=5,
        color=(255, 0, 0),
        thickness=5,
    )


def get_untracked_video_str(tracked_video_path):
    video_path = pathlib.Path(
        str(tracked_video_path)
        .replace(TRACKING_DIR, SRC_DIR)
        .split("-TunnelAlignment")[0]
    )
    if video_path.with_suffix(".mov").exists():
        return video_path.with_suffix(".mov")
    elif video_path.with_suffix(".MOV").exists():
        return video_path.with_suffix(".MOV")
    raise FileExistsError


def overlay_tracking_data(pkl_path):
    with open(f"{str(pkl_path)}", "rb") as f:
        video_data = pickle.load(f)

    video_path = get_untracked_video_str(pkl_path)
    cap = cv2.VideoCapture(str(video_path))
    size = (int(cap.get(3)), int(cap.get(4)))
    out_name = f"{str(pkl_path.with_suffix('')) + '-tracked.mov'}"
    out = cv2.VideoWriter(
        out_name,
        cv2.VideoWriter_fourcc(*"jpeg"),
        cap.get(cv2.CAP_PROP_FPS),
        size,
    )
    ret = True
    frame_n = 0
    while ret:
        ret, frame = cap.read()
        add_txt_to_frame(frame, f"{frame_n}")
        if frame_n in video_data:
            # Bounding box coords are nan if point was manually tracked
            if np.isnan(video_data[frame_n]["boundingRec"]).any() == False:
                x, y, w, h = video_data[frame_n]["boundingRec"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x_center = video_data[frame_n]["x"]
            y_center = video_data[frame_n]["y"]
            cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)
        out.write(frame)
        frame_n += 1
    logger.info("Saved overlaid tracks to %s", out_name)
    cap.release()
    cv2.destroyAllWindows()


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
        for video in data_path.glob("*.pkl"):
            overlay_tracking_data(video)
    else:
        data_path = pathlib.Path(askopenfilename(initialdir=root))
        overlay_tracking_data(data_path)
