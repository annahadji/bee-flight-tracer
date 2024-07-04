# bee-flight-tracer ✈️ 🐝 

https://github.com/annahadji/bee-flight-tracer/assets/14828356/077495c5-5869-4920-ba1d-7a5e64542aff

## Getting started

BeeFlightTracer can be setup by cloning this repository and installing the dependencies in [requirements.txt](https://github.com/annahadji/bee-flight-tracer/blob/main/requirements.txt) into a (Python) virtual environment. BeeFlightTracer is a small Python project. The following command can be run from the root of the repo,

```
(.venv) anna@ms bee-flight-tracer % python trace_flight.py --help
usage: exit_angle_tracker.py [-h] [--select_dir]

options:
  -h, --help    show this help message and exit
  --select_dir  Select directory of videos rather than an individual file.
```

This will launch the GUI. The following key bindings are available:

| Key                       | Description                                                                                 | 
|---------------------------|---------------------------------------------------------------------------------------------|
| Left click                | (Re)select a bee to track.          |
| s                         | Advance a frame (i.e. if bee is not visible). |
| n                         | Proceeds to automatic tracking. |
| m                         | Interupts 'live' tracking. Restarts video to track manually (where bee in each frame can be selected). | 
| q                         | Saves any annotations and quits program. |

The corresponding data directory (source data) and resulting directory for storing the tracking outputs is set in the `overlay_trace.py`. The parameters of contour filtering can be adjusted in `trace_flight.py`. 

The program saves a version of the video with the tracking overlaid, as well as a [`.pkl`](https://docs.python.org/3/library/pickle.html) file with the tracked coordinates and metadata:

```
with open("my_video.pkl", "rb") as f:
    video_data = pickle.load(f)
# video_data is a dictionary with (x,y) coordinates for each frame, and an 'area' and bounding box if frame was automatically tracked.
# i.e. video_data = {0: {'x': 317, 'y': 1740, 'area': 410.5, 'boundingRec': [301, 1730, 32, 21]}, 1: ...}
```

## Built with

These scripts have been tested on MacOS.

- [tkinter](https://docs.python.org/3/library/tkinter.html#module-tkinter) - Python interface to the Tk GUI toolkit
- [OpenCV](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html) - video frame processing
