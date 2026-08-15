"""Shared hardware constants for the OMX evaluation scripts.

act-sim runs policies; teleoperation and data collection live in omx-training,
which has its own utils/config.py. The two files are otherwise identical, so
keep this one limited to what act-sim actually reads -- a leader port or task
string here would be a second source of truth for something this repo does not
do, and importing the wrong copy is a real failure mode (both repos expose a
top-level ``utils`` package).
"""

# ── Hardware Ports ──
FOLLOWER_PORT = "/dev/tty.usbmodem1401"

# ── Joint Configuration ──
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# ── Camera Defaults ──
# Map of camera name → OpenCV device index. Names appear in datasets as
# observation.images.<name>, and the trained policy is fed frames matched by
# these names.
#
# On macOS, index 0 is typically the built-in FaceTime camera; external USB
# cameras are usually 1, 2, ... Run:
#   uv run python -c "from lerobot.cameras.opencv.camera_opencv import OpenCVCamera; print(OpenCVCamera.find_cameras())"
# to enumerate.
CAMERAS = {
    "wrist": 1,  # Innomaker wrist-mounted camera
    "top": 0,    # Logitech 1080P overhead camera
}
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# ── Control Loop FPS ──
FPS = 30
