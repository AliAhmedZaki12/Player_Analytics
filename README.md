# Player Motion Analytics 

---
## Description

This project analyzes football match videos to detect players using **YOLOv8-Pose**, estimate their poses, track each player consistently, and calculate physical motion metrics such as **velocity, speed, direction, and torso angle**. The system transforms 2D pixel coordinates to **real-world meters** and visualizes all data with smooth, professional output.

The output is an **annotated video** showing players with **bounding boxes, skeletons, keypoints, and motion metrics** for each frame.

> Designed as a **real-world sports analytics framework**, it can be extended for motion analysis, performance tracking, and biomechanics research.

---

## Features

* **Player Detection & Tracking:** Detects all players and maintains unique IDs across frames using `persist=True`.
* **Pose Estimation:** Computes 17 keypoints for each player with confidence scores.
* **Pixel-to-Meter Conversion:** Maps 2D coordinates to field meters using **homography**.
* **Motion Analysis:** Computes player metrics:

  * Velocity vector (m/s)
  * Speed magnitude (m/s)
  * Movement direction (degrees)
  * Torso angle (degrees) for posture
* **Noise Reduction:** Smooths velocity and metrics using **Exponential Moving Average (EMA)** for both physics calculations and visualization.
* **Visualization:** Draws bounding boxes, skeletons, keypoints, and motion metrics as text on each player.
* **Output Video:** Saves processed frames into a video file with clear visual annotations.

---

## Technical Details

| Feature                     | Method / Implementation                         | Notes / Benefits                                   |
| --------------------------- | ----------------------------------------------- | -------------------------------------------------- |
| Player Detection & Tracking | `model.track(frame, persist=True)`              | Maintains consistent IDs across frames             |
| Keypoint Estimation         | `res.keypoints.xy`                              | 17 keypoints per player with confidence scores     |
| Confidence Filtering        | `MIN_KP_CONF = 0.3`                             | Ignores low-confidence joints                      |
| Pixel → Meter Conversion    | `cv2.findHomography(src, dst)` & `px2m(kp[11])` | Allows velocity calculation in real-world units    |
| Velocity & Speed            | `vel = (hip_m - st["prev"]) / dt`               | Smoothed using EMA (`EMA_VEL` / `EMA_VIS`)         |
| Direction Calculation       | `np.degrees(np.arctan2(vel[1], vel[0]))`        | Smoothed for visual clarity                        |
| Torso Angle                 | `angle(kp[5], kp[11], kp[13])`                  | Smoothed for stable posture analysis               |
| Bounding Boxes & Skeletons  | `cv2.rectangle` & `cv2.line`                    | Visualize player and joint connections             |
| Motion Metric Display       | `cv2.putText`                                   | Speed, direction, torso angle displayed per player |

---

## Requirements

* **Python:** 3.10+
* **Libraries:**

  * OpenCV (`opencv-python`)
  * Numpy (`numpy`)
  * Ultralytics YOLOv8 (`ultralytics`)
  * SciPy (`scipy`)

```bash
pip install opencv-python numpy ultralytics scipy
```

---

## Configuration Parameters

| Parameter            | Description                                    | Default / Example                         |
| -------------------- | ---------------------------------------------- | ----------------------------------------- |
| `MODEL_PATH`         | Path to YOLOv8-Pose model                      | `"yolov8n-pose.pt"`                       |
| `VIDEO_PATH`         | Input video path                               | `"E:/Vision Project/video.mp4"`           |
| `OUTPUT_PATH`        | Output annotated video path                    | `"E:/Vision Project/sports_Football.avi"` |
| `FRAME_SKIP`         | Process every Nth frame to reduce noise        | 3                                         |
| `EMA_VEL`            | Smoothing for velocity vector                  | 0.4                                       |
| `EMA_VIS`            | Smoothing for displayed metrics                | 0.2                                       |
| `FIELD_WIDTH_METERS` | Football field width for homography conversion | 68                                        |
| `MIN_KP_CONF`        | Minimum confidence for keypoints               | 0.3                                       |

> Adjust these values according to your video resolution, field perspective, and analysis requirements.

---

## Code Structure

### 1. Imports & Configuration

* Load libraries: OpenCV, NumPy, Ultralytics YOLOv8.
* Define constants: paths, smoothing coefficients, frame skip, field dimensions, minimum confidence.

### 2. Homography & Coordinate Mapping

* Define `src` (image points) and `dst` (real-world points) to map pixels → meters.
* Function `px2m(p)` converts any 2D point into meters.
* Use `angle(a,b,c)` to compute joint angles (e.g., torso angle).

### 3. Model & Video Initialization

* Load YOLOv8-Pose model.
* Open input video and initialize output video writer.
* Extract video **FPS** for accurate velocity calculation.

### 4. Main Loop

* Read frame by frame, skip frames according to `FRAME_SKIP`.
* Detect players with `model.track(frame, persist=True)`.
* Extract **bounding boxes, IDs, keypoints, and confidence scores**.

### 5. Motion Analysis

* Compute **hip position in meters** (`kp[11]` for torso center).
* Calculate **velocity vector** and **speed**:

```python
vel = (hip_m - st["prev"]) / dt
st["vel"] = EMA_VEL * vel + (1 - EMA_VEL) * st["vel"]
st["speed"] = EMA_VIS * np.linalg.norm(st["vel"]) + (1 - EMA_VIS) * st["speed"]
```

* Compute **movement direction** and **torso angle**, both smoothed using EMA.

### 6. Visualization

* Draw **bounding boxes** (green) around each player.
* Draw **keypoints** (red circles) if confidence > `MIN_KP_CONF`.
* Draw **skeletons** connecting joints using predefined links and colors.
* Overlay **motion metrics** (speed, direction, torso angle, ID) above each player.

### 7. Output & Cleanup

* Write processed frames to output video.
* Display video in real-time (`cv2.imshow`).
* Release video resources and destroy all OpenCV windows.

---

## Visual Output

* **Bounding Boxes:** Green rectangles around each player
* **Skeletons:** Colored lines connecting keypoints
* **Keypoints:** Red circles on each joint
* **Player IDs:** Displayed above bounding box
* **Speed & Direction:** Displayed as overlay text
* **Torso Angle:** Displayed for posture analysis

