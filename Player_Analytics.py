# ======================================================
# Import Libraries
# ======================================================
import cv2
import numpy as np
from ultralytics import YOLO

# ======================================================
# System Configuration & Analytics Parameters
# ======================================================
MODEL_PATH = "yolov8n-pose.pt"
VIDEO_PATH = r"E:\Vision Project\video_2026-02-07_16-43-12.mp4"
OUTPUT_PATH = r"E:\Vision Project\sports_Football.avi"

FRAME_SKIP = 3
EMA_VEL, EMA_VIS = 0.4, 0.2
FIELD_WIDTH_METERS = 68
MIN_KP_CONF = 0.3

# ======================================================
# SKELETON DEFINITION (COLORED)
# ======================================================
SKELETON_PARTS = {
    "head": {
        "links": [(0,5),(0,6)],
        "color": (0,0,255)  # Red
    },
    "left_arm": {
        "links": [(5,7),(7,9)],
        "color": (255,0,0)        # Blue
    },
    "right_arm": {
        "links": [(6,8),(8,10)],
        "color": (0,255,255)  # Yellow
    },
    "torso": {
        "links": [(5,6),(5,11),(6,12),(11,12)],
        "color": (255,0,255)      # Purple
    },
    "left_leg": {
        "links": [(11,13),(13,15)],
        "color": (0,255,0)        # Green
    },
    "right_leg": {
        "links": [(12,14),(14,16)],
        "color": (0,165,255)      # Orange
    }
}

# ======================================================
# Model Initialization & Video Pipeline 
# ======================================================
model = YOLO(MODEL_PATH).to("cpu")
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
dt = FRAME_SKIP / fps
W,H = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"XVID"),
    fps / FRAME_SKIP,
    (W, H)
)

# ======================================================
# HOMOGRAPHY (PIXEL → METER)
# ======================================================
src = np.float32([[100,600],[1100,600],[300,200],[900,200]])
dst = np.float32([[0,0],[FIELD_WIDTH_METERS,0],[0,50],[FIELD_WIDTH_METERS,50]])
Homo,_ = cv2.findHomography(src,dst)

def px2m(p):
    return cv2.perspectiveTransform(np.array([[p]],np.float32), Homo)[0][0]

def angle(a,b,c):
    ba, bc = np.array(a)-np.array(b), np.array(c)-np.array(b)
    cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

# ======================================================
# PLAYER STATE
# ======================================================
state = {}
frame_id = 0

# ======================================================
# MAIN LOOP
# ======================================================
while True:
    ok, frame = cap.read()
    if not ok: break

    frame_id += 1
    if frame_id % FRAME_SKIP: continue

    res = model.track(frame, persist=True, verbose=False)[0]
    if res.boxes.id is None:
        out.write(frame); continue

    boxes = res.boxes.xyxy.cpu().numpy()
    ids   = res.boxes.id.cpu().numpy().astype(int)
    kps   = res.keypoints.xy.cpu().numpy()
    confs = res.keypoints.conf.cpu().numpy()

    for box,pid,kp,cf in zip(boxes,ids,kps,confs):

        st = state.setdefault(pid, {
            "prev":None,"vel":np.zeros(2),
            "speed":0,"dir":0,"torso":0
        })

        # ---------- MOTION ----------
        hip_m = px2m(kp[11])
        if st["prev"] is not None:
            vel = (hip_m - st["prev"]) / dt
            st["vel"] = EMA_VEL*vel + (1-EMA_VEL)*st["vel"]
            spd = np.linalg.norm(st["vel"])
            dir = (np.degrees(np.arctan2(st["vel"][1],st["vel"][0]))+360)%360
            st["speed"] = EMA_VIS*spd + (1-EMA_VIS)*st["speed"]
            st["dir"]   = EMA_VIS*dir + (1-EMA_VIS)*st["dir"]
        st["prev"] = hip_m

        # ---------- TORSO ----------
        if all(cf[i]>MIN_KP_CONF for i in [5,11,13]):
            t = angle(kp[5],kp[11],kp[13])
            st["torso"] = EMA_VIS*t + (1-EMA_VIS)*st["torso"]

        # ---------- DRAW BOX & INFO ----------

        # ===== Draw Bounding Box =====
        x1,y1,x2,y2 = map(int,box)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        # ===== Draw Info =====
        cv2.putText(frame,f"ID:{pid}",(x1,y1-60),0,0.5,(0,255,255),2)
        cv2.putText(frame,f"Speed:{st['speed']:.2f} m/s",(x1,y1-40),0,0.5,(0,255,0),2)
        cv2.putText(frame,f"Dir:{st['dir']:.1f}",(x1,y1-20),0,0.5,(255,255,0),2)
        cv2.putText(frame,f"Torso:{int(st['torso'])}",(x1,y1),0,0.5,(0,0,255),2)

        # ---------- Draw Keypoints ----------
        for i,(x,y) in enumerate(kp):
            if cf[i]>MIN_KP_CONF:
                cv2.circle(frame,(int(x),int(y)),4,(0,0,255),-1)

        # ---------- Draw SKELETON ----------
        for links,color in SKELETON.values():
            for a,b in links:
                if cf[a]>MIN_KP_CONF and cf[b]>MIN_KP_CONF:
                    cv2.line(frame,
                             tuple(map(int,kp[a])),
                             tuple(map(int,kp[b])),
                             color,2)

    out.write(frame)
    cv2.imshow("Sports Analytics",frame)
    if cv2.waitKey(1)==27: break

cap.release()
out.release()
cv2.destroyAllWindows()

