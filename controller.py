import os
import sys
import cv2
import time
import math
import ctypes
import urllib.request
from collections import deque
import numpy as np
import torch
import mediapipe as mp
import pyautogui

from filters import Point2DOneEuroFilter
from model import TemporalGestureCNN, GESTURE_CLASSES, normalize_landmarks
from telemetry import AsyncTelemetryLogger
from train import train_and_export

# Configure PyAutoGUI for zero-delay clicks/keys
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0
pyautogui.MINIMUM_DURATION = 0.0
pyautogui.MINIMUM_SLEEP = 0.0
SCREEN_W, SCREEN_H = pyautogui.size()

# Native Windows fast cursor mover (0.001 ms overhead vs PyAutoGUI's 15 ms block)
IS_WINDOWS = sys.platform.startswith("win")
def fast_move_cursor(x: int, y: int):
    if IS_WINDOWS:
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
    else:
        pyautogui.moveTo(int(x), int(y), _pause=False)

# FSM States (Section 7.1.3)
STATE_IDLE = 0
STATE_PRIMED = 1
STATE_ENGAGED = 2
STATE_COMMAND = 3

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]


class UnifiedHandTracker:
    """
    Optimized Hand Tracker using temporal VIDEO tracking mode so palm detection
    only runs when a hand enters the frame, eliminating per-frame detection lag.
    """
    def __init__(self):
        self.use_tasks_api = False
        self.last_ts_ms = 0
        try:
            from mediapipe.python.solutions import hands as mp_hands
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,  # Lite model for ultra-fast 30+ FPS edge inference
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            print("[MediaPipe] Loaded fast Solutions Hands API (model_complexity=0).")
        except (ImportError, AttributeError):
            self.use_tasks_api = True
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = "weights/hand_landmarker.task"
            os.makedirs("weights", exist_ok=True)
            if not os.path.exists(model_path):
                print("[MediaPipe] Downloading hand_landmarker.task model...")
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,  # VIDEO mode enables fast temporal tracking
                num_hands=1,
                min_hand_detection_confidence=0.65,
                min_hand_presence_confidence=0.65,
                min_tracking_confidence=0.65
            )
            self.detector = mp_vision.HandLandmarker.create_from_options(options)
            print("[MediaPipe] Loaded fast Tasks HandLandmarker API (VIDEO mode).")

    def process(self, rgb_frame):
        if not self.use_tasks_api:
            res = self.hands.process(rgb_frame)
            if res.multi_hand_landmarks:
                return res.multi_hand_landmarks[0].landmark
            return None
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            ts_ms = int(time.perf_counter() * 1000)
            if ts_ms <= self.last_ts_ms:
                ts_ms = self.last_ts_ms + 1
            self.last_ts_ms = ts_ms
            res = self.detector.detect_for_video(mp_image, ts_ms)
            if res.hand_landmarks:
                return res.hand_landmarks[0]
            return None

    @staticmethod
    def draw_landmarks(frame, landmarks):
        h, w, _ = frame.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, pts[start_idx], pts[end_idx], (220, 220, 220), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 165, 255), cv2.FILLED)


class IntentGatedGestureController:
    def __init__(self, weights_path: str = "weights/gesture_1dcnn.pth"):
        self.telemetry = AsyncTelemetryLogger()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not os.path.exists(weights_path):
            print("[Setup] Bootstrapping PyTorch 1D-CNN model...")
            train_and_export(weights_path=weights_path, epochs=15)

        self.model = TemporalGestureCNN(num_features=63, num_classes=len(GESTURE_CLASSES)).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.tracker = UnifiedHandTracker()

        self.state = STATE_IDLE
        self.dwell_start = 0.0
        self.dwell_threshold = 0.15  # 150 ms intentional priming window
        self.pointer_filter = None
        self.landmark_window = deque(maxlen=30)
        self.last_dynamic_trigger = 0.0
        self.dynamic_cooldown = 0.8
        self.last_gesture_label = "IDLE"
        self.is_pinched = False
        self.os_control_enabled = False  # Toggle with 'M' key

    @staticmethod
    def is_intent_clutch_pose(landmarks) -> bool:
        index_extended = landmarks[8].y < landmarks[6].y
        middle_curled = landmarks[12].y > landmarks[10].y
        ring_curled = landmarks[16].y > landmarks[14].y
        pinky_curled = landmarks[20].y > landmarks[18].y
        return index_extended and middle_curled and ring_curled and pinky_curled

    @staticmethod
    def calc_distance(lm1, lm2) -> float:
        return math.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2 + (lm1.z - lm2.z) ** 2)

    def classify_dynamic_gesture(self) -> tuple[str, float]:
        if len(self.landmark_window) < 30:
            return "IDLE", 0.0
        seq_np = np.array(self.landmark_window, dtype=np.float32)
        tensor_in = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_in)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        return GESTURE_CLASSES[pred_idx.item()], conf.item()

    def execute_dynamic_os_command(self, gesture_name: str, confidence: float):
        now = time.time()
        if now - self.last_dynamic_trigger < self.dynamic_cooldown:
            return
        self.last_dynamic_trigger = now
        self.last_gesture_label = f"{gesture_name} ({confidence*100:.0f}%)"

        if self.os_control_enabled:
            if gesture_name == "SWIPE_LEFT":
                pyautogui.press("left", _pause=False)
            elif gesture_name == "SWIPE_RIGHT":
                pyautogui.press("right", _pause=False)
            elif gesture_name == "SWIPE_UP":
                pyautogui.scroll(300, _pause=False)
            elif gesture_name == "SWIPE_DOWN":
                pyautogui.scroll(-300, _pause=False)

        self.telemetry.log_event("GESTURE_EVENT", {
            "gesture_type": gesture_name,
            "confidence_val": round(confidence, 4),
            "clutch_state": "ENGAGED"
        })

    def run(self, camera_index: int = 0, target_fps: int = 30):
        # Use DirectShow on Windows to eliminate USB camera initialization & buffer lag
        if IS_WINDOWS:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)

        # Configure MJPG fourcc & 640x480 capture for rock-solid 30 FPS camera hardware readout
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_budget = 1.0 / float(target_fps)
        prev_loop_time = time.perf_counter()
        fps_history = deque(maxlen=15)
        frame_counter = 0

        print("=================================================================")
        print(f" Intent-Gated Gesture Controller (Capped at {target_fps} FPS)")
        print(" Controls: [M] Toggle OS Mouse Injection | [Q] Quit Application")
        print("=================================================================")

        while cap.isOpened():
            loop_start = time.perf_counter()
            success, raw_frame = cap.read()
            if not success:
                break

            raw_frame = cv2.flip(raw_frame, 1)
            rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            lms = self.tracker.process(rgb_frame)

            # Upscale display canvas to 960x720 so HUD text is crisp while keeping inference fast
            frame = cv2.resize(raw_frame, (960, 720), interpolation=cv2.INTER_LINEAR)
            h, w, _ = frame.shape
            now = time.time()

            hud_state = "STATE 0: IDLE (SAFETY CLUTCH DISENGAGED)"
            hud_color = (0, 0, 255)
            jitter_var = 0.0

            if lms is not None:
                self.tracker.draw_landmarks(frame, lms)

                norm_vec = normalize_landmarks(lms)
                self.landmark_window.append(norm_vec)
                intent_pose = self.is_intent_clutch_pose(lms)

                if self.state == STATE_IDLE:
                    if intent_pose:
                        self.state = STATE_PRIMED
                        self.dwell_start = now

                elif self.state == STATE_PRIMED:
                    if intent_pose:
                        dwell_elapsed = now - self.dwell_start
                        hud_state = f"STATE 1: PRIMED ({int(dwell_elapsed * 1000)} / 150 ms)"
                        hud_color = (0, 255, 255)
                        if dwell_elapsed >= self.dwell_threshold:
                            self.state = STATE_ENGAGED
                            # Tuned 1-Euro Filter: min_cutoff=1.2, beta=0.015 for snappy response
                            self.pointer_filter = Point2DOneEuroFilter(now, lms[8].x, lms[8].y,
                                                                       min_cutoff=1.2, beta=0.015)
                    else:
                        self.state = STATE_IDLE

                elif self.state in (STATE_ENGAGED, STATE_COMMAND):
                    if not intent_pose:
                        self.state = STATE_IDLE
                        if self.is_pinched and self.os_control_enabled:
                            pyautogui.mouseUp(_pause=False)
                        self.is_pinched = False
                    else:
                        raw_x, raw_y = lms[8].x, lms[8].y
                        filt_x, filt_y = self.pointer_filter.filter(now, raw_x, raw_y)
                        jitter_var = float((raw_x - filt_x) ** 2 + (raw_y - filt_y) ** 2)

                        screen_x = np.interp(filt_x, [0.15, 0.85], [0, SCREEN_W])
                        screen_y = np.interp(filt_y, [0.15, 0.85], [0, SCREEN_H])

                        if self.os_control_enabled:
                            fast_move_cursor(screen_x, screen_y)

                        pinch_dist = self.calc_distance(lms[4], lms[8])
                        if pinch_dist < 0.045:
                            self.state = STATE_COMMAND
                            if not self.is_pinched:
                                self.is_pinched = True
                                if self.os_control_enabled:
                                    pyautogui.mouseDown(_pause=False)
                                self.telemetry.log_event("GESTURE_EVENT", {
                                    "gesture_type": "PINCH_CLICK_DOWN",
                                    "confidence_val": 1.0,
                                    "clutch_state": "COMMAND_EXECUTE"
                                })
                        else:
                            if self.is_pinched:
                                if self.os_control_enabled:
                                    pyautogui.mouseUp(_pause=False)
                                self.is_pinched = False
                            self.state = STATE_ENGAGED

                        # Run PyTorch classifier every 3rd frame to keep latency < 5ms
                        if frame_counter % 3 == 0:
                            dyn_gesture, dyn_conf = self.classify_dynamic_gesture()
                            if dyn_gesture != "IDLE" and dyn_conf >= 0.90:
                                self.execute_dynamic_os_command(dyn_gesture, dyn_conf)

                        rx, ry = int(raw_x * w), int(raw_y * h)
                        fx, fy = int(filt_x * w), int(filt_y * h)
                        cv2.circle(frame, (rx, ry), 6, (0, 0, 255), 2)
                        cv2.circle(frame, (fx, fy), 12, (255, 0, 255) if self.is_pinched else (0, 255, 0), cv2.FILLED)

                        hud_state = "STATE 3: COMMAND (PINCH)" if self.is_pinched else "STATE 2: ENGAGED (ACTIVE CURSOR)"
                        hud_color = (255, 0, 255) if self.is_pinched else (0, 255, 0)

            else:
                self.state = STATE_IDLE
                self.landmark_window.clear()
                if self.is_pinched and self.os_control_enabled:
                    pyautogui.mouseUp(_pause=False)
                self.is_pinched = False

            # Processing latency (before FPS limiter sleep)
            proc_latency_ms = (time.perf_counter() - loop_start) * 1000.0

            # Enforce strict 30 FPS cap
            elapsed = time.perf_counter() - loop_start
            if elapsed < frame_budget:
                time.sleep(frame_budget - elapsed)

            now_perf = time.perf_counter()
            inst_fps = min(float(target_fps), 1.0 / max(now_perf - prev_loop_time, 1e-6))
            prev_loop_time = now_perf
            fps_history.append(inst_fps)
            avg_fps = sum(fps_history) / len(fps_history)
            frame_counter += 1

            if frame_counter % 30 == 0:
                self.telemetry.log_event("TRACKING_METRIC", {
                    "fps_measured": round(avg_fps, 2),
                    "tracking_latency_ms": round(proc_latency_ms, 2),
                    "jitter_variance": round(jitter_var, 6),
                    "dropout_flag": lms is None
                })

            cv2.rectangle(frame, (10, 10), (740, 175), (20, 20, 20), cv2.FILLED)
            cv2.putText(frame, f"FPS: {int(round(avg_fps))} (Cap 30) | Latency: {proc_latency_ms:.1f} ms | Device: {str(self.device).upper()}",
                        (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(frame, hud_state, (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
            cv2.putText(frame, f"PyTorch Dynamic Class: {self.last_gesture_label}",
                        (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
            os_status = "ON (Press 'M' to pause)" if self.os_control_enabled else "PAUSED (Press 'M' to enable)"
            cv2.putText(frame, f"OS Mouse Injection: {os_status}",
                        (25, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

            cv2.imshow("Intent-Gated Contactless Gesture Controller (EC3319)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                self.os_control_enabled = not self.os_control_enabled

        cap.release()
        cv2.destroyAllWindows()
        self.telemetry.shutdown()


if __name__ == "__main__":
    controller = IntentGatedGestureController()
    # Set camera_index=0 for built-in webcam, or camera_index=1 for USB Phone Camera
    controller.run(camera_index=0, target_fps=30)