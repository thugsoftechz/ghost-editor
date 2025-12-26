
import os
import cv2
import numpy as np
import mediapipe as mp
# Robust MoviePy import
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

class Analyzer:
    def __init__(self):
        # Initialize Mediapipe
        try:
            import mediapipe.python.solutions.face_detection as face_detection
            self.mp_face = face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except:
            try:
                self.mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            except:
                self.mp_face = None

    def analyze(self, video_path):
        """
        Deep analysis of video content: Motion, Face, Audio, Quality.
        """
        metrics = {
            "path": video_path,
            "duration": 0,
            "resolution": (0, 0),
            "fps": 0,
            "has_audio": False,
            "avg_volume": 0,
            "motion_score": 0,
            "face_score": 0,  # 0 to 1 (frequency of faces)
            "face_locations": [], # List of (t, bbox)
            "is_corrupted": False
        }

        try:
            # 1. MoviePy Metadata & Audio
            clip = VideoFileClip(video_path)
            metrics["duration"] = clip.duration
            metrics["resolution"] = clip.size
            metrics["fps"] = clip.fps
            metrics["has_audio"] = clip.audio is not None

            if metrics["has_audio"]:
                # Sample audio (first 60s)
                dur = min(clip.duration, 60)
                try:
                    # Compatibility shim
                    if hasattr(clip.audio, 'subclipped'):
                        chunk = clip.audio.subclipped(0, dur)
                    else:
                        chunk = clip.audio.subclip(0, dur)

                    sarray = chunk.to_soundarray(fps=22050)
                    if sarray is not None:
                        metrics["avg_volume"] = float(np.sqrt(np.mean(sarray**2)))
                except:
                    pass

            clip.close()

            # 2. OpenCV Visual Analysis (Motion & Faces)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                metrics["is_corrupted"] = True
                return metrics

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 24

            # Analyze every 1 second
            step = int(fps)

            prev_gray = None
            total_motion = 0
            samples = 0
            faces_found = 0
            face_locs = []

            for i in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret: break

                # Resize for speed
                h, w = frame.shape[:2]
                scale = 360 / h
                small = cv2.resize(frame, (int(w*scale), 360))

                # Motion
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    score = np.mean(diff)
                    total_motion += score
                prev_gray = gray

                # Face
                if self.mp_face:
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    results = self.mp_face.process(rgb)
                    if results.detections:
                        faces_found += 1
                        # Store bbox for later auto-crop logic
                        # bbox relative [0,1]
                        bbox = results.detections[0].location_data.relative_bounding_box
                        t = i / fps
                        face_locs.append((t, bbox))

                samples += 1

            cap.release()

            if samples > 0:
                metrics["motion_score"] = total_motion / samples
                metrics["face_score"] = faces_found / samples
                metrics["face_locations"] = face_locs

        except Exception as e:
            print(f"⚠️ Analysis Error {video_path}: {e}")
            metrics["is_corrupted"] = True

        return metrics
