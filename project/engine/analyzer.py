
import os
import cv2
import numpy as np
import librosa
import mediapipe as mp
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

class Analyzer:
    def __init__(self):
        try:
            import mediapipe.python.solutions.face_detection as face_detection
            self.mp_face_detection = face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        except ImportError:
            # Fallback if specific module path fails or mediapipe structure differs
            try:
                self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
            except AttributeError:
                 self.mp_face_detection = None

    def analyze(self, video_path):
        """
        Analyzes video for metadata, motion, audio levels, and face presence.
        """
        print(f"🔍 Analyzing: {os.path.basename(video_path)}")

        metrics = {
            "path": video_path,
            "duration": 0,
            "has_audio": False,
            "avg_volume": 0,
            "motion_score": 0,
            "face_count": 0,
            "resolution": (0, 0),
            "is_corrupted": False
        }

        try:
            # 1. MoviePy / FFmpeg Metadata
            clip = VideoFileClip(video_path)
            metrics["duration"] = clip.duration
            metrics["resolution"] = clip.size
            metrics["has_audio"] = clip.audio is not None

            # 2. Audio Analysis (Volume / Speech Density Proxy)
            if metrics["has_audio"]:
                # Analyze first 30s max to save time
                chunk_duration = min(clip.duration, 30)
                try:
                    # chunk audio
                    # v2 audio clip subclipped or subclip?
                    # v1: subclip(0, t)
                    # v2: subclipped(0, t)
                    if hasattr(clip.audio, 'subclipped'):
                        audio_chunk = clip.audio.subclipped(0, chunk_duration)
                    else:
                        audio_chunk = clip.audio.subclip(0, chunk_duration)

                    # to numpy
                    sarray = audio_chunk.to_soundarray(fps=22050)

                    if sarray is not None:
                        # RMS
                        rms = np.sqrt(np.mean(sarray**2))
                        metrics["avg_volume"] = float(rms)
                except Exception as e:
                    # fallback to full audio analysis if subclip fails or try accessing directly
                    try:
                         sarray = clip.audio.to_soundarray(fps=22050, nbytes=2, buffersize=1000)
                         if sarray is not None:
                            rms = np.sqrt(np.mean(sarray**2))
                            metrics["avg_volume"] = float(rms)
                    except Exception as e2:
                        print(f"⚠️ Audio analysis warning: {e}, {e2}")

            clip.close()

            # 3. OpenCV Analysis (Motion & Face)
            # Sample frames: 1 frame every 2 seconds
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                metrics["is_corrupted"] = True
                return metrics

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 24

            sample_interval = int(fps * 2) # Every 2 sec

            prev_gray = None
            total_motion = 0
            motion_samples = 0
            faces_found = 0

            for i in range(0, frame_count, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret: break

                # Face Detection
                if self.mp_face_detection:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.mp_face_detection.process(rgb_frame)
                    if results.detections:
                        faces_found += 1

                # Motion (Frame Diff)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 240)) # Low res for speed

                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    score = np.mean(diff)
                    total_motion += score
                    motion_samples += 1

                prev_gray = gray

            cap.release()

            metrics["motion_score"] = (total_motion / motion_samples) if motion_samples > 0 else 0
            metrics["face_presence_ratio"] = faces_found / (motion_samples + 1) # Ratio of frames with face

        except Exception as e:
            print(f"❌ Corrupted video: {video_path} ({e})")
            metrics["is_corrupted"] = True

        return metrics
