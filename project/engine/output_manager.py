
import os
import cv2
import json
import shutil
import warnings

# Suppress potential whisper warnings
warnings.filterwarnings("ignore")

class OutputManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_thumbnail(self, video_path):
        """
        Extracts a middle frame as thumbnail.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            if ret:
                thumb_path = os.path.join(self.output_dir, "thumbnail.jpg")
                cv2.imwrite(thumb_path, frame)
                print("🖼️ Thumbnail generated.")
            cap.release()
        except Exception as e:
            print(f"⚠️ Thumbnail generation failed: {e}")

    def generate_captions(self, video_path):
        """
        Uses faster-whisper to generate SRT.
        """
        try:
            from faster_whisper import WhisperModel

            # Check if model exists or network allowed? Prompt says NO internet.
            # faster-whisper downloads model on first run.
            # If not present, this will fail.
            # We assume environment is prepped (God Mode implied it).
            # We wrap in try-except.

            print("📝 Generating Captions (Whisper)...")
            model_size = "tiny" # Fastest, smallest
            # Run on CPU INT8
            model = WhisperModel(model_size, device="cpu", compute_type="int8")

            segments, info = model.transcribe(video_path, beam_size=5)

            srt_path = os.path.join(self.output_dir, "captions.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(segments):
                    # Format time
                    start = self._fmt_time(segment.start)
                    end = self._fmt_time(segment.end)
                    text = segment.text.strip()

                    f.write(f"{i+1}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{text}\n\n")

            print("✅ Captions saved.")

        except Exception as e:
            print(f"⚠️ Caption generation failed (Model missing?): {e}")
            # Write empty srt to satisfy requirement?
            # Or just skip. Prompt: "captions.srt (if speech exists)"

    def _fmt_time(self, seconds):
        """Convert seconds to SRT time format."""
        import datetime
        dt = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=seconds)
        return dt.strftime("%H:%M:%S,%f")[:-3]

    def save_metadata(self, meta_dict):
        path = os.path.join(self.output_dir, "metadata.json")
        with open(path, "w") as f:
            json.dump(meta_dict, f, indent=2)
        print("📄 Metadata saved.")

    def create_upload_ready(self):
        path = os.path.join(self.output_dir, "upload_ready.txt")
        with open(path, "w") as f:
            f.write("VIDEO READY FOR UPLOAD\n")
            f.write("----------------------\n")
            f.write("1. Check thumbnail.jpg\n")
            f.write("2. Upload final_video.mp4\n")
            f.write("3. Add captions.srt\n")
            f.write("4. Copy description from metadata.json\n")
