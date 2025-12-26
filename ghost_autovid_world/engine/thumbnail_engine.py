
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class ThumbnailEngine:
    def generate(self, video_path, output_path, title_text):
        """
        Extracts best frame and adds text.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Simple heuristic: Middle of video often good
            # Better: Analyze brightness/contrast

            best_frame = None
            max_score = -1

            # Check 10 candidate frames
            for i in range(10):
                fid = int(frame_count * (0.1 + 0.8 * (i/10)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
                ret, frame = cap.read()
                if not ret: continue

                # Score: Brightness + Variance (Sharpness/Detail)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

                score = brightness + (sharpness * 0.5)
                if score > max_score:
                    max_score = score
                    best_frame = frame

            cap.release()

            if best_frame is not None:
                # Add Text
                # Convert to PIL
                img_pil = Image.fromarray(cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)

                # Dynamic font size
                w, h = img_pil.size
                fontsize = int(h / 8)

                # Default font (system dependent, fallbacks)
                try:
                    font = ImageFont.truetype("arial.ttf", fontsize)
                except:
                    font = ImageFont.load_default()

                # Text Position (Bottom Center)
                text = title_text.upper()

                # Shadow/Stroke simulation
                x = w/2
                y = h * 0.8

                # Simple centered text logic requires text size calculation which depends on PIL version
                # Simplified: Just draw

                draw.text((x-2, y-2), text, font=font, fill="black", anchor="mm")
                draw.text((x+2, y+2), text, font=font, fill="black", anchor="mm")
                draw.text((x, y), text, font=font, fill="yellow", anchor="mm")

                img_pil.save(output_path)
                return True

        except Exception as e:
            print(f"Thumbnail Error: {e}")
        return False
