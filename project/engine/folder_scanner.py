
import os

class FolderScanner:
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}

    def scan(self, folder_path):
        """
        Recursively scans for valid video files.
        Returns a list of absolute file paths.
        """
        video_files = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        print(f"📂 Scanning: {folder_path}")

        for root, dirs, files in os.walk(folder_path):
            # Ignore output folder if scanning recursively in project root (safety)
            if 'output' in dirs:
                dirs.remove('output')

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in self.VIDEO_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    # Basic check if file is valid/readable size
                    if os.path.getsize(full_path) > 1024:
                        video_files.append(full_path)

        print(f"✅ Found {len(video_files)} video files.")
        return video_files
