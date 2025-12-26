
import os

class FolderScanner:
    VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}

    def scan(self, folder_path):
        """
        Recursively scans for valid video files.
        Ignores output directories to prevent infinite loops.
        """
        video_files = []
        folder_path = os.path.abspath(folder_path)

        if not os.path.exists(folder_path):
            return []

        print(f"📂 SCANNING: {folder_path}")

        for root, dirs, files in os.walk(folder_path):
            # Ignore output folders
            if 'output' in dirs:
                dirs.remove('output')

            # Avoid scanning the app's own source code if user points to parent dir
            # Allow scanning unless explicitly inside ghost_autovid_world/engine
            # If user provides a path like /app/test_footage, it should work.
            # Only ignore if root ends with ghost_autovid_world/engine
            if root.endswith(os.path.join('ghost_autovid_world', 'engine')):
                continue

            for file in files:
                if file.startswith("._"): continue # Mac artifacts

                ext = os.path.splitext(file)[1].lower()
                if ext in self.VIDEO_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    # Basic validity check (size > 10KB to allow test files)
                    try:
                        if os.path.getsize(full_path) > 10240:
                            video_files.append(full_path)
                    except:
                        pass

        return video_files
