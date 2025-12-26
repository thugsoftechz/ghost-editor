
import json
import os
import datetime

class MetadataEngine:
    def generate(self, main_path, output_dir):
        """
        Generates upload-ready metadata.
        """
        filename = os.path.basename(main_path)
        base_name = os.path.splitext(filename)[0].replace("_", " ").title()

        today = datetime.date.today().strftime("%Y-%m-%d")

        metadata = {
            "title": f"{base_name} - Official Video ({today})",
            "description": f"Automatically produced by Ghost AutoVid World.\n\nProcessed: {today}\nSource: {filename}",
            "tags": ["AI Video", "Automated Editing", "Tech", "Viral"],
            "category": "Science & Technology",
            "privacy": "Private"
        }

        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        with open(os.path.join(output_dir, "upload_ready.txt"), "w") as f:
            f.write(f"Title: {metadata['title']}\n")
            f.write(f"Tags: {', '.join(metadata['tags'])}\n")

        return metadata
