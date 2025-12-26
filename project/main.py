
import os
import argparse
import sys
import shutil

# Add project root to path
sys.path.append(os.getcwd())

from project.engine.folder_scanner import FolderScanner
from project.engine.analyzer import Analyzer
from project.engine.selector import Selector
from project.engine.auto_editor import AutoEditor
from project.engine.output_manager import OutputManager

def main():
    parser = argparse.ArgumentParser(description="Jules: Automatic Video Decision System")
    parser.add_argument("folder", help="Input folder containing raw footage")
    args = parser.parse_args()

    input_folder = os.path.abspath(args.folder)
    output_base = os.path.join(os.getcwd(), "project", "output")

    # 1. Setup Output
    # Create a unique output folder for this run?
    # Or just overwrite "project/output" as per structure requirement.
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    output_mgr = OutputManager(output_base)

    print("🚀 Jules System Initialized")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    try:
        # 2. Scan
        scanner = FolderScanner()
        video_files = scanner.scan(input_folder)

        if not video_files:
            print("❌ No videos found. Exiting.")
            return

        # 3. Analyze
        analyzer = Analyzer()
        results = []
        for v in video_files:
            meta = analyzer.analyze(v)
            results.append(meta)

        # 4. Select
        selector = Selector()
        main_video, b_roll, junk = selector.select(results)

        if not main_video:
            print("❌ No suitable main video found. Exiting.")
            return

        # 5. Edit
        editor = AutoEditor()
        final_video_path = os.path.join(output_base, "final_video.mp4")

        # Pass metadata to editor
        editor.edit(main_video, b_roll, final_video_path)

        # 6. Final Output Generation
        output_mgr.generate_thumbnail(final_video_path)

        if main_video["has_audio"]:
            output_mgr.generate_captions(final_video_path)

        # Save Metadata
        meta = {
            "source_folder": input_folder,
            "main_video": main_video["path"],
            "b_roll_count": len(b_roll),
            "junk_count": len(junk),
            "output_file": final_video_path
        }
        output_mgr.save_metadata(meta)
        output_mgr.create_upload_ready()

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✅ MISSION COMPLETE")
        print(f"📂 Output: {output_base}")

    except Exception as e:
        print(f"💥 CRITICAL FAILURE: {e}")
        # Graceful exit implies log and quit, not crash stacktrace if possible
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
