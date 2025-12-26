
import argparse
import sys
import os
import time

# Robust path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ghost_autovid_world.engine.folder_scanner import FolderScanner
from ghost_autovid_world.engine.analyzer import Analyzer
from ghost_autovid_world.engine.selector import Selector
from ghost_autovid_world.engine.auto_editor import AutoEditor
from ghost_autovid_world.engine.thumbnail_engine import ThumbnailEngine
from ghost_autovid_world.engine.metadata_engine import MetadataEngine
from ghost_autovid_world.engine.hardware_manager import HardwareManager

def main():
    parser = argparse.ArgumentParser(description="GHOST_AUTOVID_WORLD: Headless Mode")
    parser.add_argument("folder", help="Input Folder Path")
    args = parser.parse_args()

    input_path = os.path.abspath(args.folder)

    print("\n👻 GHOST_AUTOVID_WORLD | HEADLESS MODE")
    print("======================================")

    hw = HardwareManager()
    hw.log_status()

    # 1. Scan
    scanner = FolderScanner()
    files = scanner.scan(input_path)
    if not files:
        print("❌ No files found.")
        return

    # 2. Analyze
    print(f"\n🧠 Analyzing {len(files)} files...")
    analyzer = Analyzer()
    data = []
    for f in files:
        data.append(analyzer.analyze(f))

    # 3. Select
    selector = Selector()
    main, broll, discard = selector.select(data)
    if not main:
        print("❌ No main content.")
        return

    print(f"🎯 Selected Main: {os.path.basename(main['path'])}")

    # 4. Edit
    print("\n🎬 Editing...")
    editor = AutoEditor()
    output_dir = os.path.join(os.getcwd(), "ghost_autovid_world", "output")
    os.makedirs(output_dir, exist_ok=True)
    final_path = os.path.join(output_dir, "final_render.mp4")

    # Pass metadata to process
    if editor.process(main, broll, final_path, status_callback=print):
        # 5. Assets
        print("\n📦 Assets...")
        thumb = ThumbnailEngine()
        thumb.generate(final_path, os.path.join(output_dir, "thumbnail.jpg"), "NEW VIDEO")

        meta = MetadataEngine()
        meta.generate(final_path, output_dir)

        print("\n✅ DONE.")
        print(f"Output: {output_dir}")
    else:
        print("❌ Failed.")

if __name__ == "__main__":
    main()
