
import streamlit as st
import os
import time
from ghost_autovid_world.engine.folder_scanner import FolderScanner
from ghost_autovid_world.engine.analyzer import Analyzer
from ghost_autovid_world.engine.selector import Selector
from ghost_autovid_world.engine.auto_editor import AutoEditor
from ghost_autovid_world.engine.thumbnail_engine import ThumbnailEngine
from ghost_autovid_world.engine.metadata_engine import MetadataEngine
from ghost_autovid_world.engine.hardware_manager import HardwareManager

def render_dashboard():
    st.set_page_config(page_title="GHOST_AUTOVID_WORLD", layout="wide", page_icon="👻")

    # CSS
    st.markdown("""
    <style>
        .stApp { background-color: #050505; color: #EEE; }
        .stButton>button { width: 100%; background-color: #333; color: white; border: 1px solid #555; }
        .stButton>button:hover { border-color: #0F0; color: #0F0; }
        h1 { color: #0F0; font-family: 'Courier New', monospace; }
    </style>
    """, unsafe_allow_html=True)

    st.title("👻 GHOST_AUTOVID_WORLD")
    st.caption("Autonomous Video Production System | v1.0 | Offline Mode")

    # Sidebar: Hardware
    hw = HardwareManager()
    status = hw.get_status()

    with st.sidebar:
        st.header("SYSTEM STATUS")
        st.metric("CPU Cores", status["cpu_cores"])
        st.metric("RAM Free (GB)", status["memory_free_gb"])
        st.progress(status["memory_free_gb"] / status["memory_total_gb"])
        st.divider()
        st.write("ENGINE: READY")

    # Main
    input_path = st.text_input("RAW FOOTAGE PATH", placeholder="/path/to/folder")

    if st.button("INITIATE GHOST PROTOCOL"):
        if not input_path or not os.path.exists(input_path):
            st.error("INVALID PATH")
            return

        logs = st.empty()
        progress = st.progress(0)

        def log(msg):
            logs.code(f"[{time.strftime('%H:%M:%S')}] {msg}")
            print(msg)

        log("🚀 SYSTEM STARTING...")

        # 1. Scan
        scanner = FolderScanner()
        files = scanner.scan(input_path)
        log(f"📂 FOUND {len(files)} FILES")
        progress.progress(10)

        if not files:
            st.error("NO FOOTAGE DETECTED")
            return

        # 2. Analyze
        log("🧠 ANALYZING CONTENT MATRIX...")
        analyzer = Analyzer()
        analysis_data = []
        for i, f in enumerate(files):
            meta = analyzer.analyze(f)
            analysis_data.append(meta)
            progress.progress(10 + int(30 * (i/len(files))))

        # 3. Select
        selector = Selector()
        main, broll, discard = selector.select(analysis_data)

        if not main:
            st.error("NO MAIN CONTENT IDENTIFIED")
            return

        log(f"🎯 DECISION: Main={os.path.basename(main['path'])}")
        log(f"   B-Roll={len(broll)} | Discard={len(discard)}")
        progress.progress(50)

        # 4. Edit
        log("🎬 ENGAGING AUTO-EDITOR...")
        editor = AutoEditor()
        output_dir = os.path.join(os.getcwd(), "ghost_autovid_world/output")
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, "final_render.mp4")

        success = editor.process(main, broll, final_path, status_callback=log)
        if not success:
            st.error("RENDER FAILED")
            return
        progress.progress(80)

        # 5. Post-Process
        log("📦 GENERATING ASSETS...")
        thumb = ThumbnailEngine()
        thumb_path = os.path.join(output_dir, "thumbnail.jpg")
        title = os.path.basename(main['path']).split('.')[0]
        thumb.generate(final_path, thumb_path, title)

        meta = MetadataEngine()
        meta.generate(final_path, output_dir)

        progress.progress(100)
        log("✅ GHOST PROTOCOL COMPLETE")
        st.success("VIDEO PRODUCED SUCCESSFULLY")
        st.video(final_path)
        st.image(thumb_path, caption="Auto-Generated Thumbnail")
