
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
from ghost_autovid_world.engine.audio_engine import AudioEngine
from ghost_autovid_world.engine.ai_engine import AIEngine

def render_dashboard():
    st.set_page_config(page_title="GHOST_AUTOVID_WORLD", layout="wide", page_icon="👻")

    # CSS
    st.markdown("""
    <style>
        .stApp { background-color: #050505; color: #EEE; }
        .stButton>button { width: 100%; background-color: #333; color: white; border: 1px solid #555; }
        .stButton>button:hover { border-color: #0F0; color: #0F0; }
        h1 { color: #0F0; font-family: 'Courier New', monospace; }
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #111; border-radius: 4px; color: #888; }
        .stTabs [aria-selected="true"] { background-color: #222; color: #0F0; border-bottom-color: #0F0; }
    </style>
    """, unsafe_allow_html=True)

    st.title("👻 GHOST_AUTOVID_WORLD")
    st.caption("Autonomous Video Production System | v2.1 | AI Enhanced")

    # Sidebar: Hardware
    hw = HardwareManager()
    status = hw.get_status()

    with st.sidebar:
        st.header("SYSTEM STATUS")
        st.metric("CPU Cores", status["cpu_cores"])
        st.metric("RAM Free (GB)", status["memory_free_gb"])
        st.progress(status["memory_free_gb"] / status["memory_total_gb"])

        st.divider()
        st.header("GPU ACCELERATION")
        if status["gpu_vendor"] != "CPU":
            st.success(f"ACTIVE: {status['gpu_vendor']}")
            st.caption(f"Encoder: {status['encoder']}")
        else:
            st.warning("ACTIVE: CPU ONLY")
            st.caption("No supported GPU found")

        st.divider()
        st.write("ENGINE: READY")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Auto-Pilot", "🎛️ Advanced Editor", "🔊 Audio Lab", "📸 Photo Studio", "🤖 AI Tools"])

    # --- TAB 1: AUTO-PILOT ---
    with tab1:
        st.header("One-Click Production")
        input_path = st.text_input("RAW FOOTAGE PATH", placeholder="/path/to/folder", key="autopilot_input")

        if st.button("INITIATE GHOST PROTOCOL"):
            if not input_path or not os.path.exists(input_path):
                st.error("INVALID PATH")
            else:
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
                else:
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
                    else:
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
                        else:
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

    # --- TAB 2: ADVANCED EDITOR ---
    with tab2:
        st.header("Advanced Video Editor")

        col1, col2 = st.columns([1, 1])
        with col1:
            adv_input_path = st.text_input("RAW FOOTAGE PATH", placeholder="/path/to/folder", key="adv_input")
        with col2:
            if st.button("SCAN & ANALYZE"):
                if not adv_input_path or not os.path.exists(adv_input_path):
                    st.error("INVALID PATH")
                else:
                    with st.spinner("Analyzing..."):
                        scanner = FolderScanner()
                        files = scanner.scan(adv_input_path)
                        analyzer = Analyzer()
                        data = []
                        prog = st.progress(0)
                        for i, f in enumerate(files):
                            data.append(analyzer.analyze(f))
                            prog.progress((i+1)/len(files))
                        st.session_state.adv_data = data
                        st.success(f"Analyzed {len(data)} files")

        if 'adv_data' in st.session_state and st.session_state.adv_data:
            st.divider()
            st.subheader("Selection")

            data = st.session_state.adv_data
            paths = [d['path'] for d in data]
            names = [os.path.basename(p) for p in paths]

            # Use Selector for default suggestion
            if 'adv_defaults_set' not in st.session_state:
                selector = Selector()
                main, broll, _ = selector.select(data)
                st.session_state.main_idx = paths.index(main['path']) if main else 0
                st.session_state.broll_idxs = [paths.index(b['path']) for b in broll]
                st.session_state.adv_defaults_set = True

            main_idx = st.selectbox("Select Main Video", range(len(names)), format_func=lambda x: names[x], index=st.session_state.main_idx)
            broll_idxs = st.multiselect("Select B-Roll", range(len(names)), format_func=lambda x: names[x], default=st.session_state.broll_idxs)

            st.divider()
            st.subheader("Parameters")
            c1, c2, c3 = st.columns(3)
            with c1:
                silence_thresh = st.slider("Silence Sensitivity", 0.01, 0.2, 0.05, 0.01)
            with c2:
                zoom_factor = st.slider("Zoom Factor", 1.0, 2.0, 1.2, 0.1)
            with c3:
                broll_int = st.number_input("B-Roll Interval (cuts)", 1, 10, 3)

            if st.button("RENDER ADVANCED VIDEO"):
                main_meta = data[main_idx]
                broll_metas = [data[i] for i in broll_idxs]

                output_dir = os.path.join(os.getcwd(), "ghost_autovid_world/output")
                os.makedirs(output_dir, exist_ok=True)
                final_path = os.path.join(output_dir, "advanced_render.mp4")

                status_box = st.empty()
                def adv_log(msg): status_box.info(msg)

                editor = AutoEditor()
                success = editor.process(
                    main_meta, broll_metas, final_path,
                    status_callback=adv_log,
                    silence_threshold=silence_thresh,
                    zoom_factor=zoom_factor,
                    b_roll_interval=broll_int
                )

                if success:
                    st.success("Render Complete!")
                    st.video(final_path)
                else:
                    st.error("Render Failed")

    # --- TAB 3: AUDIO LAB ---
    with tab3:
        st.header("Audio Studio")
        audio_path = st.text_input("INPUT FILE (Video/Audio)", placeholder="/path/to/video.mp4", key="audio_input")
        target_lufs = st.slider("Target LUFS", -30.0, -5.0, -14.0, 1.0)

        if st.button("NORMALIZE AUDIO"):
            if not audio_path or not os.path.exists(audio_path):
                st.error("File not found")
            else:
                with st.spinner("Processing Audio..."):
                    engine = AudioEngine()
                    out_dir = os.path.join(os.getcwd(), "ghost_autovid_world/output")
                    os.makedirs(out_dir, exist_ok=True)
                    fname = os.path.basename(audio_path)
                    name, ext = os.path.splitext(fname)
                    out_path = os.path.join(out_dir, f"{name}_norm{ext}")

                    if engine.process_file(audio_path, out_path, target_lufs):
                        st.success(f"Saved to: {out_path}")
                        if ext.lower() in ['.mp4', '.mov', '.mkv']:
                            st.video(out_path)
                        else:
                            st.audio(out_path)
                    else:
                        st.error("Processing failed")

    # --- TAB 4: PHOTO STUDIO ---
    with tab4:
        st.header("Photo / Thumbnail Lab")
        vid_path = st.text_input("INPUT VIDEO", placeholder="/path/to/video.mp4", key="photo_input")
        title_text = st.text_input("OVERLAY TEXT", "EPIC VIDEO")

        if vid_path and os.path.exists(vid_path):
            import cv2
            cap = cv2.VideoCapture(vid_path)
            duration = 0
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps > 0: duration = count / fps
            cap.release()

            time_pos = st.slider("Frame Timestamp (seconds)", 0.0, duration, duration/2)

            if st.button("GENERATE THUMBNAIL"):
                out_dir = os.path.join(os.getcwd(), "ghost_autovid_world/output")
                os.makedirs(out_dir, exist_ok=True)
                thumb_out = os.path.join(out_dir, "custom_thumb.jpg")

                eng = ThumbnailEngine()
                if eng.generate(vid_path, thumb_out, title_text, time_pos=time_pos):
                    st.success("Thumbnail Generated")
                    st.image(thumb_out)
                else:
                    st.error("Generation Failed")

    # --- TAB 5: AI TOOLS ---
    with tab5:
        st.header("AI Intelligence Center")
        ai_input = st.text_input("INPUT MEDIA PATH", placeholder="/path/to/video_or_audio.mp4", key="ai_input")
        model_size = st.selectbox("Model Size", ["tiny", "base", "small", "medium", "large-v3"], index=0)

        if st.button("RUN TRANSCRIPTION"):
            if not ai_input or not os.path.exists(ai_input):
                st.error("Invalid Path")
            else:
                with st.spinner("Initializing AI Model..."):
                    engine = AIEngine(model_size=model_size)

                with st.spinner("Transcribing... (This may take a while)"):
                    segments, info = engine.transcribe(ai_input)

                    if segments is None:
                        st.error("Transcription Failed. Check logs.")
                    else:
                        st.success(f"Transcription Complete! Detected Language: {info.language} ({info.language_probability:.2f})")

                        # Generate SRT
                        out_dir = os.path.join(os.getcwd(), "ghost_autovid_world/output")
                        os.makedirs(out_dir, exist_ok=True)
                        base_name = os.path.basename(ai_input).rsplit('.', 1)[0]
                        srt_path = os.path.join(out_dir, f"{base_name}.srt")

                        if engine.generate_srt(segments, srt_path):
                            st.success(f"SRT Saved: {srt_path}")

                            # Display text preview
                            st.subheader("Transcript Preview")
                            full_text = "\n".join([s.text for s in segments])
                            st.text_area("Content", full_text, height=300)

                            # Download button logic (Streamlit specific)
                            with open(srt_path, "r") as f:
                                st.download_button("Download SRT", f, file_name=f"{base_name}.srt")
                        else:
                            st.error("Failed to save SRT")
