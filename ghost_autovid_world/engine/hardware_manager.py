
import os
import multiprocessing
import psutil
import subprocess
import shutil

class HardwareManager:
    """
    Manages hardware resources to optimize performance.
    Detects CPU cores, available memory, and GPU acceleration (NVIDIA/Intel).
    """
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.memory_available = psutil.virtual_memory().available

        # Heuristic optimization settings
        self.max_threads = max(1, self.cpu_count - 1)
        self.ffmpeg_threads = self.max_threads

        # GPU Detection
        self.gpu_vendor = "CPU"
        self.encoder = "libx264"
        self.preset = "medium"
        self._detect_hardware_encoder()

    def _get_ffmpeg_path(self):
        path = shutil.which("ffmpeg")
        if path:
            return path
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            return None

    def _detect_hardware_encoder(self):
        """
        Detects available FFmpeg hardware encoders.
        """
        ffmpeg_bin = self._get_ffmpeg_path()
        if not ffmpeg_bin:
            print("⚠️ FFmpeg not found.")
            return

        try:
            # Run ffmpeg -encoders
            # Adding check=True to raise exception on failure
            result = subprocess.run(
                [ffmpeg_bin, "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            output = result.stdout

            # Priority 1: NVIDIA NVENC
            if "h264_nvenc" in output:
                self.gpu_vendor = "NVIDIA (NVENC)"
                self.encoder = "h264_nvenc"
                self.preset = "p4" # Medium-High performance preset for NVENC
                print("🚀 NVIDIA Hardware Acceleration Detected")

            # Priority 2: Intel QSV
            elif "h264_qsv" in output:
                self.gpu_vendor = "INTEL (QSV)"
                self.encoder = "h264_qsv"
                self.preset = "medium"
                print("🚀 INTEL Hardware Acceleration Detected")

            # Priority 3: Apple VideoToolbox (Bonus)
            elif "h264_videotoolbox" in output:
                self.gpu_vendor = "APPLE (M-Chip)"
                self.encoder = "h264_videotoolbox"
                self.preset = "default"
                print("🚀 APPLE Hardware Acceleration Detected")

            else:
                print("ℹ️ No Hardware Encoder detected. Using CPU.")

        except Exception as e:
            print(f"Hardware Detection Error: {e}")

    def get_status(self):
        return {
            "cpu_cores": self.cpu_count,
            "memory_total_gb": round(self.memory_total / (1024**3), 2),
            "memory_free_gb": round(self.memory_available / (1024**3), 2),
            "recommended_threads": self.max_threads,
            "gpu_vendor": self.gpu_vendor,
            "encoder": self.encoder
        }

    def get_render_settings(self):
        """
        Returns dictionary of keyword arguments for moviepy write_videofile
        """
        settings = {
            "codec": self.encoder,
            "threads": self.ffmpeg_threads,
            "preset": self.preset
        }

        # Adjust arguments based on encoder
        if self.encoder == "h264_nvenc":
            # NVENC specific
            settings["threads"] = None
            # moviepy default ffmpeg params might need tweaking for nvenc but usually defaults work
        elif self.encoder == "h264_qsv":
             settings["threads"] = None
        elif self.encoder == "h264_videotoolbox":
             settings["threads"] = None

        return settings

    def log_status(self):
        status = self.get_status()
        print(f"⚡ HARDWARE OPTIMIZATION ACTIVE")
        print(f"   • Cores: {status['cpu_cores']}")
        print(f"   • RAM: {status['memory_free_gb']}GB free / {status['memory_total_gb']}GB total")
        print(f"   • GPU: {status['gpu_vendor']} (Encoder: {status['encoder']})")
