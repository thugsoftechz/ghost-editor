
import os
import multiprocessing
import psutil

class HardwareManager:
    """
    Manages hardware resources to optimize performance.
    Detects CPU cores and available memory.
    """
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.memory_total = psutil.virtual_memory().total
        self.memory_available = psutil.virtual_memory().available

        # Heuristic optimization settings
        self.max_threads = max(1, self.cpu_count - 1)
        self.ffmpeg_threads = self.max_threads

    def get_status(self):
        return {
            "cpu_cores": self.cpu_count,
            "memory_total_gb": round(self.memory_total / (1024**3), 2),
            "memory_free_gb": round(self.memory_available / (1024**3), 2),
            "recommended_threads": self.max_threads
        }

    def log_status(self):
        status = self.get_status()
        print(f"⚡ HARDWARE OPTIMIZATION ACTIVE")
        print(f"   • Cores: {status['cpu_cores']}")
        print(f"   • RAM: {status['memory_free_gb']}GB free / {status['memory_total_gb']}GB total")
        print(f"   • Threads allocated: {status['recommended_threads']}")
