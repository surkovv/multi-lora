import torch
import os
import time
import threading
import subprocess
from contextlib import contextmanager
from pynvml import *
from time import sleep
from dataclasses import dataclass
import gc

@dataclass
class Measurement:
    allocated_start: int = None
    reserved_start: int = None
    allocated_finish: int = None
    reserved_finish: int = None
    total_start: int = None
    total_finish: int = None
    peak_allocated: int = None
    peak_reserved: int = None
    total_peak: int = None

    def __str__(self):
        return (
            f"{'':>15} | {'Start':>10} | {'Finish':>10} | {'Peak':>10}\n"
            f"{'-'*55}\n"
            f"{'Allocated':>15} | {self.allocated_start!s:>10} | {self.allocated_finish!s:>10} | {self.peak_allocated!s:>10}\n"
            f"{'Reserved':>15} | {self.reserved_start!s:>10} | {self.reserved_finish!s:>10} | {self.peak_reserved!s:>10}\n"
            f"{'Total':>15} | {self.total_start!s:>10} | {self.total_finish!s:>10} | {self.total_peak!s:>10}"
        )

@contextmanager
def monitor_gpu_memory_smi(interval=0.05, gpu_index=0):
    """
    Context manager to track:
    - PyTorch allocated & reserved (live and peak)
    - Total GPU memory used by this process (via `nvidia-smi`)
    Does NOT use pynvml.
    """
    pid = str(os.getpid())
    peak_total = [0]
    running = [True]

    def get_process_memory_mb():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                text=True
            )
            for line in result.stdout.strip().split("\n"):
                mem = int(line.split(",")[1].strip())
                return mem
        except Exception:
            return 0
        return 0

    def poll_nvidia_smi():
        while 1:
            mem = get_process_memory_mb()
            peak_total[0] = max(peak_total[0], mem)
            if not running[0]:
                break
            time.sleep(interval)

    # Reset PyTorch memory stats
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    allocated_start = torch.cuda.memory_allocated() // 1024**2
    reserved_start = torch.cuda.memory_reserved() // 1024**2
    total_start = get_process_memory_mb()

    # Start background polling thread
    thread = threading.Thread(target=poll_nvidia_smi)
    thread.start()

    measurement = Measurement()

    try:
        yield measurement
    finally:
        # Stop thread
        running[0] = False
        thread.join()
        torch.cuda.synchronize()

        # Gather PyTorch stats
        allocated_finish = torch.cuda.memory_allocated() // 1024**2
        reserved_finish = torch.cuda.memory_reserved() // 1024**2
        total_finish = get_process_memory_mb()
        peak_allocated = torch.cuda.max_memory_allocated() // 1024**2
        peak_reserved = torch.cuda.max_memory_reserved() // 1024**2
        total_peak = peak_total[0]

        # Reset PyTorch memory stats
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        measurement.allocated_start = allocated_start
        measurement.allocated_finish = allocated_finish
        measurement.peak_allocated = peak_allocated
        measurement.reserved_start = reserved_start
        measurement.reserved_finish = reserved_finish
        measurement.peak_reserved = peak_reserved
        measurement.total_start = total_start
        measurement.total_finish = total_finish
        measurement.total_peak = total_peak

