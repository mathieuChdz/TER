from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetCount, nvmlShutdown # type: ignore

def init_NVML():
    """
    Initialize NVML (NVIDIA Management Library).
    """
    nvmlInit()

def stop_NVML():
    """
    Stop NVML (NVIDIA Management Library).
    """
    nvmlShutdown()

def get_total_power_usage()->int:
    """
    Retourne la consommation énergétique totale de tous les GPUs en watts
    """
    device_count = nvmlDeviceGetCount()
    total_power_usage = 0
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        power_usage = nvmlDeviceGetPowerUsage(handle) / 1000  #On convertit en watts
        total_power_usage += power_usage
    return total_power_usage
