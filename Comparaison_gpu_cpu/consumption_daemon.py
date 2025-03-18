import threading
import gpu_consumption as gc

class Consumtion_daemon(threading.Thread):
    def __init__(self, name, interval=1):
        threading.Thread.__init__(self)
        self.name = name
        self.interval = interval
        self.flagRun = True
        self.power_list = []
        self.event = threading.Event()
        self.begin_power = 0
        self.power_sum_list = []

    def initializePower(self):
        gc.init_NVML()
        self.begin_power = gc.get_total_power_usage()
        gc.stop_NVML()
    
    def getPowerTotalSum(self):
        return self.power_sum_list
    
    def addPowerSumList(self):
        self.power_sum_list.append(sum(self.power_list))

    def addLastPower(self):
        gc.init_NVML()
        power = gc.get_total_power_usage()
        self.power_list.append(power - self.begin_power)
    
    def resetPowerList(self):
        self.power_list = []

    def run(self):
        gc.init_NVML()
        while self.flagRun:
            power = gc.get_total_power_usage()
            self.power_list.append(power - self.begin_power)
            self.event.wait(self.interval)
        gc.stop_NVML()
    
    def stop(self):
        self.flagRun = False
        self.event.set()
        self.join()

    def getConsumption(self):
        return self.power_list