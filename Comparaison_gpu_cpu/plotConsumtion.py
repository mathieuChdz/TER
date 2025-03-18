import matrix_multiplication_test as mm
import matplotlib.pyplot as plt
import os
import gpu_consumption as gc
import consumption_daemon as daemon

def plot_consumption(size):
    
    result_gpu = mm.gpu_matrix_multiplication(size)
    return result_gpu

def draw(x, y, filename):
    """
    Draws a plot of the time taken to multiply matrices of increasing size on CPU.

    Args:
        x (list): The size of the matrices.
        y (list): The time taken to multiply the matrices.
        filename (str): The name of the file to save the plot.
    """
    plt.plot(x, y)
    plt.xlabel('Matrix size')
    plt.ylabel('Power consumption (sum W/s)')
    plt.title('Matrix multiplication on CPU')
    plt.savefig(filename)

def run_main(sizes):

    # On créer et on initialise le daemon
    deamon = daemon.Consumtion_daemon("consumption_daemon")
    deamon.start()
    deamon.initializePower()
    
    for size in sizes:
        gpu_consumption = plot_consumption(size)
        deamon.addLastPower()
        deamon.addPowerSumList()
        deamon.resetPowerList()

    # On stop le daemon
    deamon.stop()
    deamon.join()
    
    daemon_consumption = deamon.getPowerTotalSum()
    print(daemon_consumption)

    if not os.path.exists('graphs'):
            os.makedirs('graphs')

    draw(sizes, daemon_consumption, "graphs/gpu_consumption.png")

if __name__ == "__main__":
    sizes = [10000, 20000, 30000, 40000, 50000, 60000]
    sizes_5000 = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]

    run_main(sizes)
    # run_main(sizes_5000)