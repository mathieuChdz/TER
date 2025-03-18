import matrix_multiplication_test as mm
import matplotlib.pyplot as plt
import os
import gpu_consumption as gc
import consumption_daemon as daemon
import fibonacci as fib
import sys

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

def run_mult_gpu(sizes, deamon):
    
    for size in sizes:
        plot_consumption(size)
        deamon.addLastPower()
        deamon.addPowerSumList()
        deamon.resetPowerList()
    
    return sizes, deamon.getPowerTotalSum(), "graphs/gpu_mult_consumption.png"

def run_fibonacci(n_number, max_size, deamon):
    
    fibonacci_kernel = fib.compile_Fibonacci_GPU()
    x = []
    for size in range(0, max_size, max_size // n_number):
        fib.fibonacci_GPU(size, fibonacci_kernel)
        x.append(size)
        deamon.addLastPower()
        deamon.addPowerSumList()
        deamon.resetPowerList()
    
    return x, deamon.getPowerTotalSum(), "graphs/gpu_fibonacci_consumption.png"

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python3 plotConsumption.py <option>")
        sys.exit(1)

    option = sys.argv[1]

    # On créer et on initialise le daemon
    deamon = daemon.Consumtion_daemon("consumption_daemon")
    deamon.start()

    x = y = title = None

    if option == "mult_gpu" or option == "1":
        sizes = [10000, 20000, 30000, 40000, 50000, 60000]
        sizes_5000 = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]
        x, y, title = run_mult_gpu(sizes, deamon)
        # run_mult_gpu(sizes_5000n deamon)
    
    elif option == "fib" or option == "2":
        point_num = 20
        max_size = 40
        x, y, title = run_fibonacci(point_num, max_size, deamon)
    
    # On stop le daemon
    deamon.stop()
    deamon.join()
    
    daemon_consumption = deamon.getPowerTotalSum()
    print(daemon_consumption)

    if not os.path.exists('graphs'):
            os.makedirs('graphs')

    draw(x, y, title)

    