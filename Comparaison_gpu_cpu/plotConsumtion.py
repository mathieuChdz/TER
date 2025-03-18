import matrix_multiplication_test as mm
import matplotlib.pyplot as plt
import os
import gpu_consumption as gc


def plot_consumption(sizes):
    gc.init_NVML()
    power = gc.get_total_power_usage()
    gc.stop_NVML()
    gpu_consumption = []
    for size in sizes:
        result_gpu, power = mm.run_gpu_with_consumtion(size, power)
        gpu_consumption.append(result_gpu)

    return gpu_consumption

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
    plt.ylabel('Power consumption (W)')
    plt.title('Matrix multiplication on CPU')
    plt.savefig(filename)

if __name__ == "__main__":
    sizes = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
    gpu_consumption = plot_consumption(sizes)

    if not os.path.exists('graphs'):
            os.makedirs('graphs')

    draw(sizes, gpu_consumption, "graphs/gpu_consumption.png")
