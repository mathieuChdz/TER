import matplotlib.pyplot as plt
import os

def Fibonacci(n):
    """
    Calculates the nth Fibonacci number.

    Args:
        n (int): The index of the Fibonacci number to calculate.

    Returns:
        int: The nth Fibonacci number.
    """
    if n < 0:
        print("Incorrect input")
    elif n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return Fibonacci(n-1) + Fibonacci(n-2)
    
def multiple_Fibonacci(n, max_size):
    """
    Calculates n Fibonacci numbers of increasing size.

    Args:
        n (int): The number of Fibonacci numbers to calculate.
        max_size (int): The maximum index of the Fibonacci numbers to calculate.

    Returns:
        tuple: Two lists containing the index of the Fibonacci numbers and the time taken to calculate them.
    """
    x = []
    y = []
    for size in range(0, max_size, max_size // n):
        time = Fibonacci(size)
        x.append(size)
        y.append(time)
    return x, y
    
def draw(x, y, filename):
    """
    Draws a plot of the time taken to calculate Fibonacci numbers of increasing size.

    Args:
        x (list): The index of the Fibonacci numbers.
        y (list): The time taken to calculate the Fibonacci numbers.
        filename (str): The name of the file to save the plot.
    """
    plt.plot(x, y)
    plt.xlabel('Fibonacci number')
    plt.ylabel('Time (s)')
    plt.title('Fibonacci sequence')
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    x, y = multiple_Fibonacci(10, 20)
    draw(x, y, "graphs/fibonacci.png")