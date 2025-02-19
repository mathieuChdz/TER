import matplotlib.pyplot as plt

def Fibonacci(n):
    if n < 0:
        print("Incorrect input")
    elif n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return Fibonacci(n-1) + Fibonacci(n-2)
    
def multiple_Fibonacci(n, max_size):
    x = []
    y = []
    for size in range(0, max_size, max_size // n):
        time = Fibonacci(size)
        x.append(size)
        y.append(time)
    return x, y
    
def draw(x, y, filename):
    plt.plot(x, y)
    plt.xlabel('Fibonacci number')
    plt.ylabel('Time (s)')
    plt.title('Fibonacci sequence')
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    x, y = multiple_Fibonacci(10, 20)
    draw(x, y, "cpu_fibonacci.png")