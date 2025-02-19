import numpy as np

def bitonic_merge(arr, low, cnt, up):
    print(f"bitonic_merge called with low={low}, cnt={cnt}, up={up}")
    if cnt > 1:
        k = cnt // 2
        for i in range(low, low + k):
            if (arr[i] > arr[i + k]) == up:
                arr[i], arr[i + k] = arr[i + k], arr[i]
        print(f"Array after merge step: {arr}")
        bitonic_merge(arr, low, k, up)
        bitonic_merge(arr, low + k, k, up)

def bitonic_sort(arr, low, cnt, up):
    print(f"bitonic_sort called with low={low}, cnt={cnt}, up={up}")
    if cnt > 1:
        k = cnt // 2
        bitonic_sort(arr, low, k, 1)
        bitonic_sort(arr, low + k, k, 0)
        bitonic_merge(arr, low, cnt, up)
    print(f"Array after sort step: {arr}")

def sort(arr):
    print(f"Initial array: {arr}")
    arr = np.array(arr)
    bitonic_sort(arr, 0, len(arr), 1)
    print(f"Sorted array: {arr}")
    return arr

if __name__ == "__main__":
    arr = [3, 7, 2, 5, 1, 4, 6, 8]
    sorted_arr = sort(arr)
    print(f"Final sorted array: {sorted_arr}")