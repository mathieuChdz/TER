run : bitonic_merge_sort
	./bitonic_merge_sort

bitonic_merge_sort : kernel.cu
	nvcc -o bitonic_merge_sort kernel.cu

clean:
	rm -f bitonic_merge_sort