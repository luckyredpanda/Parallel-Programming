## Parallel-Programming
21ws Parallel Programming labs TU Darmstadt

### lab1:
1. Set the CUDA device
2. Allocate necessary CPU and GPU memory (you have to use cudaMallocPitch() for GPU memory) 3. Transfer input matrices to the GPU
4. Multiply matrices on the GPU
5. Transfer the result matrix back to the CPU
6. Ceck your output matrix using the pmpp::test_gpu function.
7. Compare the result matrix against the CPU version. Where do the differences come from?

### lab2:
1. Your task is to implement Gaussian blur by using the given code to load an image, generate a Gaussian filter kernel, apply Gaussian blur to the image and save the blurred image to a file named out_cpu.ppm. Your program should takes two command-line parameters. The first one is the file name of the image that should be blurred and the second is the filter kernel size.
2. Write a GPU implementation of the horizontal and vertical 1D convolution which works on a ppm image. For this task, you should only use global memory in your implementation. The ppm struct represents the pixels of an image in 4 bytes (1 byte per color channel, 4 color channels (RGBA)). The convolution should only be applied to the RGB color channels; you can ignore the alpha channel. Your implementation should be able to handle different filter kernel sizes. Finally, save the resulting image to a file named out_gpu_gmem.ppm.
3. Copy your convolution CUDA kernels from task 2 and modify them to use shared memory for the pixel data. For an idea on how shared memory can help with convolution, see the whitepaper about the convolutionSeparable CUDA sample. The blurred image should be written to a file named out_gpu_smem.ppm.
4. Copy your convolution CUDA kernels from task 2 and modify them to access the image in global memory and the filter kernel in constant memory. Because the amount of constant memory is determined at compile-time, you may restrict the maximum filter kernel size (e.g. 127). Save the blurred image to a file named out_gpu_cmem.ppm.
5. Copy your convolution CUDA kernels from task 2 and modify them such that they combine all of the optimizations from the previous tasks: cache pixel data in shared memory and access the filter kernel in constant memory. Save the blurred image to a file named out_gpu_all.ppm.

### final project:
1. There’s a lot of noise in the code-base that don’t have do deal with. 
2. Find examples of how to write algorithms in Increaser.hpp, QuickSorter.hpp, Nopper.hpp, Reorderer.hpp, and SelectionSorter.hpp.
3. Students should add classes that encapsulate GPU kernels: Start with doing nothing and just calling a nop-kernel. 
4. Also add kernels to find the max and min. 
5. If managed to do so, also add one to calculate the scalar product of two vectors. 
6. Once students have those four classes, find ones that have super-linear time complexity.
7. If you have done that, you need to test the functionality within patterns. See main.cpp for how to accomplish that.
8. Once you can do this, test the “compositionality”: If you have some functions F, G, and use composition(F, G) just take just as long F and G summed up. For iteration(vector V, F), it should take size(V) * time(F).

Record Table

|flag|function|time (of vector with 1 element/100 elements/10000 elements/65536 elements)|
|---|---|---|
|NK|nop-kernel | 2.84161e-05 sec|
|MIK|min-kernel| 6.28596e-05 sec /6.48727e-05 sec/1.07273e-04 sec/3.56748e-04  sec|
|MAK|max-kernel| 3.81907e-05 sec /4.10672e-05 sec/ 8.46112e-05 sec/ 2.97976e-04 sec|
|SPK|scalarProduct-kernel| 7.21873e-05 sec/7.00473e-05/1.90116e-04 sec/3.18319e-04 sec|
|N|Nopper |1.774e-06 sec|
|I|Increaser|2.2273e-06 sec/1.9257e-06 sec/1.8722e-06 sec/1.79621e-05 sec|
|QS|QuickSorter|2.0591e-06 sec/1.6206e-06 sec/7.84031e-05 sec/6.2228e-04 sec|
|RA|ReduceAdd|2.5477e-06 sec/3.495e-06 sec/1.84241e-05 sec/1.25739e-04 sec|
|RM|ReduceMin|2.031e-06 sec/2.3293e-06 sec/2.3133e-06 sec/2.0059e-06 sec|
|RO|Reorderer|1.7526e-06 sec/2.4009e-06 sec/6.80051e-05 sec/4.78822e-04 sec|
|SS|SelectionSorter|2.121e-06 sec/2.0785e-06 sec/4.50931e-03 sec/8.00362e-02 sec|

|FunA & FunB|timeA+timeB/s|test_time/s|Relative error|
|---|---|---|---|
|MAK+NK|4.549e-04|4.640e.04|2%|
|SPK+NK|4.559e-04|4.702e-04|3.1%|
|MIK + NK|4.554e-04|4.633e-04|1.7%|
|N + NK|2.38e-06|3.885e-06|63.2%|
|I + MIK|4.579e-04|4.745e-04|3.6%|
|QS+MAK|4.426e-04|4.590e-04|3.7%|
|SS+MAK|4.444e-04|4.395e-04|-1.1%|
|RO + SPK|4.589e-04|4.5046e-04|-1.8%|
|SS+SPK|5.003e-03|4.700e-03|-6%|

**Conclusion:**
1. Except for N+NK, the relative errors of the experimental results of all other groups are less than 10%, and it can be considered that the method of estimating the function combination by the sum of each single function is feasible.

2. We find that the relative error is positive for all GPU-to-GPU combinations,We think there may be other factors to consider in the GPU-to-GPU combination.

3. We observe that the relative error is positive for all experiments including the nop-kernel. Especially in the N+NK experiment, the error is as high as 63.2%.I think it can be deduced that there is still something not taken into account in the modeling with nop-kernel.

