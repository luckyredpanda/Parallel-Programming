#include "filtering.h"
#include "image.h"
#include "common.h"
#include "conv_cpu.h"
#include "conv_gpu.h"
#include <stdio.h>

//extern __constant__ float mask[];

void filtering(const char *imgfile, int ks)
{
	// === Task 1 ===
	// TODO: Load image
    image_cpu src = {imgfile};
    int w = src.width, h = src.height;

	// TODO: Generate gaussian filter kernel
    filterkernel_cpu kernel_cpu = {ks};

    // TODO: Blur image on CPU
    image_cpu dst = {w, h}; // dst for download and saveing ppm image after bluring

    // 2D convolution on cpu
    conv_h_cpu(dst, src, kernel_cpu);
    conv_v_cpu(dst, src, kernel_cpu);

    dst.save("out_cpu.ppm");


	// === Task 2 ===
	// TODO: Blur image on GPU (Global memory)
    image_gpu gpu_src = {w, h};
    image_gpu gpu_dst = {w, h};
    src.upload(gpu_src);

    filterkernel_gpu kernel_gpu = {ks};
    kernel_cpu.upload(kernel_gpu);
    conv_h_gpu_gmem(gpu_dst, gpu_src, kernel_gpu);
    conv_v_gpu_gmem(gpu_dst, gpu_src, kernel_gpu);

    dst.download(gpu_dst);
    dst.save("out_gpu_gmem.ppm");



    // === Task 3 ===
	// TODO: Blur image on GPU (Shared memory)
    image_gpu gpu_dst3 = {w, h};
    conv_h_gpu_smem(gpu_dst3, gpu_src, kernel_gpu);
    conv_v_gpu_smem(gpu_dst3, gpu_src, kernel_gpu);

    dst.download(gpu_dst3);
    dst.save("out_gpu_smem.ppm");


	// === Task 4 ===
	// TODO: Blur image on GPU (Constant memory)
    image_gpu gpu_dst4 = {w, h};

    conv_h_gpu_cmem(gpu_dst4, gpu_src, ks);
    conv_v_gpu_cmem(gpu_dst4, gpu_src, ks);

    dst.download(gpu_dst4);
    dst.save("out_gpu_cmem.ppm");



	// === Task 5 ===
	// TODO: Blur image on GPU (L1/texture cache)



    // === Task 6 ===
	// TODO: Blur image on GPU (all memory types)
}


/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 7) nvprof output
 * 
 * Answer: TODO
 *
==30508== Profiling application: ./gauss_filter cornellBoxSphere_2048x2048.ppm 127
==30508== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   19.34%  13.225ms         1  13.225ms  13.225ms  13.225ms  conv_h_gpu_gmem_kernel(unsigned int*, unsigned int*, int, float*, int)
                   17.83%  12.197ms         1  12.197ms  12.197ms  12.197ms  conv_v_gpu_gmem_kernel(unsigned int*, unsigned int*, int, int, float*, int)
                   14.12%  9.6537ms         1  9.6537ms  9.6537ms  9.6537ms  conv_h_gpu_smem_kernel(unsigned int*, unsigned int*, int, int, float*, int)
                   14.06%  9.6179ms         1  9.6179ms  9.6179ms  9.6179ms  conv_v_gpu_smem_kernel(unsigned int*, unsigned int*, int, int, float*, int)
                   13.82%  9.4516ms         3  3.1505ms  1.8934ms  5.5936ms  [CUDA memcpy DtoH]
                    8.47%  5.7916ms         1  5.7916ms  5.7916ms  5.7916ms  conv_h_gpu_cmem_kernel(unsigned int*, unsigned int*, int, int, int)
                    7.11%  4.8596ms         1  4.8596ms  4.8596ms  4.8596ms  conv_v_gpu_cmem_kernel(unsigned int*, unsigned int*, int, int, int)
                    5.26%  3.5946ms         3  1.1982ms  1.5680us  3.5914ms  [CUDA memcpy HtoD]
      API calls:   69.55%  198.14ms         5  39.627ms  392.34us  196.41ms  cudaMalloc
                   24.38%  69.439ms         5  13.888ms  24.828us  31.646ms  cudaMemcpy
                    3.47%  9.8724ms       808  12.218us     204ns  793.92us  cuDeviceGetAttribute
                    1.53%  4.3563ms         5  871.26us  211.38us  1.2696ms  cudaFree
                    0.69%  1.9593ms         8  244.92us  237.09us  255.65us  cuDeviceTotalMem
                    0.27%  764.61us         8  95.576us  86.425us  140.96us  cuDeviceGetName
                    0.08%  219.11us         6  36.519us  14.237us  66.229us  cudaLaunchKernel
                    0.02%  60.846us         1  60.846us  60.846us  60.846us  cudaMemcpyToSymbol
                    0.01%  36.981us         8  4.6220us  1.3610us  20.605us  cuDeviceGetPCIBusId
                    0.00%  11.663us        16     728ns     214ns  7.3070us  cuDeviceGet
                    0.00%  3.5120us         3  1.1700us     377ns  2.5840us  cuDeviceGetCount
                    0.00%  2.6080us         8     326ns     267ns     405ns  cuDeviceGetUuid
                    0.00%     948ns         1     948ns     948ns     948ns  cudaGetLastError

 * 
 ************************************************************/
