#include <iostream>

#include "matmul.h"
#include "test.h"
#include "common.h"
#include "mul_cpu.h"
#include "mul_gpu.h"
#include "timer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h> 


void print_cuda_devices()
{
	/*int nDevices;
  	cudaGetDeviceCount(&nDevices);
  	for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("Device name: %s\n", prop.name);
    printf("Device Prop: %d.%d\n", 
			prop.major, prop.minor);
    printf("Clock rate: %.2f GHz\n", 
			prop.clockRate * 1e-6f);
	printf("Total amount of Global Memory: %u MiB\n", 
			prop.totalGlobalMem/(1024*1024));
	printf("L2 Cache size: %d KiB\n", 
			prop.l2CacheSize/1024);
	
  }*/

}

 void matmul()
{
	CPUMatrix M = matrix_alloc_cpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);
	CPUMatrix N = matrix_alloc_cpu(pmpp::N_WIDTH, pmpp::N_HEIGHT);
	CPUMatrix P = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
	pmpp::fill(M,N);
	timer_tp start = timer_now();
	matrix_mul_cpu(M,N,P);
	timer_tp end=timer_now();
	float time_diff = timer_elapsed (start,end);
	printf("CPU processing : %f ms\n",time_diff);
	pmpp::test_cpu(P);
	
	//cudaSetDevice
	GPUMatrix A = matrix_alloc_gpu(pmpp::M_WIDTH, pmpp::M_HEIGHT);
	GPUMatrix B = matrix_alloc_gpu(pmpp::N_WIDTH, pmpp::N_HEIGHT);
	GPUMatrix C = matrix_alloc_gpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);	
	matrix_upload(M,A);
	matrix_upload(N,B);
	
	cudaEvent_t evStart,evStop;
	cudaEventCreate(&evStart);
	cudaEventCreate(&evStop);
	cudaEventRecord(evStart,0);
	matrix_mul_gpu(A,B,C);

	cudaEventRecord(evStop,0);
    cudaEventSynchronize(evStop);

	float elapsedTime_ms;
	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
	printf("GPU processing : %f ms\n",elapsedTime_ms);

	//cudaEventDestory(evStart);
	//cudaEventDestory(evStop);


	CPUMatrix I = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
	matrix_download(C,I);	
	
	//correctness (pmpp::test_gpu(const CPUMatrix &p))

    

	matrix_compare_cpu(P,I);
	
	pmpp::test_gpu(I);

	matrix_free_gpu(A);
	matrix_free_gpu(B);
	matrix_free_gpu(C);

	


	// === Task 3 ===
	// TODO: Allocate CPU matrices (see matrix.cc)
	//       Matrix sizes:
	//       Input matrices:
	//       Matrix M: pmpp::M_WIDTH, pmpp::M_HEIGHT
	//       Matrix N: pmpp::N_WIDTH, pmpp::N_HEIGHT
	//       Output matrices:
	//       Matrix P: pmpp::P_WIDTH, pmpp::P_HEIGHT
	// TODO: Fill the CPU input matrices with the provided test values (pmpp::fill(CPUMatrix &m, CPUMatrix &n))
	// TODO (Task 5): Start CPU timing here!
	// TODO: Run your implementation on the CPU (see mul_cpu.cc)
	// TODO (Task 5): Stop CPU timing here!
	// TODO: Check your matrix for correctness (pmpp::test_cpu(const CPUMatrix &p))

	// === Task 4 ===
	// TODO: Set CUDA device
	// TODO: Allocate GPU matrices (see matrix.cc)
	// TODO: Upload the CPU input matrices to the GPU (see matrix.cc)
	// TODO (Task 5): Start GPU timing here!
	// TODO: Run your implementation on the GPU (see mul_gpu.cu)
	// TODO (Task 5): Stop GPU timing here!
	// TODO: Download the GPU output matrix to the CPU (see matrix.cc)
	// TODO: Check your downloaded matrix for correctness (pmpp::test_gpu(const CPUMatrix &p))
	// TODO: Compare CPU result with GPU result (see matrix.cc)

	// TODO (Task3/4/5): Cleanup ALL matrices and and events
}




/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 4) 6. Where do the differences come from?
 * 
 * Answer: TODO
 * 
 * 
 ************************************************************/
