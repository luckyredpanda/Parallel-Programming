#include <cuda_runtime.h>
#include <stdio.h>

// NOTE: if you include stdio.h, you can use printf inside your kernel

#include "common.h"
#include "matrix.h"
#include "mul_gpu.h"

// TODO (Task 4): Implement matrix multiplication CUDA kernel
__global__ void matrix_multiply(GPUMatrix m, GPUMatrix n, GPUMatrix p)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;	
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	float pValue=0;
	for(int k=0;k<m.width;++k)
	{
		float Melement = m.elements[row * m.width + k]; 
		float Nelement = n.elements[k * n.width+ col]; 
		pValue += Melement * Nelement;
	}
	p.elements[row * p.width + col]= pValue;
}


void matrix_mul_gpu(const GPUMatrix &m, const GPUMatrix &n, GPUMatrix &p)
{	
	dim3 block(32, 32);
    dim3 grid(div_up(n.width,block.x),div_up(m.height,block.y));
	matrix_multiply<<<grid,block>>>(m,n,p);
	// TODO (Task 4): Determine execution configuration and call CUDA kernel

}
	
