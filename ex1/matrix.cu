#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

#include "common.h"
#include "matrix.h"


CPUMatrix matrix_alloc_cpu(int width, int height)
{
	CPUMatrix m;
	m.width = width;
	m.height = height;
	m.elements = new float[m.width * m.height];
	return m;
}
void matrix_free_cpu(CPUMatrix &m)
{
	delete[] m.elements;
}

GPUMatrix matrix_alloc_gpu(int width, int height)
{
	// TODO (Task 4): Allocate memory at the GPU
	GPUMatrix Md;
	Md.width=width;
	Md.height=height;
	cudaMallocPitch((void**)&Md.elements, &Md.pitch, width * sizeof(float), height);
	return Md;
	
}
void matrix_free_gpu(GPUMatrix &m)
{
	// TODO (Task 4): Free the memory
	cudaFree(m.elements);
}

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst)
{
	// TODO (Task 4): Upload CPU matrix to the GPU
	int size=src.width*src.height*sizeof(float);
	//cudaMalloc(&src.elements, size);
	cudaMemcpy(dst.elements,src.elements,size,cudaMemcpyHostToDevice);

}
void matrix_download(const GPUMatrix &src, CPUMatrix &dst)
{
	// TODO (Task 4): Download matrix from the GPU
	int size=src.width*src.height*sizeof(float);
	//cudaMalloc(&dst.elements, size);
	cudaMemcpy(dst.elements,src.elements,size,cudaMemcpyDeviceToHost);
}

void matrix_compare_cpu(const CPUMatrix &a, const CPUMatrix &b)
{
	// TODO (Task 4): compare both matrices a and b and print differences to the console
	if(a.width==b.width&&a.height==b.height) 
	{
		std::cout << "compare success" << std::endl;
	}
	else {
		std::cout << "compare fail" << std::endl;
	}
}


