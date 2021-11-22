#include "conv_gpu.h"
#include <math.h>
#define BLOCK_SIZE 32

__constant__ float mask[127];
__host__ void upload_filterkernel(const filterkernel_cpu &kernel) {
    cudaMemcpyToSymbol(mask, kernel.data, kernel.ks * sizeof(float));
    CUDA_CHECK_ERROR;
}

__global__ void conv_h_gpu_gmem_kernel(unsigned int * dst_data, unsigned int * src_data, const int w, float *kernel_data, int ks){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < ks; i++) {
        int xx = col + (i - ks / 2);
                // Clamp to [0, w-1]
                xx = max(min(xx, w - 1), 0);
                unsigned int pixel = src_data[row * w + xx];

                unsigned char r = pixel & 0xff;
                unsigned char g = (pixel >> 8) & 0xff;
                unsigned char b = (pixel >> 16) & 0xff;

                rr += r * kernel_data[i];
                gg += g * kernel_data[i];
                bb += b * kernel_data[i];
            }

            unsigned char rr_c = rr + 0.5f;
            unsigned char gg_c = gg + 0.5f;
            unsigned char bb_c = bb + 0.5f;
            dst_data[row * w + col] = rr_c | (gg_c << 8) | (bb_c << 16);
}

__global__ void conv_v_gpu_gmem_kernel(unsigned int * dst_data, unsigned int * src_data, const int w, const int h, 	float* kernel_data, int ks) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
            float rr = 0.0f, gg = 0.0f, bb = 0.0f;

            for (int i = 0; i < ks; i++) {
                int yy = row + (i - ks / 2);

                // Clamp to [0, h-1]
                yy = max(min(yy, h - 1), 0);
                unsigned int pixel = src_data[yy * w + col];
                unsigned char r = pixel & 0xff;
                unsigned char g = (pixel >> 8) & 0xff;
                unsigned char b = (pixel >> 16) & 0xff;

                rr += r * kernel_data[i];
                gg += g * kernel_data[i];
                bb += b * kernel_data[i];

            }
            unsigned char rr_c = rr + 0.5f;
            unsigned char gg_c = gg + 0.5f;
            unsigned char bb_c = bb + 0.5f;
            dst_data[row * w + col] = rr_c | (gg_c << 8) | (bb_c << 16);
}

void conv_h_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(div_up(src.width, dimBlock.x), div_up(src.height, dimBlock.y));
    conv_h_gpu_gmem_kernel<<<dimGrid, dimBlock>>>(dst.data, src.data, src.width, kernel.data, kernel.ks);
}

void conv_v_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(div_up(src.width, dimBlock.x), div_up(src.height, dimBlock.y));
    conv_v_gpu_gmem_kernel<<<dimGrid, dimBlock>>>(dst.data, src.data, src.width, src.height, kernel.data, kernel.ks);
}
__global__
void conv_h_gpu_smem_kernel(unsigned int * dst_data, unsigned int * src_data, const int w, const int h, float *kernel_data, int ks){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ int src_shared[BLOCK_SIZE][BLOCK_SIZE];

    //coordinative loading of src_shared from global memory
    src_shared[ty][tx] = src_data[row * w + col];
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;

    for (int i = 0; i < ks; i++) {
        int offset = i - ks / 2;
        int xx = tx + offset;

        if((col + offset)>=0 && (col + offset <w)){
        // Clamp to [0, tile_width-1]
        xx = max(min(xx, BLOCK_SIZE - 1), 0);
        unsigned int pixel = src_shared[ty][xx];

        unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;

        rr += r * kernel_data[i];
        gg += g * kernel_data[i];
        bb += b * kernel_data[i];
    }
    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;

    dst_data[row * w + col] = rr_c | (gg_c << 8) | (bb_c << 16);
}
__global__
void conv_v_gpu_smem_kernel(unsigned int* dst_data, unsigned int* src_data, const int w, const int h, float* kernel_data, int ks){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tx=threadIdx.x, ty=threadIdx.y;
    __shared__ int src_shared[BLOCK_SIZE][BLOCK_SIZE];

    float rr = 0.0f, gg = 0.0f, bb = 0.0f;

    src_shared[ty][tx] = dst_data[row * w + col];

    for (int i = 0; i < ks; i++) {
        int yy = ty + (i - ks / 2);
        int offset = i - ks / 2;
        // Clamp to [0, w-1]
        if((row + offset)>=0 && (row + offset <h)){
        yy = max(min(yy, BLOCK_SIZE - 1), 0);

        unsigned int pixel = src_shared[yy][tx];
        unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;

        rr += r * kernel_data[i];
        gg += g * kernel_data[i];
        bb += b * kernel_data[i];
    }
    }

    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;

    dst_data[row * w + col] = rr_c | (gg_c << 8) | (bb_c << 16);
    //printf("dst_data[row * w + col]: %d ", dst_data[row * w + col]);
}

void conv_h_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(div_up(src.width, dimBlock.x), div_up(src.height, dimBlock.y));
    conv_h_gpu_smem_kernel<<<dimGrid, dimBlock>>>(dst.data, src.data, src.width, src.height, kernel.data, kernel.ks);
}
void conv_v_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(div_up(src.width, dimBlock.x), div_up(src.height, dimBlock.y));
    conv_v_gpu_smem_kernel<<<dimGrid, dimBlock>>>(dst.data, src.data, src.width, src.height, kernel.data, kernel.ks);
}


__global__
void conv_h_gpu_cmem_kernel(unsigned int* dst_data, unsigned int* src_data, const int w, const int h, int ks){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;
    for (int i = 0; i < ks; i++) {
        int xx = col + (i - ks / 2);
        // Clamp to [0, w-1]
        xx = max(min(xx, w - 1), 0);
        unsigned int pixel = src_data[row * w + xx];

        unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;

        rr += r * mask[i];
        gg += g * mask[i];
        bb += b * mask[i];
    }

    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dst_data[row * w + col] = rr_c | (gg_c << 8) | (bb_c << 16);
}
__global__
void conv_v_gpu_cmem_kernel(unsigned int* dst_data, unsigned int* src_data, const int w, const int h,  int ks){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float rr = 0.0f, gg = 0.0f, bb = 0.0f;

    for (int i = 0; i < ks; i++) {
        int yy = row + (i - ks / 2);

        // Clamp to [0, w-1]
        yy = max(min(yy, h - 1), 0);

        unsigned int pixel = src_data[yy * w + col];
        unsigned char r = pixel & 0xff;
        unsigned char g = (pixel >> 8) & 0xff;
        unsigned char b = (pixel >> 16) & 0xff;

        rr += r * mask[i];
        gg += g * mask[i];
        bb += b * mask[i];
        //printf("kernel_data[i]: %f ", kernel_data[i]);

    }
    unsigned char rr_c = rr + 0.5f;
    unsigned char gg_c = gg + 0.5f;
    unsigned char bb_c = bb + 0.5f;
    dst_data[row * w + col] = rr_c | (gg_c << 8) | (bb_c << 16);
}

void conv_h_gpu_cmem(image_gpu &dst, const image_gpu &src,const filterkernel_cpu &kernel){
    upload_filterkernel(kernel);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(div_up(src.width, dimBlock.x), div_up(src.height, dimBlock.y));
    conv_h_gpu_cmem_kernel<<<dimGrid, dimBlock>>>(dst.data, src.data, src.width, src.height,  kernel.ks);
}
void conv_v_gpu_cmem(image_gpu &dst, const image_gpu &src, const int ks){
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(div_up(src.width, dimBlock.x), div_up(src.height, dimBlock.y));
    conv_v_gpu_cmem_kernel<<<dimGrid, dimBlock>>>(dst.data, src.data, src.width, src.height,  ks);
}