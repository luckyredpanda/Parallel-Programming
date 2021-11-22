#pragma once

#include "image.h"
#include "common.h"

void conv_h_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_cpu &kernel);//changed filterkernel_gpu to filterkernel_cpu
void conv_v_gpu_cmem(image_gpu &dst, const image_gpu &src, const int ks);    //changed const filterkernel_gpu &kernel to ks

void conv_h_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);

void conv_h_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
void conv_v_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel);
