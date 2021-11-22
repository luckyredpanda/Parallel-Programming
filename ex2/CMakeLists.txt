project(filtering CXX CUDA)
cmake_minimum_required(VERSION 3.8)

# Set the default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

add_executable(gauss_filter
	common.cu
	main.cc
	filtering.cc
	image.cu
	conv_cpu.cc
	conv_gpu.cu)

set_target_properties(gauss_filter PROPERTIES CUDA_ARCHITECTURES "75")
