#ifndef CLOVER_OCL_KNL_H_
#define CLOVER_OCL_KNL_H_

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define ARRAYXY(x_index, y_index, x_width) ((y_index)*(x_width)+(x_index))

#define ARRAY1D(i_index,i_lb) ((i_index)-(i_lb))
#define ARRAY2D(i_index,j_index,i_size,i_lb,j_lb) ((i_size)*((j_index)-(j_lb))+(i_index)-(i_lb))

#endif
