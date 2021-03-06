/*Crown Copyright 2012 AWE.
*
* This file is part of CloverLeaf.
*
* CloverLeaf is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3 of the License, or (at your option)
* any later version.
*
* CloverLeaf is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
* details.
*
* You should have received a copy of the GNU General Public License along with
* CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *  @brief OCL device-side summation reduction kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Calculates the sum of the input field (CPU and GPU)
 */

#include "ocl_knls.h"

__kernel void reduction_sum_cpu_ocl_kernel(
	__global const double * restrict sum_val_input,
	__global double * restrict sum_val_output,
    const int total_num_elements)
{
    int group_id= get_group_id(0);
    int num_groups = get_num_groups(0);
    int inc; 
    int one_more; 
    int remainder = total_num_elements % num_groups;
    int num_per_group = total_num_elements / num_groups;
    double sum_value = 0; 
    
    if (group_id < remainder) {
        inc = group_id;
        one_more = 1;
    }   
    else {
        inc = remainder; 
        one_more = 0;
    }

    int start_index = group_id*num_per_group+inc;
    int i;

    for ( i=start_index; i<start_index+num_per_group+one_more; i++) {
        sum_value += sum_val_input[i];    
    }
    
    sum_val_output[group_id] = sum_value; 
}

__kernel void reduction_sum_ocl_kernel(
	__global const double * restrict sum_val_input,
	__local double * restrict sum_val_local,
	__global double * restrict sum_val_output)
{
    uint lj = get_local_id(0);
    uint wg_size_x = get_local_size(0);
    uint wg_id_x = get_group_id(0);

    uint j = wg_id_x * (wg_size_x * 2) + lj;

    sum_val_local[lj] = sum_val_input[j] + sum_val_input[j+wg_size_x]; 

    barrier(CLK_LOCAL_MEM_FENCE); 

    for (uint s=wg_size_x >> 1; s > 32; s >>= 1) {
        if (lj < s) {
            sum_val_local[lj] += sum_val_local[lj+s];
	}

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lj < 32) sum_val_local[lj] += sum_val_local[lj+32];
    if (lj < 16) sum_val_local[lj] += sum_val_local[lj+16];
    if (lj < 8) sum_val_local[lj] += sum_val_local[lj+8];
    if (lj < 4) sum_val_local[lj] += sum_val_local[lj+4];
    if (lj < 2) sum_val_local[lj] += sum_val_local[lj+2];
    if (lj < 1) sum_val_local[lj] += sum_val_local[lj+1];

    if (lj==0) sum_val_output[wg_id_x] = sum_val_local[0];
}

__kernel void reduction_sum_last_ocl_kernel(
	__global const double * restrict sum_val_input,
	__local double * restrict sum_val_local,
	__global double * restrict sum_val_output,
	const int limit,
	const int even)
{
    uint lj = get_local_id(0);
    uint wg_size_x = get_local_size(0);
    uint wg_id_x = get_group_id(0);

    uint j = wg_id_x * (wg_size_x * 2) + lj;

    if ( wg_id_x != get_num_groups(0)-1 ) {
        sum_val_local[lj] = sum_val_input[j] + sum_val_input[j+wg_size_x]; 
    }
    else if (lj < limit) {
        sum_val_local[lj] = sum_val_input[j] + sum_val_input[j+limit]; 
    }
    else if ( (lj==limit) && (even==0) ) { //even is false
        sum_val_local[lj] = sum_val_input[j+limit];
    }
    else {
        sum_val_local[lj] = 0; 
    }

    barrier(CLK_LOCAL_MEM_FENCE); 

    for (uint s = wg_size_x >> 1; s > 16; s >>= 1) {
        if (lj < s) {
            sum_val_local[lj] += sum_val_local[lj+s];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (lj < 16) sum_val_local[lj] += sum_val_local[lj+16];
    if (lj < 8) sum_val_local[lj] += sum_val_local[lj+8];
    if (lj < 4) sum_val_local[lj] += sum_val_local[lj+4];
    if (lj < 2) sum_val_local[lj] += sum_val_local[lj+2];
    if (lj < 1) sum_val_local[lj] += sum_val_local[lj+1];  

    if (lj==0) sum_val_output[wg_id_x] = sum_val_local[0];
}
