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
 *  @brief OCL device-side field summary kernel
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details The total mass, internal energy, kinetic energy and volume weighted
 *  pressure for the chunk is calculated.
 */

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define ARRAYXY(x_index, y_index, x_width) ((y_index)*(x_width)+(x_index))

__kernel void field_summary_ocl_kernel(
    __global double *volume,
    __global double *density0,
    __global double *energy0,
    __global double *pressure,
    __global double *xvel0,
    __global double *yvel0,
    __global double *vol_tmp_array,
    __global double *mass_tmp_array,
    __global double *ie_tmp_array,
    __global double *ke_tmp_array,
    __global double *press_tmp_array)
{   

    double vsqrd,cell_vol,cell_mass;

    __local double vol_sum_local[WORKGROUP_SIZE];
    __local double mass_sum_local[WORKGROUP_SIZE];
    __local double ie_sum_local[WORKGROUP_SIZE];
    __local double ke_sum_local[WORKGROUP_SIZE];
    __local double press_sum_local[WORKGROUP_SIZE];

    int k = get_global_id(1);
    int j = get_global_id(0);

    int localid = get_local_id(0);

    vol_sum_local[localid] = 0;
    mass_sum_local[localid]= 0;
    ie_sum_local[localid] = 0;
    ke_sum_local[localid] = 0;
    press_sum_local[localid] = 0;

    if ( (j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE) ) {

        vsqrd = 0.25 * ( pow(xvel0[ARRAYXY(j  ,k  , XMAXPLUSFIVE)], 2) + pow(yvel0[ARRAYXY(j  ,k  , XMAXPLUSFIVE)], 2) ) +
	            0.25 * ( pow(xvel0[ARRAYXY(j+1,k  , XMAXPLUSFIVE)], 2) + pow(yvel0[ARRAYXY(j+1,k  , XMAXPLUSFIVE)], 2) ) +
	            0.25 * ( pow(xvel0[ARRAYXY(j  ,k+1, XMAXPLUSFIVE)], 2) + pow(yvel0[ARRAYXY(j  ,k+1, XMAXPLUSFIVE)], 2) ) +
	            0.25 * ( pow(xvel0[ARRAYXY(j+1,k+1, XMAXPLUSFIVE)], 2) + pow(yvel0[ARRAYXY(j+1,k+1, XMAXPLUSFIVE)], 2) );

        cell_vol = volume[ARRAYXY(j, k, XMAXPLUSFOUR)];
        cell_mass = cell_vol * density0[ARRAYXY(j, k, XMAXPLUSFOUR)];

        vol_sum_local[localid] = cell_vol;
        mass_sum_local[localid] = cell_mass;
        ie_sum_local[localid] = cell_mass * energy0[ARRAYXY(j, k, XMAXPLUSFOUR)]; 
        ke_sum_local[localid] = cell_mass * 0.5 * vsqrd;
        press_sum_local[localid] = cell_vol * pressure[ARRAYXY(j, k, XMAXPLUSFOUR)];

    }

    if ((j>=2) || (k>=2)) {

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int limit = WORKGROUP_SIZE_DIVTWO; limit > 0; limit >>= 1) {

            if (localid < limit) {

                vol_sum_local[localid] = vol_sum_local[localid] + vol_sum_local[localid + limit];
                mass_sum_local[localid] = mass_sum_local[localid] + mass_sum_local[localid + limit];
                ie_sum_local[localid] = ie_sum_local[localid] + ie_sum_local[localid + limit];
                ke_sum_local[localid] = ke_sum_local[localid] + ke_sum_local[localid + limit];
                press_sum_local[localid] = press_sum_local[localid] + press_sum_local[localid + limit];
            }
            barrier(CLK_LOCAL_MEM_FENCE);  
        }
    }

    if (localid==0) {

        int write_loc = get_group_id(1)*get_num_groups(0) + get_group_id(0);
        vol_tmp_array[write_loc] = vol_sum_local[0];
        mass_tmp_array[write_loc] = mass_sum_local[0];
        ie_tmp_array[write_loc] = ie_sum_local[0]; 
        ke_tmp_array[write_loc] = ke_sum_local[0];
        press_tmp_array[write_loc] = press_sum_local[0];
    }

}
