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
 *  @brief OCL device-side timestep calculation kernel
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details calculates the minimum timestep on the mesh chunk based on the CFL
 *  condition, the velocity gradient and the velocity divergence. A safety
 *  factor is used to ensure numerical stability.
 */

#include "ocl_knls.h"

__kernel void calc_dt_ocl_kernel(
        const double g_small,
        const double g_big,
        const double dtmin,                 
        const double dtc_safe,              
        const double dtu_safe,              
        const double dtv_safe,              
        const double dtdiv_safe,            
        __global const double * restrict xarea,
        __global const double * restrict yarea,
        __global const double * restrict cellx,
        __global const double * restrict celly,
        __global const double * restrict celldx,
        __global const double * restrict celldy,
        __global const double * restrict volume,
        __global const double * restrict density0,
        __global const double * restrict energy0,
        __global const double * restrict pressure,
        __global const double * restrict viscosity,
        __global const double * restrict soundspeed,
        __global const double * restrict xvel0,
        __global const double * restrict yvel0,
	    __global double * restrict dt_min_val_array)
{
    double dsx,dsy,cc,dv1,dv2,div,dtct,dtut,dtvt,dtdivt; 

    __local double dt_min_local[WORKGROUP_SIZE];

    int k = get_global_id(1);
    int j = get_global_id(0);

    int localid = get_local_id(1)*get_local_size(0)+get_local_id(0);
    dt_min_local[localid] = 100000;

    dt_min_val_array[ARRAYXY(j,k,XMAXPLUSFIVE)] = g_big;

    if ( (j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE) ) {

        dsx = celldx[j];
        dsy = celldy[k];

        cc = pow( soundspeed[ARRAYXY(j,k,XMAXPLUSFOUR)], 2);
        cc = cc + 2.0 * viscosity[ARRAYXY(j,k,XMAXPLUSFOUR)] / density0[ARRAYXY(j,k,XMAXPLUSFOUR)];
        cc = fmax(sqrt(cc),g_small);

        dtct = dtc_safe * fmin(dsx,dsy)/cc;

        div = 0.0;

        dv1 = (xvel0[ARRAYXY(j  ,k, XMAXPLUSFIVE)]+xvel0[ARRAYXY(j  ,k+1, XMAXPLUSFIVE)])
              * xarea[ARRAYXY(j, k, XMAXPLUSFIVE )];

        dv2 = (xvel0[ARRAYXY(j+1, k, XMAXPLUSFIVE)]+ xvel0[ARRAYXY(j+1, k+1, XMAXPLUSFIVE)])
              * xarea[ARRAYXY(j+1, k, XMAXPLUSFIVE)];

        div = div + dv2 - dv1;

        dtut = dtu_safe * 2.0 * volume[ARRAYXY(j, k, XMAXPLUSFOUR)] 
	           / fmax(fabs(dv1), fmax( fabs(dv2), g_small * volume[ARRAYXY(j, k, XMAXPLUSFOUR)] ) );

        dv1 = ( yvel0[ARRAYXY(j, k, XMAXPLUSFIVE)]+yvel0[ARRAYXY(j+1, k, XMAXPLUSFIVE)])
              * yarea[ARRAYXY(j, k, XMAXPLUSFOUR)];

        dv2 = ( yvel0[ARRAYXY(j, k+1, XMAXPLUSFIVE)] + yvel0[ARRAYXY(j+1, k+1, XMAXPLUSFIVE)]) 
              * yarea[ARRAYXY(j, k+1, XMAXPLUSFOUR)];

        div = div + dv2 - dv1; 

        dtvt = dtv_safe * 2.0 * volume[ARRAYXY(j, k, XMAXPLUSFOUR)] 
	           / fmax( fabs(dv1), fmax( fabs(dv2), g_small * volume[ARRAYXY(j, k, XMAXPLUSFOUR)] ) );

        div = div / ( 2.0 * volume[ARRAYXY(j, k, XMAXPLUSFOUR)] );

        if (div < (-1*g_small)) {
	        dtdivt = dtdiv_safe * (-1.0/div); 
	    } else {
            dtdivt = g_big;
	    }

	    dt_min_local[localid] = fmin( fmin( fmin(dtvt, dtdivt), dtut ), dtct ); 

    }

    //if (k>=2) { 

#ifdef GPU_REDUCTION 

        //GPU reduction 
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int limit = WORKGROUP_SIZE_DIVTWO; limit > 0; limit >>= 1 ) {

            if (localid < limit) {
            
                dt_min_local[localid] = fmin(dt_min_local[localid], dt_min_local[localid + limit]);

            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

#else

        //CPU reduction 
        barrier(CLK_LOCAL_MEM_FENCE);

        if (localid==0) {
            for (int index = 1; index < WORKGROUP_SIZE; index++) {
                dt_min_local[localid] = fmin( dt_min_local[localid], dt_min_local[index] );  
            }
        }

#endif

    //}

    if (localid==0) { dt_min_val_array[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = dt_min_local[0]; }
}


