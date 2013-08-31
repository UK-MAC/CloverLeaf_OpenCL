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
 *  @brief OCL device-side generate chunk kernel
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Generate the chunks for the simulation.
 *  Note that state one is always used as the background state, which is then
 *  overwritten by further state definitions.
 */

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define ARRAY1D(i_index,i_lb) ((i_index)-(i_lb))
#define ARRAY2D(i_index,j_index,i_size,i_lb,j_lb) ((i_size)*((j_index)-(j_lb))+(i_index)-(i_lb))

__kernel void generate_chunk_ocl_kernel(
        __global double *vertexx,    
        __global double *vertexy,    
        __global double *cellx,
        __global double *celly,
        __global double *density0,
        __global double *energy0,
        __global double *xvel0,
        __global double *yvel0,
        const int number_of_states,
        __global double *state_density,
        __global double *state_energy,
        __global double *state_xvel,
        __global double *state_yvel,
        __global double *state_xmin,
        __global double *state_xmax,
        __global double *state_ymin,
        __global double *state_ymax,
        __global double *state_radius,
        __global int *state_geometry,
        const int g_rect,
        const int g_circ)
{
    double radius;

    int k = get_global_id(1)-2;
    int j = get_global_id(0)-2;

    if ( (j<=XMAX+2) && (k<=YMAX+2) ) {

        int highest_state = 1;

        for (int state = 1; state <= number_of_states; state++) {
            if(state_geometry[ARRAY1D(state,1)] == g_rect) {
              if(vertexx[ARRAY1D(j,-1)] >= state_xmin[ARRAY1D(state,1)] && vertexx[ARRAY1D(j,-1)] < state_xmax[ARRAY1D(state,1)]) {
                if(vertexy[ARRAY1D(k,-1)] >= state_ymin[ARRAY1D(state,1)] && vertexy[ARRAY1D(k,-1)] < state_ymax[ARRAY1D(state,1)]) {
                    highest_state = state;
                }
              }
            } else if(state_geometry[ARRAY1D(state,1)] == g_circ ) {
              radius=sqrt(cellx[ARRAY1D(j,-1)]*cellx[ARRAY1D(j,-1)]+celly[ARRAY1D(k,-1)]*celly[ARRAY1D(k,-1)]);
              if(radius <= state_radius[ARRAY1D(state,1)]){
                    highest_state = state;
              }
            }
        }

        energy0[ARRAY2D(j,k,XMAX+4,XMIN-2,YMIN-2)]=state_energy[ARRAY1D(highest_state,1)];
        density0[ARRAY2D(j,k,XMAX+4,XMIN-2,YMIN-2)]=state_density[ARRAY1D(highest_state,1)];
        xvel0[ARRAY2D(j,k,XMAX+5,XMIN-2,YMIN-2)]=state_xvel[ARRAY1D(highest_state,1)];
        xvel0[ARRAY2D(j+1,k,XMAX+5,XMIN-2,YMIN-2)]=state_xvel[ARRAY1D(highest_state,1)];
        xvel0[ARRAY2D(j,k+1,XMAX+5,XMIN-2,YMIN-2)]=state_xvel[ARRAY1D(highest_state,1)];
        xvel0[ARRAY2D(j+1,k+1,XMAX+5,XMIN-2,YMIN-2)]=state_xvel[ARRAY1D(highest_state,1)];
        yvel0[ARRAY2D(j,k,XMAX+5,XMIN-2,YMIN-2)]=state_yvel[ARRAY1D(highest_state,1)];
        yvel0[ARRAY2D(j+1,k,XMAX+5,XMIN-2,YMIN-2)]=state_yvel[ARRAY1D(highest_state,1)];
        yvel0[ARRAY2D(j,k+1,XMAX+5,XMIN-2,YMIN-2)]=state_yvel[ARRAY1D(highest_state,1)];
        yvel0[ARRAY2D(j+1,k+1,XMAX+5,XMIN-2,YMIN-2)]=state_yvel[ARRAY1D(highest_state,1)];
  }

}
