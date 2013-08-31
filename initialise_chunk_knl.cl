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
 *  @brief OCL device-side initialise chunk kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Invokes the user specified chunk initialisation kernels.
 */

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define ARRAY1D(i_index,i_lb) ((i_index)-(i_lb))
#define ARRAY2D(i_index,j_index,i_size,i_lb,j_lb) ((i_size)*((j_index)-(j_lb))+(i_index)-(i_lb))

#define ARRAYXY(x_index, y_index, x_width) ((y_index)*(x_width)+(x_index))

__kernel void initialise_chunk_cell_x_ocl_kernel(
    const double dx,
    __global double *vertexx,
    __global double *cellx,
    __global double *celldx)
{

    int j = get_global_id(0)-2;

    if (j<=XMAX+2) {

        cellx[ARRAY1D(j, XMIN-2)] = 0.5 * ( vertexx[ARRAY1D(j, XMIN-2)] + vertexx[ARRAY1D(j+1, XMIN-2)] );
        
        celldx[ARRAY1D(j, XMIN-2)] = dx;

    }
}


__kernel void initialise_chunk_cell_y_ocl_kernel(
    const double dy,
    __global double *vertexy,
    __global double *celly,
    __global double *celldy)
{

    int k = get_global_id(0)-2;
    
    if (k<=YMAX+2) {

        celly[ARRAY1D(k, YMIN-2)] = 0.5 * ( vertexy[ARRAY1D(k, YMIN-2)] + vertexy[ARRAY1D(k+1, YMIN-2)] );
        
        celldy[ARRAY1D(k, YMIN-2)] = dy;
    }
}


__kernel void initialise_chunk_vertex_x_ocl_kernel(
    const double xmin,
    const double dx,
    __global double *vertexx,
    __global double *vertexdx)
{

    int j = get_global_id(0)-2;

    if (j<=XMAX+3) {

        vertexx[ARRAY1D(j,XMIN-2)] = xmin + dx * (j-XMIN);
        vertexdx[ARRAY1D(j,XMIN-2)] = dx;

    }
}


__kernel void initialise_chunk_vertex_y_ocl_kernel(
    const double ymin,
    const double dy,
    __global double *vertexy,
    __global double *vertexdy)
{  

    int k = get_global_id(0)-2;

    if (k<=YMAX+3) {

        vertexy[ARRAY1D(k, YMIN-2)] = ymin + dy * (k-YMIN);

        vertexdy[ARRAY1D(k, YMIN-2)] = dy;

    }

}

__kernel void initialise_chunk_volume_area_ocl_kernel(
    const double dx,
    const double dy,
    __global double *volume,
    __global double *celldx,
    __global double *celldy,
    __global double *xarea,
    __global double *yarea)
{   
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j<=XMAXPLUSTHREE) && (k<=YMAXPLUSTHREE) ) {
    
        volume[ARRAYXY(j,k, XMAXPLUSFOUR)] = dx * dy;

        xarea[ARRAYXY(j,k, XMAXPLUSFIVE)] = celldy[k];

        yarea[ARRAYXY(j,k, XMAXPLUSFOUR)] = celldx[j];
    }
}


