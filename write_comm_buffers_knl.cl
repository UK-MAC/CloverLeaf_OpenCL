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
 *  @brief OCL device-side buffer write kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Provides functionality to write data into device side buffers
 */

#include "ocl_knls.h"

__kernel void write_top_buffer_ocl_kernel(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double * restrict field,
    __global const double * restrict snd_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0)-depth;

    if (j<=XMAX+x_inc+depth) {
        int index = j + depth + (k-1)*(XMAX+x_inc+(2*depth));
        field[ARRAY2D(j,YMAX+y_inc+k, XMAX+4+x_inc, XMIN-2, YMIN-2)] = snd_buffer[ARRAY1D(index, 1)];
    }
}

__kernel void write_bottom_buffer_ocl_kernel(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double * restrict field,
    __global const double * restrict snd_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0)-depth;

    if (j<=XMAX+x_inc+depth ) {
        int index = j + depth + (k-1)*(XMAX+x_inc+(2*depth));
        field[ARRAY2D(j,YMIN-k, XMAX+4+x_inc, XMIN-2, YMIN-2)] = snd_buffer[ARRAY1D(index, 1)];
    }
}

__kernel void write_left_buffer_ocl_kernel(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double * restrict field,
    __global const double * restrict snd_buffer)
{
    int j = get_global_id(1);
    int k = get_global_id(0)-depth;

    if (k<=YMAX+y_inc+depth) {
        int index = j + (k+depth-1)*depth;
        field[ARRAY2D(XMIN-j, k, XMAX+4+x_inc, XMIN-2, YMIN-2)] = snd_buffer[ARRAY1D(index, 1)];
    }
}

__kernel void write_right_buffer_ocl_kernel(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double * restrict field,
    __global const double * restrict snd_buffer)
{
    int j = get_global_id(1);
    int k = get_global_id(0)-depth;

    if (k<=YMAX+y_inc+depth) {
        int index = j + (k+depth-1)*depth;
        field[ARRAY2D(XMAX+x_inc+j ,k, XMAX+4+x_inc, XMIN-2, YMIN-2)] = snd_buffer[ARRAY1D(index, 1)];
    }
}
