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
 *  @brief OCL device-side comms buffer unpacking kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Unpack the comms buffers on the target device
 */

#include "ocl_knls.h"

__kernel void left_comm_buffer_unpack(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double * restrict field,
    __global const double * restrict rcv_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0);
    int index; 

    if ((k>=2-depth) && (k<=YMAXPLUSONE+y_inc+depth)) {

        index = j + (k+depth-2)*depth; 

        field[ ARRAYXY(XMIN-j, k, XMAXPLUSFOUR+x_inc) ] = rcv_buffer[index];        
    }
}

__kernel void right_comm_buffer_unpack(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double * restrict field,
    __global const double * restrict rcv_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0);
    int index; 

    if ((k>=2-depth) && (k<=YMAXPLUSONE+y_inc+depth)) {

        index = j + (k+depth-2)*depth; 

        field[ ARRAYXY(XMAXPLUSTWO+x_inc+j, k, XMAXPLUSFOUR+x_inc) ] = rcv_buffer[index];        
    }
}

__kernel void top_comm_buffer_unpack(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double * restrict field,
    __global const double * restrict rcv_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0);
    int index; 

    if ( (j>=2-depth) && (j<=XMAXPLUSONE+x_inc+depth) ) {

        index = j - (2-depth) + k*(XMAX+x_inc+(2*depth)); 

        field[ ARRAYXY(j, YMAXPLUSTWO+y_inc+k, XMAXPLUSFOUR+x_inc) ] = rcv_buffer[index];        
    }
}

__kernel void bottom_comm_buffer_unpack(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double * restrict field,
    __global const double * restrict rcv_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0);
    int index; 

    if ( (j>=2-depth) && (j<=XMAXPLUSONE+x_inc+depth) ) {

        index = j - (2-depth) + k*(XMAX+x_inc+(2*depth)); 

        field[ ARRAYXY(j, YMIN-k, XMAXPLUSFOUR+x_inc) ] = rcv_buffer[index];        
    }
}
