#include "ocl_knls.h"

__kernel void left_comm_buffer_unpack(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double *field,
    __global double *rcv_buffer)
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
    __global double *field,
    __global double *rcv_buffer)
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
    __global double *field,
    __global double *rcv_buffer)
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
    __global double *field,
    __global double *rcv_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0);
    int index; 

    if ( (j>=2-depth) && (j<=XMAXPLUSONE+x_inc+depth) ) {

        index = j - (2-depth) + k*(XMAX+x_inc+(2*depth)); 

        field[ ARRAYXY(j, YMIN-k, XMAXPLUSFOUR+x_inc) ] = rcv_buffer[index];        
    }
}
