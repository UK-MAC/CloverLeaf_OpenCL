#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#define ARRAYXY(x_index, y_index, x_width) ((y_index)*(x_width)+(x_index))


__kernel void left_comm_buffer_pack(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double *field,
    __global doube *snd_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0);
    int index; 

    if ((k>=2-depth) && (k<=YMAXPLUSONE+y_inc+depth)) {

        index = j + (k+depth-2)*depth; 

        snd_buffer[index] = field[ ARRAYXY( XMINPLUSONE+x_inc+j, k, XMAXPLUSFOUR+x_inc ) ]; 

    }
}

__kernel void right_comm_buffer_pack(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double *field,
    __global doube *snd_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0);
    int index; 

    if ((k>=2-depth) && (k<=YMAXPLUSONE+y_inc+depth)) {

        index = j + (k+depth-2)*depth; 

        snd_buffer[index] = field[ ARRAYXY( XMAXPLUSONE-j, k, XMAXPLUSFOUR+x_inc ) ]; 

    }
}

__kernel void top_comm_buffer_pack(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double *field,
    __global doube *snd_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0);
    int index; 

    if ( (j>=2-depth) && (j<=XMAXPLUSONE+x_inc+depth) ) {

        //need to check this 
        index = j - (2-depth) + k*(XMAX+x_inc+(2*depth)); 

        snd_buffer[index] = field[ ARRAYXY( j, YMAXPLUSONE-k, XMAXPLUSFOUR+x_inc ) ]; 

    }
}

__kernel void bottom_comm_buffer_pack(
    const int depth,
    const int x_inc,
    const int y_inc,
    __global double *field,
    __global doube *snd_buffer)
{
    int k = get_global_id(1);
    int j = get_global_id(0);
    int index; 

    if ( (j>=2-depth) && (j<=XMAXPLUSONE+x_inc+depth) ) {

        //need to check this 
        index = j - (2-depth) + k*(XMAX+x_inc+(2*depth)); 

        snd_buffer[index] = field[ ARRAYXY( j, YMINPLUSONE+y_inc+k, XMAXPLUSFOUR+x_inc) ]; 

    }
}
