#include "./kernel_files/macros_cl.cl"

// left/right
#if 1
#define VERT_IDX                                                    \
    ((column - 1) +                                                 \
    ((row    - 1) + depth - 1)*depth +                              \
    ((slice  - 1) + depth - 1)*(y_max + y_extra + 2*depth)*depth)
#else
#define VERT_IDX                                            \
    (slice  + depth - 1 +                                                           \
    (row    + depth - 1)* (x_max + x_extra + 2*depth) +                             \
    (column         - 1)*((z_max + z_extra + 2*depth)*(y_max + y_extra + 2*depth)))
#endif

// bottom/top
#define HORZ_IDX                                                                    \
    (column + depth - 1 +                                                           \
    (slice  + depth - 1)* (x_max + x_extra + 2*depth) +                             \
    (row            - 1)*((x_max + x_extra + 2*depth)*(z_max + z_extra + 2*depth)))

// back/front
#define DEPTH_IDX                                                                   \
    (row    + depth - 1 +                                                           \
    (column + depth - 1)* (x_max + x_extra + 2*depth) +                             \
    (slice          - 1)*((x_max + x_extra + 2*depth)*(y_max + y_extra + 2*depth)))

__kernel void pack_left_buffer
(int x_extra, int y_extra, int z_extra,
const  __global double * __restrict array,
       __global double * __restrict left_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice >= (z_min + 1) - depth && slice <= (z_max + 1) + z_extra + depth)
    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    if (column < depth)
    {
        left_buffer[VERT_IDX] = array[THARR3D((x_min + 1) + x_extra, 0, 0, x_extra, y_extra)];
    }
}

__kernel void unpack_left_buffer
(int x_extra, int y_extra, int z_extra,
       __global double * __restrict array,
const  __global double * __restrict left_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice >= (z_min + 1) - depth && slice <= (z_max + 1) + z_extra + depth)
    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    if (column < depth)
    {
        /*
         *  offset in column 0 =  1
         *  offset in column 1 = -1
         *  'swaps' column destination
         */
        array[THARR3D(1 - 2*column, 0, 0, x_extra, y_extra)] = left_buffer[VERT_IDX];
    }
}

/************************************************************/

__kernel void pack_right_buffer
(int x_extra, int y_extra, int z_extra,
const  __global double * __restrict array,
       __global double * __restrict right_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice >= (z_min + 1) - depth && slice <= (z_max + 1) + z_extra + depth)
    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    if (column < depth)
    {
        right_buffer[VERT_IDX] = array[THARR3D((x_max + 1) - 2*column, 0, 0, x_extra, y_extra)];
    }
}

__kernel void unpack_right_buffer
(int x_extra, int y_extra, int z_extra,
       __global double * __restrict array,
const  __global double * __restrict right_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice >= (z_min + 1) - depth && slice <= (z_max + 1) + z_extra + depth)
    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    if (column < depth)
    {
        array[THARR3D((x_max + 1) + x_extra + 1, 0, 0, x_extra, y_extra)] = right_buffer[VERT_IDX];
    }
}

/************************************************************/

__kernel void pack_bottom_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict array,
 __global double * __restrict bottom_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice >= (z_min + 1) - depth && slice <= (z_max + 1) + z_extra + depth)
    if (row < depth)
    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        bottom_buffer[HORZ_IDX] = array[THARR3D(0, (y_min + 1) + y_extra, 0, x_extra, y_extra)];
    }
}

__kernel void unpack_bottom_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict array,
 __global double * __restrict bottom_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice >= (z_min + 1) - depth && slice <= (z_max + 1) + z_extra + depth)
    if (row < depth)
    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        array[THARR3D(0, 1 - 2*row, 0, x_extra, y_extra)] = bottom_buffer[HORZ_IDX];
    }
}

/************************************************************/

__kernel void pack_top_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict array,
 __global double * __restrict top_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice >= (z_min + 1) - depth && slice <= (z_max + 1) + z_extra + depth)
    if (row < depth)
    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        top_buffer[HORZ_IDX] = array[THARR3D(0, (y_max + 1) - 2*row, 0, x_extra, y_extra)];
    }
}

__kernel void unpack_top_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict array,
 __global double * __restrict top_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice >= (z_min + 1) - depth && slice <= (z_max + 1) + z_extra + depth)
    if (row < depth)
    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        array[THARR3D(0, (y_max + 1) + y_extra + 1, 0, x_extra, y_extra)] = top_buffer[HORZ_IDX];
    }
}

/************************************************************/

__kernel void pack_back_buffer
(int x_extra, int y_extra, int z_extra,
 const __global double * __restrict array,
 __global double * __restrict back_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice < depth)
    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        back_buffer[DEPTH_IDX] = array[THARR3D(0, 0, (z_min + 1) + z_extra, x_extra, y_extra)];
    }
}

__kernel void unpack_back_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict array,
 const __global double * __restrict back_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice < depth)
    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        array[THARR3D(0, 0, 1 - 2*slice, x_extra, y_extra)] = back_buffer[DEPTH_IDX];
    }
}

/************************************************************/

__kernel void pack_front_buffer
(int x_extra, int y_extra, int z_extra,
 const __global double * __restrict array,
 __global double * __restrict front_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice < depth)
    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        front_buffer[DEPTH_IDX] = array[THARR3D(0, 0, (z_max + 1) - 2*slice, x_extra, y_extra)];
    }
}

__kernel void unpack_front_buffer
(int x_extra, int y_extra, int z_extra,
 __global double * __restrict array,
 const __global double * __restrict front_buffer,
const int depth)
{
    __kernel_indexes;

    if (slice < depth)
    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        array[THARR3D(0, 0, (z_max + 1) + z_extra + 1, x_extra, y_extra)] = front_buffer[DEPTH_IDX];
    }
}
