#include "./kernel_files/macros_cl.cl"

// left/right
#if 1
#define VERT_IDX                                            \
    (column         - 2 +                                   \
    (row    + depth - 1)*depth +                            \
    (slice  + depth - 2)*(y_max + y_extra + 2*depth)*depth)
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
#if 0
        const int row_begin = row * (x_max + 4 + x_extra) +
            (slice)*(x_max + 4 + x_extra)*(y_max + 4 + y_extra);

        left_buffer[VERT_IDX] =
            array[row_begin + (x_min + 1) + x_extra - 1 + (1 + column)];
#else
        left_buffer[VERT_IDX] =
            array[THARR3D((x_min + 1) + x_extra - 1 + (1 + column),
            0, 0, x_extra, y_extra)];
#endif
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
#if 0
        const int row_begin = row * (x_max + 4 + x_extra) +
            (slice)*(x_max + 4 + x_extra)*(y_max + 4 + y_extra);

        array[row_begin + (x_min + 1) - (1 + column)] = left_buffer[VERT_IDX];
#else
        array[THARR3D((x_min + 1) - (1 + column), 0, 0, x_extra, y_extra)] =
            left_buffer[VERT_IDX];
#endif
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
#if 0
        const int row_begin = row * (x_max + 4 + x_extra) +
            (slice)*(x_max + 4 + x_extra)*(y_max + 4 + y_extra);

        right_buffer[VERT_IDX] = array[row_begin + (x_max + 1) + 1 - (1 + column)];
#else
        right_buffer[VERT_IDX] =
            array[THARR3D((x_max + 1) + 1 - (1 + column), 0, 0, x_extra, y_extra)];
#endif
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
#if 0
        const int row_begin = row * (x_max + 4 + x_extra) +
            (slice)*(x_max + 4 + x_extra)*(y_max + 4 + y_extra);

        array[row_begin + (x_max + 1) + x_extra + (1 + column)] = right_buffer[VERT_IDX];
#else
        array[THARR3D((x_max + 1) + x_extra + (1 + column), 0, 0, x_extra, y_extra)] =
            right_buffer[VERT_IDX];
#endif
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
        bottom_buffer[HORZ_IDX] = array[THARR3D(0, (y_min + 1) + y_extra - 1 + row + 1, 0, x_extra, y_extra)];
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
        array[THARR3D(0, (y_min + 1) - (1 + 2*row), 0, x_extra, y_extra)] = bottom_buffer[HORZ_IDX];
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
        top_buffer[HORZ_IDX] = array[THARR3D(0, (y_max + 1) + 1 - (1 + 2*row), 0, x_extra, y_extra)];
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
        array[THARR3D(0, (y_max + 1) + y_extra + row + 1, 0, x_extra, y_extra)] = top_buffer[HORZ_IDX];
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
        back_buffer[DEPTH_IDX] = array[THARR3D(0, 0, (z_min + 1) + z_extra - 1 + (slice + 1), x_extra, y_extra)];
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
        array[THARR3D(0, 0, (z_min + 1) - (1 + slice), x_extra, y_extra)] = back_buffer[DEPTH_IDX];
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
        front_buffer[DEPTH_IDX] = array[THARR3D(0, 0, (z_max + 1) + 1 - (2 + slice), x_extra, y_extra)];
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
        array[THARR3D(0, 0, (z_max + 1) + z_extra + (slice + 0), x_extra, y_extra)] = front_buffer[DEPTH_IDX];
    }
}
