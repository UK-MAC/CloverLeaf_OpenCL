#include <kernel_files/macros_cl.cl>

__kernel void update_halo_bottom_BARE
(int depth,
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

    // offset by 1 if it is anything but a CELL grid
    int b_offset = (GRID_TYPE != CELL_DATA) ? 1 : 0;

    if (column >= 2 - depth && column <= (x_max + 1) + X_EXTRA + depth)
    {
        if (row < depth)
        {
            const int offset = 2 + b_offset;

            /*
             * 1 - 2 * row means that row 0 services row 1, and vice versa
             * this means that it can be dispatched with 'depth' rows only
             */
            cur_array[THARR2D(0, 1 - (2 * row), X_EXTRA)] =
                Y_INVERT * cur_array[THARR2D(0, offset, X_EXTRA)];
        }
    }
}

__kernel void update_halo_top_BARE
(int depth,
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

    // if x face data, offset source/dest by - 1
    int x_f_offset = (X_FACE) ? 1 : 0;

    if (column >= 2 - depth && column <= (x_max + 1) + X_EXTRA + depth)
    {
        if (row < depth)
        {
            const int offset = (- row) * 2 - 1 - x_f_offset;

            cur_array[THARR2D(0, Y_EXTRA + y_max + 2, X_EXTRA)] =
                Y_INVERT * cur_array[THARR2D(0, y_max + 2 + offset, X_EXTRA)];
        }
    }
}

__kernel void update_halo_left_BARE
(int depth,
 __global double * __restrict const cur_array)
{
    // offset by 1 if it is anything but a CELL grid
    int l_offset = (GRID_TYPE != CELL_DATA) ? 1 : 0;

    // special indexes for specific depth
    //const int glob_id = threadIdx.x + blockIdx.x * blockDim.x;
    //const int row = glob_id / depth;
    //const int column = glob_id % depth;
    __kernel_indexes;

    if (row >= 2 - depth && row <= (y_max + 1) + Y_EXTRA + depth)
    {
        // first in row
        const int row_begin = row * (x_max + 4 + X_EXTRA);

        cur_array[row_begin + (1 - column)] = X_INVERT * cur_array[row_begin + 2 + column + l_offset];
    }
}

__kernel void update_halo_right_BARE
(int depth,
 __global double * __restrict const cur_array)
{
    // offset source by -1 if its a y face
    int y_f_offset = (Y_FACE) ? 1 : 0;

    //const int glob_id = threadIdx.x + blockIdx.x * blockDim.x;
    //const int row = glob_id / depth;
    //const int column = glob_id % depth;
    __kernel_indexes;

    if (row >= 2 - depth && row <= (y_max + 1) + Y_EXTRA + depth)
    {
        const int row_begin = row * (x_max + 4 + X_EXTRA);

        cur_array[row_begin + x_max + 2 + X_EXTRA + column] = X_INVERT * cur_array[row_begin + x_max + 1 - (column + y_f_offset)];
    }
}

