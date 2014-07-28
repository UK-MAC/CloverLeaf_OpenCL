#include "./kernel_files/macros_cl.cl"
#include <kernel_files/macros_cl.cl>

__kernel void update_halo_bottom
(int x_extra,   int y_extra, int z_extra,
 int x_invert,  int y_invert, int z_invert,
 int x_face,    int y_face, int z_face,
 int grid_type, int depth, 
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

    // offset by 1 if it is anything but a CELL grid
    int b_offset = (grid_type != CELL_DATA) ? 1 : 0;
  if (slice >= 2 - depth && slice <= (z_max + 1) + z_extra + depth)
  {
    if (column >= 2 - depth && column <= (x_max + 1) + x_extra + depth)
    {
        if (row < depth)
        {
            const int offset = 2 + b_offset;

            /*
             * 1 - 2 * row means that row 0 services row 1, and vice versa
             * this means that it can be dispatched with 'depth' rows only
             */
            cur_array[THARR3D(0, 1 - (2 * row),0, x_extra,y_extra)] =
                y_invert * cur_array[THARR3D(0, offset,0, x_extra,y_extra)];
        }
    }
  }
}

__kernel void update_halo_top
(int x_extra, int y_extra, int z_extra,
 int x_invert, int y_invert, int z_invert,
 int x_face, int y_face, int z_face,
 int grid_type, int depth, 
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

    // if x face data, offset source/dest by - 1
    int x_f_offset = (x_face || z_face);
  if (slice >= 2 - depth && slice <= (z_max + 1) + z_extra + depth)
  {
    if (column >= 2 - depth && column <= (x_max + 1) + x_extra + depth)
    {
        if (row < depth)
        {
            const int offset = (- row) * 2 - 1 - x_f_offset;

            cur_array[THARR3D(0, y_extra + (y_max + 2),0, x_extra,y_extra)] =
                y_invert * cur_array[THARR3D(0, (y_max + 2) + offset,0, x_extra,y_extra)];
        }
    }
  }
}

__kernel void update_halo_left
(int x_extra, int y_extra, int z_extra,
 int x_invert, int y_invert, int z_invert,
 int x_face, int y_face, int z_face,
 int grid_type, int depth, 
 __global double * __restrict const cur_array)
{
    // offset by 1 if it is anything but a CELL grid
    int l_offset = (grid_type != CELL_DATA) ? 1 : 0;

    __kernel_indexes;
  if (slice >= 2 - depth && slice <= (z_max + 1) + z_extra + depth)
  {
    if (column < depth)
    if (row >= 2 - depth && row <= (y_max + 1) + y_extra + depth)
    {
        // first in row
        const int row_begin = row * (x_max + 4 + x_extra) +
            (slice)*(x_max + 4 + x_extra)*(y_max + 4 + y_extra);

        cur_array[row_begin + (1 - column)] =
            x_invert * cur_array[row_begin + 2 + column + l_offset];
    }
  }
}

__kernel void update_halo_right
(int x_extra, int y_extra, int z_extra,
 int x_invert, int y_invert, int z_invert,
 int x_face, int y_face, int z_face,
 int grid_type, int depth, 
 __global double * __restrict const cur_array)
{
    // offset source by -1 if its a y face
    int y_f_offset = (y_face || z_face);

    __kernel_indexes;
  if (slice >= 2 - depth && slice <= (z_max + 1) + z_extra + depth)
  {
    if (column < depth)
    if (row >= 2 - depth && row <= (y_max + 1) + y_extra + depth)
    {
        const int row_begin = row * (x_max + 4 + x_extra) +
            (slice)*(x_max + 4 + x_extra)*(y_max + 4 + y_extra);

        cur_array[row_begin + x_max + 2 + x_extra + column] =
            x_invert * cur_array[row_begin + x_max + 1 - (column + y_f_offset)];
    }
  }
}

__kernel void update_halo_back
(int x_extra, int y_extra, int z_extra,
 int x_invert, int y_invert, int z_invert,
 int x_face, int y_face, int z_face,
 int grid_type, int depth, 
 __global double * __restrict const cur_array)
{
    // offset by 1 if it is anything but a CELL grid
    int z_offset = (grid_type != CELL_DATA) ? 1 : 0;

    __kernel_indexes;
  if (column >= 2 - depth && column <= (x_max + 1) + x_extra + depth)
  {
    if (row >= 2 - depth && row <= (y_max + 1) + y_extra + depth)
    {
        cur_array[THARR3D(0, 0, 1 - (2 * slice), x_extra,y_extra)] =
            z_invert * cur_array[THARR3D(0, 0, 2 + z_offset, x_extra,y_extra)];
    }
  }
}

__kernel void update_halo_front
(int x_extra, int y_extra, int z_extra,
 int x_invert, int y_invert, int z_invert,
 int x_face, int y_face, int z_face,
 int grid_type, int depth, 
 __global double * __restrict const cur_array)
{
    __kernel_indexes;

  int z_offset = (x_face || y_face);

  if (column >= 2 - depth && column <= (x_max + 1) + x_extra + depth)
  {
    if (row >= 2 - depth && row <= (y_max + 1) + y_extra + depth)
    {
        const int offset = (- slice) * 2 - 1 - z_offset;

        cur_array[THARR3D(0, 0, (z_max + 2) + z_extra, x_extra,y_extra)] =
            z_invert * cur_array[THARR3D(0, 0, (z_max + 2) + offset, x_extra,y_extra)];
    }
  }
}

