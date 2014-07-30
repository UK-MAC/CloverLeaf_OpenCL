#include "./kernel_files/macros_cl.cl"

/********************/

// j is column
// k is row

// could put this check in, prob doesnt need it
// if (row > 1 - depth && row < y_max + 2 + depth + y_extra)

// left/right buffer
// index=j+(k+depth-1)*depth

// left index 
// left_snd_buffer(index)=field(chunks(chunk)%field%x_min+x_extra-1+j,k)
// field(chunks(chunk)%field%x_min-j,k)=left_rcv_buffer(index)

// right index
// right_snd_buffer(index)=field(chunks(chunk)%field%x_max+1-j,k)
// field(chunks(chunk)%field%x_max+x_extra+j,k)=right_rcv_buffer(index)

/********************/

// top/bottom buffer
// index=j+depth+(k-1)*(chunks(chunk)%field%x_max+x_extra+(2*depth))

// bottom index
// bottom_snd_buffer(index)=field(j,chunks(chunk)%field%y_min+y_extra-1+k)
// field(j,chunks(chunk)%field%y_min-k)=bottom_rcv_buffer(index)

// top index
// top_snd_buffer(index)=field(j,chunks(chunk)%field%y_max+1-k)
// field(j,chunks(chunk)%field%y_max+y_extra+k)=top_rcv_buffer(index)

/********************/

// for top/bottom
#define HORZ_IDX(add) (column + depth + ((add + 1) - 1)*((x_max + 1) + x_extra + (2 * depth)) - 2)
// for left/right
#define VERT_IDX(add) ((add) + (row + depth - 1)*depth - 2)

__kernel void pack_left_buffer
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
const  __global double * __restrict array,
       __global double * __restrict left_buffer,
const int depth)
{
    __kernel_indexes;

    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    {
        const int row_begin = row * (x_max + 4 + x_extra);

        left_buffer[VERT_IDX(column)] = array[row_begin + (x_min + 1) + x_extra - 1 + (1 + column)];
    }
}

__kernel void unpack_left_buffer
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
       __global double * __restrict array,
const  __global double * __restrict left_buffer,
const int depth)
{
    __kernel_indexes;

    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    {
        const int row_begin = row * (x_max + 4 + x_extra);

        array[row_begin + (x_min + 1) - (1 + column)] = left_buffer[VERT_IDX(column)];
    }
}

/************************************************************/

__kernel void pack_right_buffer
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
const  __global double * __restrict array,
       __global double * __restrict right_buffer,
const int depth)
{
    __kernel_indexes;

    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    {
        const int row_begin = row * (x_max + 4 + x_extra);

        right_buffer[VERT_IDX(column)] = array[row_begin + (x_max + 1) + 1 - (1 + column)];
    }
}

__kernel void unpack_right_buffer
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
       __global double * __restrict array,
const  __global double * __restrict right_buffer,
const int depth)
{
    __kernel_indexes;

    if (row >= (y_min + 1) - depth && row <= (y_max + 1) + y_extra + depth)
    {
        const int row_begin = row * (x_max + 4 + x_extra);

        array[row_begin + (x_max + 1) + x_extra + 1 + column] = right_buffer[VERT_IDX(column)];
    }
}

/************************************************************/

__kernel void pack_bottom_buffer
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
 __global double * __restrict array,
 __global double * __restrict bottom_buffer,
const int depth)
{
    __kernel_indexes;

    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        if (row < depth)
        {
            bottom_buffer[HORZ_IDX(row)] = array[THARR2D(0, (y_min + 1) + y_extra - 1 + row + 1, x_extra)];
        }
    }
}

__kernel void unpack_bottom_buffer
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
 __global double * __restrict array,
 __global double * __restrict bottom_buffer,
const int depth)
{
    __kernel_indexes;

    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        if (row < depth)
        {
            array[THARR2D(0, (y_min + 1) - (1 + 2*row), x_extra)] = bottom_buffer[HORZ_IDX(row)];
        }
    }
}

/************************************************************/

__kernel void pack_top_buffer
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
 __global double * __restrict array,
 __global double * __restrict top_buffer,
const int depth)
{
    __kernel_indexes;

    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        if (row < depth)
        {
            top_buffer[HORZ_IDX(row)] = array[THARR2D(0, (y_max + 1) + 1 - (1 + 2*row), x_extra)];
        }
    }
}

__kernel void unpack_top_buffer
(int x_extra, int y_extra,
 int x_invert, int y_invert,
 int x_face, int y_face,
 __global double * __restrict array,
 __global double * __restrict top_buffer,
const int depth)
{
    __kernel_indexes;

    if (column >= (x_min + 1) - depth && column <= (x_max + 1) + x_extra + depth)
    {
        if (row < depth)
        {
            array[THARR2D(0, (y_max + 1) + y_extra + row + 1, x_extra)] = top_buffer[HORZ_IDX(row)];
        }
    }
}

