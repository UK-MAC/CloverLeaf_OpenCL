#include <kernel_files/macros_cl.cl>

__kernel void initialise_chunk_first
(const double d_xmin,
 const double d_ymin,
 const double d_dx,
 const double d_dy,
 __global       double * __restrict const vertexx,
 __global       double * __restrict const vertexdx,
 __global       double * __restrict const vertexy,
 __global       double * __restrict const vertexdy)
{
    __kernel_indexes;

    // fill out x arrays
    if (row == 0 && column <= (x_max + 1) + 3)
    {
        vertexx[column] = d_xmin +
            d_dx*(double)((((int)column) - 1) - x_min);
        vertexdx[column] = d_dx;
    }

    // fill out y arrays
    if (column == 0 && row <= (y_max + 1) + 3)
    {
        vertexy[row] = d_ymin +
            d_dy*(double)((((int)row) - 1) - y_min);
        vertexdy[row] = d_dy;
    }
}

__kernel void initialise_chunk_second
(const double d_xmin,
 const double d_ymin,
 const double d_dx,
 const double d_dy,
 __global const double * __restrict const vertexx,
 __global const double * __restrict const vertexdx,
 __global const double * __restrict const vertexy,
 __global const double * __restrict const vertexdy,
 __global       double * __restrict const cellx,
 __global       double * __restrict const celldx,
 __global       double * __restrict const celly,
 __global       double * __restrict const celldy,
 __global       double * __restrict const volume, 
 __global       double * __restrict const xarea, 
 __global       double * __restrict const yarea)
{
    __kernel_indexes;

    //fill x arrays
    if (row == 0 && column <= (x_max + 1) + 2)
    {
        cellx[column] = 0.5 * (vertexx[column] + vertexx[column + 1]);
        celldx[column] = d_dx;
    }

    //fill y arrays
    if (column == 0 && row <= (y_max + 1) + 2)
    {
        celly[row] = 0.5 * (vertexy[row] + vertexy[row + 1]);
        celldy[row] = d_dy;
    }

    if (/*row >= (y_min + 1) - 2 &&*/ row <= (y_max + 1) + 2
    && /*column >= (x_min + 1) - 2 &&*/ column <= (x_max + 1) + 2)
    {
        volume[THARR2D(0, 0, 0)] = d_dx * d_dy;
        xarea[THARR2D(0, 0, 1)] = d_dy;
        yarea[THARR2D(0, 0, 0)] = d_dx;
    }
}

