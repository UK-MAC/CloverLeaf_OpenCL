#include <kernel_files/macros_cl.cl>

__kernel void flux_calc_x
(double dt,
 __global const double * __restrict const xarea,
 __global const double * __restrict const xvel0,
 __global const double * __restrict const xvel1,
 __global       double * __restrict const vol_flux_x)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1)
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)
    {
        vol_flux_x[THARR2D(0, 0, 1)] = 0.25 * dt * xarea[THARR2D(0, 0, 1)]
            * (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]
            + xvel1[THARR2D(0, 0, 1)] + xvel1[THARR2D(0, 1, 1)]);
    }
}

__kernel void flux_calc_y
(double dt,
 __global const double * __restrict const yarea,
 __global const double * __restrict const yvel0,
 __global const double * __restrict const yvel1,
 __global       double * __restrict const vol_flux_y)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))
    {
        vol_flux_y[THARR2D(0, 0, 0)] = 0.25 * dt * yarea[THARR2D(0, 0, 0)]
            * (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]
            + yvel1[THARR2D(0, 0, 1)] + yvel1[THARR2D(1, 0, 1)]);
    }

}

