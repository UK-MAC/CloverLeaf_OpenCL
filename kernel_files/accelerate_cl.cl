#include <kernel_files/macros_cl.cl>

__kernel void accelerate
(double dbyt,
 __global const double * const __restrict xarea,
 __global const double * const __restrict yarea,
 __global const double * const __restrict volume,
 __global const double * const __restrict density0,
 __global const double * const __restrict pressure,
 __global const double * const __restrict viscosity,
 __global const double * const __restrict xvel0,
 __global const double * const __restrict yvel0,
 __global       double * const __restrict xvel1,
 __global       double * const __restrict yvel1)
{
    __kernel_indexes;

    double nodal_mass, step_by_mass;

    // prevent writing to *vel1, then read from it, then write to it again
    double xvel_temp, yvel_temp;

    if(/*row >= (x_min + 1) &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)
    {
        nodal_mass =
            (density0[THARR2D(-1, -1, 0)] * volume[THARR2D(-1, -1, 0)]
            + density0[THARR2D(0, -1, 0)] * volume[THARR2D(0, -1, 0)]
            + density0[THARR2D(0, 0, 0)] * volume[THARR2D(0, 0, 0)]
            + density0[THARR2D(-1, 0, 0)] * volume[THARR2D(-1, 0, 0)])
            * 0.25;

        step_by_mass = 0.5 * dbyt / nodal_mass;

        // x velocities
        xvel_temp = xvel0[THARR2D(0, 0, 1)] - step_by_mass
            * (xarea[THARR2D(0, 0, 1)] * (pressure[THARR2D(0, 0, 0)]  - pressure[THARR2D(-1, 0, 0)])
            + xarea[THARR2D(0, -1, 1)] * (pressure[THARR2D(0, -1, 0)] - pressure[THARR2D(-1, -1, 0)]));

        xvel1[THARR2D(0, 0, 1)] = xvel_temp - step_by_mass
            * (xarea[THARR2D(0, 0, 1)] * (viscosity[THARR2D(0, 0, 0)]  - viscosity[THARR2D(-1, 0, 0)])
            + xarea[THARR2D(0, -1, 1)] * (viscosity[THARR2D(0, -1, 0)] - viscosity[THARR2D(-1, -1, 0)]));

        // y velocities
        yvel_temp = yvel0[THARR2D(0, 0, 1)] - step_by_mass
            * (yarea[THARR2D(0, 0, 0)] * (pressure[THARR2D(0, 0, 0)]  - pressure[THARR2D(0, -1, 0)])
            + yarea[THARR2D(-1, 0, 0)] * (pressure[THARR2D(-1, 0, 0)] - pressure[THARR2D(-1, -1, 0)]));

        yvel1[THARR2D(0, 0, 1)] = yvel_temp - step_by_mass
            * (yarea[THARR2D(0, 0, 0)] * (viscosity[THARR2D(0, 0, 0)]  - viscosity[THARR2D(0, -1, 0)])
            + yarea[THARR2D(-1, 0, 0)] * (viscosity[THARR2D(-1, 0, 0)] - viscosity[THARR2D(-1, -1, 0)]));

    }
    
}
