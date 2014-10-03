#include "./kernel_files/macros_cl.cl"
__kernel void field_summary
(__global const double * __restrict const volume,
 __global const double * __restrict const density0,
 __global const double * __restrict const energy0,
 __global const double * __restrict const pressure,
 __global const double * __restrict const xvel0,
 __global const double * __restrict const yvel0,
 __global const double * __restrict const zvel0,

 __global double * __restrict const vol,
 __global double * __restrict const mass,
 __global double * __restrict const ie,
 __global double * __restrict const ke,
 __global double * __restrict const press)
{
    __kernel_indexes;

    __local double vol_shared[BLOCK_SZ];
    __local double mass_shared[BLOCK_SZ];
    __local double ie_shared[BLOCK_SZ];
    __local double ke_shared[BLOCK_SZ];
    __local double press_shared[BLOCK_SZ];
    vol_shared[lid] = 0.0;
    mass_shared[lid] = 0.0;
    ie_shared[lid] = 0.0;
    ke_shared[lid] = 0.0;
    press_shared[lid] = 0.0;

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1)
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1)
    && /*slice >= (z_min + 1) &&*/ slice <= (z_max + 1))
    {
        double vsqrd = 0.0;

        //unrolled do loop
        vsqrd += 0.125 *(xvel0[THARR3D(0, 0,0, 1,1)] * xvel0[THARR3D(0 ,0,0, 1,1)]
                        +yvel0[THARR3D(0, 0,0, 1,1)] * yvel0[THARR3D(0, 0,0, 1,1)]
                        +zvel0[THARR3D(0, 0,0, 1,1)] * zvel0[THARR3D(0, 0,0, 1,1)]);

        vsqrd += 0.125 *(xvel0[THARR3D(1, 0,0, 1,1)] * xvel0[THARR3D(1, 0,0, 1,1)]
                        +yvel0[THARR3D(1, 0,0, 1,1)] * yvel0[THARR3D(1, 0,0, 1,1)]
                        +zvel0[THARR3D(1, 0,0, 1,1)] * zvel0[THARR3D(1, 0,0, 1,1)]);

        vsqrd += 0.125 *(xvel0[THARR3D(0, 1,0, 1,1)] * xvel0[THARR3D(0, 1,0, 1,1)]
                        +yvel0[THARR3D(0, 1,0, 1,1)] * yvel0[THARR3D(0, 1,0, 1,1)]
                        +zvel0[THARR3D(0, 1,0, 1,1)] * zvel0[THARR3D(0, 1,0, 1,1)]);

        vsqrd += 0.125 *(xvel0[THARR3D(1, 1,0, 1,1)] * xvel0[THARR3D(1, 1,0, 1,1)]
                        +yvel0[THARR3D(1, 1,0, 1,1)] * yvel0[THARR3D(1, 1,0, 1,1)]
                        +zvel0[THARR3D(1, 1,0, 1,1)] * zvel0[THARR3D(1, 1,0, 1,1)]);

	//for z

        vsqrd += 0.125 *(xvel0[THARR3D(0, 0,1, 1,1)] * xvel0[THARR3D(0, 0,1, 1,1)]
                        +yvel0[THARR3D(0, 0,1, 1,1)] * yvel0[THARR3D(0, 0,1, 1,1)]
                        +zvel0[THARR3D(0, 0,1, 1,1)] * zvel0[THARR3D(0, 0,1, 1,1)]);

        vsqrd += 0.125 *(xvel0[THARR3D(1, 0,1, 1,1)] * xvel0[THARR3D(1, 0,1, 1,1)]
                        +yvel0[THARR3D(1, 0,1, 1,1)] * yvel0[THARR3D(1, 0,1, 1,1)]
                        +zvel0[THARR3D(1, 0,1, 1,1)] * zvel0[THARR3D(1, 0,1, 1,1)]);

        vsqrd += 0.125 *(xvel0[THARR3D(0, 1,1, 1,1)] * xvel0[THARR3D(0, 1,1, 1,1)]
                        +yvel0[THARR3D(0, 1,1, 1,1)] * yvel0[THARR3D(0, 1,1, 1,1)]
                        +zvel0[THARR3D(0, 1,1, 1,1)] * zvel0[THARR3D(0, 1,1, 1,1)]);

        vsqrd += 0.125 *(xvel0[THARR3D(1, 1,1, 1,1)] * xvel0[THARR3D(1, 1,1, 1,1)]
                        +yvel0[THARR3D(1, 1,1, 1,1)] * yvel0[THARR3D(1, 1,1, 1,1)]
                        +zvel0[THARR3D(1, 1,1, 1,1)] * zvel0[THARR3D(1, 1,1, 1,1)]);

        const double cell_vol = volume[THARR3D(0, 0, 0,0,0)];
        const double cell_mass = cell_vol * density0[THARR3D(0, 0, 0,0,0)];

        vol_shared[lid] = cell_vol;
        mass_shared[lid] = cell_mass;
        ie_shared[lid] = cell_mass * energy0[THARR3D(0, 0, 0,0,0)];
        ke_shared[lid] = cell_mass * 0.5 * vsqrd;
        press_shared[lid] = cell_vol * pressure[THARR3D(0, 0, 0,0,0)];
    }

    REDUCTION(vol_shared, vol, SUM)
    REDUCTION(mass_shared, mass, SUM)
    REDUCTION(ie_shared, ie, SUM)
    REDUCTION(ke_shared, ke, SUM)
    REDUCTION(press_shared, press, SUM)
}
