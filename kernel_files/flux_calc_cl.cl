__kernel void flux_calc
(double dt,
 __global const double * __restrict const xarea,
 __global const double * __restrict const yarea,
 __global const double * __restrict const zarea,
 __global const double * __restrict const xvel0,
 __global const double * __restrict const yvel0,
 __global const double * __restrict const zvel0,
 __global const double * __restrict const xvel1,
 __global const double * __restrict const yvel1,
 __global const double * __restrict const zvel1,
 __global double * __restrict const vol_flux_x,
 __global double * __restrict const vol_flux_y,
 __global double * __restrict const vol_flux_z)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1)
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1
    && /*slice >= (z_min + 1) &&*/ slice <= (z_max + 1))
    {
        vol_flux_x[THARR3D(0,0,0,1,0)]=0.125*dt*xarea[THARR3D(0,0,0,1,0)]
		*(xvel0[THARR3D(0,0,0,1,1)]+xvel0[THARR3D(0,1,0,1,1)]+xvel0[THARR3D(0,0,1,1,1)]+xvel0[THARR3D(0,1,1,1,1)]
		+xvel1[THARR3D(0,0,0,1,1)]+xvel1[THARR3D(0,1,0,1,1)]+xvel1[THARR3D(0,0,1,1,1)]+xvel1[THARR3D(0,1,1,1,1)]);

    }
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1)
    && /*slice >= (z_min + 1) &&*/ slice <= (z_max + 1))
    {
        vol_flux_y[THARR3D(0,0,0,0,1)]=0.125*dt*yarea[THARR3D(0,0,0,0,1)]
		*(yvel0[THARR3D(0,0,0,1,1)]+yvel0[THARR3D(1,0,0,1,1)]+yvel0[THARR3D(0,0,1,1,1)]+yvel0[THARR3D(1,0,1,1,1)]
		+yvel1[THARR3D(0,0,0,1,1)]+yvel1[THARR3D(1,0,0,1,1)]+yvel1[THARR3D(0,0,1,1,1)]+yvel1[THARR3D(1,0,1,1,1)]);
    }

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1)
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1)
    && /*slice >= (z_min + 1) &&*/ slice <= (z_max + 1)+1)
    {
        vol_flux_z[THARR3D(0,0,0,0,0)]=0.125*dt*zarea[THARR3D(0,0,0,0,0)]
		*(zvel0[THARR3D(0,0,0,1,1)]+zvel0[THARR3D(1,0,0,1,1)]+zvel0[THARR3D(1,0,0,1,1)]+zvel0[THARR3D(1,1,0,1,1)]
		+zvel1[THARR3D(0,0,0,1,1)]+zvel1[THARR3D(1,0,0,1,1)]+zvel1[THARR3D(0,1,0,1,1)]+zvel1[THARR3D(1,1,0,1,1)]);
    }

}
