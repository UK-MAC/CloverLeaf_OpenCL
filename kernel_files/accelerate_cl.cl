__kernel void accelerate
(double dbyt,
 __global const double * const __restrict xarea,
 __global const double * const __restrict yarea,
 __global const double * const __restrict zarea,
 __global const double * const __restrict volume,
 __global const double * const __restrict density0,
 __global const double * const __restrict pressure,
 __global const double * const __restrict viscosity,
 __global const double * const __restrict xvel0,
 __global const double * const __restrict yvel0,
 __global const double * const __restrict zvel0,
 __global double * const __restrict xvel1,
 __global double * const __restrict yvel1,
 __global double * const __restrict zvel1)
{
    __kernel_indexes;

    double nodal_mass, step_by_mass;

    // prevent writing to *vel1, then read from it, then write to it again
    double xvel_temp, yvel_temp, zvel_temp;

    if (/*row >= (y_min + 1) &&*/ row <= (y_max + 1)
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1)
    && /*slice >= (z_min + 1) &&*/ slice <= (z_max + 1))
    {
	nodal_mass=(density0(-1,-1,0,0,0 )*volume(-1,-1,0,0,0 )
                   +density0(0 ,-1,0,0,0 )*volume(0 ,-1,0,0,0 )
                   +density0(0 ,0 ,0,0,0 )*volume(0 ,0 ,0,0,0 )
                   +density0(-1,0 ,0,0,0 )*volume(-1,0 ,0,0,0 )
                   +density0(-1,-1,-1,0,0)*volume(-1,-1,-1,0,0)
                   +density0(0 ,-1,-1,0,0)*volume(0 ,-1,-1,0,0)
                   +density0(0 ,0 ,-1,0,0)*volume(0 ,0 ,-1,0,0)
                   +density0(-1,0 ,-1,0,0)*volume(-1,0 ,-1,0,0))
                   *0.125;

        step_by_mass = 0.25 * dbyt / nodal_mass;

        // x velocities
        xvel_temp=xvel0[THARR3D(0,0,0,1,1)]-step_by_mass*(xarea[THARR3D(0 ,0 ,0,1,0 )]*(pressure[THARR3D(0 ,0 ,0,0,0 )]-pressure[THARR3D(-1,0 ,0,0,0 )])
		+xarea[THARR3D(0 ,-1,0,1,0 )]*(pressure[THARR3D(0 ,-1,0,0,0 )]-pressure[THARR3D(-1,-1,0,0,0 )])
		+xarea[THARR3D(0 ,0 ,-1,1,0)]*(pressure[THARR3D(0 ,0 ,-1,0,0)]-pressure[THARR3D(-1,0 ,-1,0,0)])
		+xarea[THARR3D(0 ,-1,-1,1,0)]*(pressure[THARR3D(0 ,-1,-1,0,0)]-pressure[THARR3D(-1,-1,-1,0,0)]));

        xvel1[THARR3D(0,0,0,1,1)]=xvel_temp-step_by_mass*(xarea[THARR3D(0 ,0 ,0,1,0 )]*(viscosity[THARR3D(0 ,0 ,0,0,0 )]-viscosity[THARR3D(-1,0 ,0,0,0 )])
		+xarea[THARR3D(0 ,-1,0,1,0 )]*(viscosity[THARR3D(0 ,-1,0,0,0 )]-viscosity[THARR3D(-1,-1,0,0,0 )])
		+xarea[THARR3D(0 ,0 ,-1,1,0)]*(viscosity[THARR3D(0 ,0 ,-1,0,0)]-viscosity[THARR3D(-1,0 ,-1,0,0)])
		+xarea[THARR3D(0 ,-1,-1,1,0)]*(viscosity[THARR3D(0 ,-1,-1,0,0)]-viscosity[THARR3D(-1,-1,-1,0,0)]));


        // y velocities
	yvel_temp=yvel0[THARR3D(0,0,0,1,1)]-step_by_mass*(yarea[THARR3D(0 ,0 ,0,0,1 )]*(pressure[THARR3D(0 ,0 ,0,0,0 )]-pressure[THARR3D(0 ,-1,0,0,0 )])
		+yarea[THARR3D(-1,0 ,0,0,1 )]*(pressure[THARR3D(-1,0 ,0,0,0 )]-pressure[THARR3D(-1,-1,0,0,0 )])
		+yarea[THARR3D(0 ,0 ,-1,0,1)]*(pressure[THARR3D(0 ,0 ,-1,0,0)]-pressure[THARR3D(0 ,-1,-1,0,0)])
		+yarea[THARR3D(-1,0 ,-1,0,1)]*(pressure[THARR3D(-1,0 ,-1,0,0)]-pressure[THARR3D(-1,-1,-1,0,0)]));

        yvel1[THARR3D(0,0,0,1,1)]=yvel_temp-step_by_mass*(yarea[THARR3D(0 ,0 ,0,0,1 )]*(viscosity[THARR3D(0 ,0 ,0,0,0 )]-viscosity[THARR3D(0 ,-1,0,0,0 )])
                                                    +yarea[THARR3D(-1,0 ,0,0,1 )]*(viscosity[THARR3D(-1,0 ,0,0,0 )]-viscosity[THARR3D(-1,-1,0,0,0 )])
                                                    +yarea[THARR3D(0 ,0 ,-1,0,1)]*(viscosity[THARR3D(0 ,0 ,-1,0,0)]-viscosity[THARR3D(0 ,-1,-1,0,0)])
                                                    +yarea[THARR3D(-1,0 ,-1,0,1)]*(viscosity[THARR3D(-1,0 ,-1,0,0)]-viscosity[THARR3D(-1,-1,-1,0,0)]));
	// z velocities
        zvel_temp=zvel0[THARR3D(0,0,0,1,1)]-step_by_mass*(zarea[THARR3D(0 ,0 ,0,0,0 )]*(pressure[THARR3D(0 ,0 ,0,0,0 )]-pressure[THARR3D(0 ,0 ,-1,0,0)])
                                                    +zarea[THARR3D(0 ,-1,0,0,0 )]*(pressure[THARR3D(0 ,-1,0,0,0 )]-pressure[THARR3D(0 ,-1,-1,0,0)])
                                                    +zarea[THARR3D(-1,0 ,0,0,0 )]*(pressure[THARR3D(-1,0 ,0,0,0 )]-pressure[THARR3D(-1,0 ,-1,0,0)])
                                                    +zarea[THARR3D(-1,-1,0,0,0 )]*(pressure[THARR3D(-1,-1,0,0,0 )]-pressure[THARR3D(-1,-1,-1,0,0)]));

        zvel1[THARR3D(0,0,0,1,1)]=zvel_temp-step_by_mass*(zarea[THARR3D(0 ,0 ,0,0,0 )]*(viscosity[THARR3D(0 ,0 ,0,0,0 )]-viscosity[THARR3D(0 ,0 ,-1,0,0)])
                                                    +zarea[THARR3D(0 ,-1,0,0,0 )]*(viscosity[THARR3D(0 ,-1,0,0,0 )]-viscosity[THARR3D(0 ,-1,-1,0,0)])
                                                    +zarea[THARR3D(-1,0 ,0,0,0 )]*(viscosity[THARR3D(-1,0 ,0,0,0 )]-viscosity[THARR3D(-1,0 ,-1,0,0)])
                                                    +zarea[THARR3D(-1,-1,0,0,0 )]*(viscosity[THARR3D(-1,-1,0,0,0 )]-viscosity[THARR3D(-1,-1,-1,0,0)]));

    }
    
}
