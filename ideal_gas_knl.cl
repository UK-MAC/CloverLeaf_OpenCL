/*Crown Copyright 2012 AWE.
*
* This file is part of CloverLeaf.
*
* CloverLeaf is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3 of the License, or (at your option)
* any later version.
*
* CloverLeaf is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
* details.
*
* You should have received a copy of the GNU General Public License along with
* CloverLeaf. If not, see http://www.gnu.org/licenses/. */

/**
 *  @brief OCL device-side ideal gas kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Calculates the pressure and sound speed for the mesh chunk using
 *  the ideal gas equation of state, with a fixed gamma of 1.4.
 */

#include "ocl_knls.h"

__kernel void ideal_gas_ocl_kernel(
    __global double *density,
    __global double *energy,
    __global double *pressure,
    __global double *soundspeed)
{

    int k = get_global_id(1);
    int j = get_global_id(0);

    double sound_speed_squared,v,pressurebyenergy,pressurebyvolume;

    if ( (j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSONE) ) {

        v = 1.0/density[ARRAYXY(j,k,XMAXPLUSFOUR)];

        pressure[ARRAYXY(j,k,XMAXPLUSFOUR)]=(1.4-1.0)*density[ARRAYXY(j,k,XMAXPLUSFOUR)]*energy[ARRAYXY(j,k,XMAXPLUSFOUR)];

        pressurebyenergy=(1.4-1.0)*density[ARRAYXY(j,k,XMAXPLUSFOUR)];

        pressurebyvolume=-density[ARRAYXY(j,k,XMAXPLUSFOUR)]*pressure[ARRAYXY(j,k,XMAXPLUSFOUR)];

        sound_speed_squared=v*v*(pressure[ARRAYXY(j,k,XMAXPLUSFOUR)]*pressurebyenergy-pressurebyvolume);

        soundspeed[ARRAYXY(j,k,XMAXPLUSFOUR)]=sqrt(sound_speed_squared);
    }
}
