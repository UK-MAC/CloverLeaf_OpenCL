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
 *  @brief OCL device-side acceleration kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details The pressure and viscosity gradients are used to update the
 *  velocity field.
 */

#include "ocl_knls.h"

__kernel void accelerate_ocl_kernel(
    const double dt,
    __global double *xarea,
    __global double *yarea,
    __global double *volume,
    __global double *density0,
    __global double *pressure,
    __global double *viscosity,
    __global double *xvel0,
    __global double *yvel0,
    __global double *xvel1,
    __global double *yvel1,
    __global double *stepbymass)
{
    double nodal_mass;

    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2) && (j<=XMAXPLUSTWO) && (k>=2) && (k<=YMAXPLUSTWO) ) {

        nodal_mass=(density0[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]*volume[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]
                   +density0[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)]*volume[ARRAYXY(j  ,k-1,XMAXPLUSFOUR)]
                   +density0[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]*volume[ARRAYXY(j  ,k  ,XMAXPLUSFOUR)]
                   +density0[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)]*volume[ARRAYXY(j-1,k  ,XMAXPLUSFOUR)])
                   *0.25;
        stepbymass[ARRAYXY(j,k,XMAXPLUSFIVE)]=0.5*dt/nodal_mass;

        xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)]=xvel0[ARRAYXY(j      ,k  ,XMAXPLUSFIVE)]
                                                    -stepbymass[ARRAYXY(j,k  ,XMAXPLUSFIVE)]
                                                    *(xarea[ARRAYXY(j    ,k  ,XMAXPLUSFIVE)]
                                                    *(pressure[ARRAYXY(j ,k  ,XMAXPLUSFOUR)]-pressure[ARRAYXY(j-1,k,XMAXPLUSFOUR)])
                                                    +xarea[ARRAYXY(j     ,k-1,XMAXPLUSFIVE)]
                                                    *(pressure[ARRAYXY(j ,k-1,XMAXPLUSFOUR)]-pressure[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]));

        yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)]=yvel0[ARRAYXY(j      ,k,XMAXPLUSFIVE)]
                                                   -stepbymass[ARRAYXY(j ,k,XMAXPLUSFIVE)]
                                                   *(yarea[ARRAYXY(j     ,k,XMAXPLUSFOUR)]
                                                   *(pressure[ARRAYXY(j  ,k,XMAXPLUSFOUR)]-pressure[ARRAYXY(j,k-1,XMAXPLUSFOUR)])
                                                   +yarea[ARRAYXY(j-1    ,k,XMAXPLUSFOUR)]
                                                   *(pressure[ARRAYXY(j-1,k,XMAXPLUSFOUR)]-pressure[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]));

        xvel1[ARRAYXY(j,k,XMAXPLUSFIVE)]=xvel1[ARRAYXY(j     ,k  ,XMAXPLUSFIVE)]
                                                   -stepbymass[ARRAYXY(j,k  ,XMAXPLUSFIVE)]
                                                   *(xarea[ARRAYXY(j    ,k  ,XMAXPLUSFIVE)]
                                                   *(viscosity[ARRAYXY(j,k  ,XMAXPLUSFOUR)]-viscosity[ARRAYXY(j-1,k,XMAXPLUSFOUR)])
                                                   +xarea[ARRAYXY(j     ,k-1,XMAXPLUSFIVE)]
                                                   *(viscosity[ARRAYXY(j,k-1,XMAXPLUSFOUR)]-viscosity[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]));

        yvel1[ARRAYXY(j,k,XMAXPLUSFIVE)]=yvel1[ARRAYXY(j       ,k,XMAXPLUSFIVE)]
                                                   -stepbymass[ARRAYXY(j  ,k,XMAXPLUSFIVE)]
                                                   *(yarea[ARRAYXY(j      ,k,XMAXPLUSFOUR)]
                                                   *(viscosity[ARRAYXY(j  ,k,XMAXPLUSFOUR)]-viscosity[ARRAYXY(j,k-1,XMAXPLUSFOUR)])
                                                   +yarea[ARRAYXY(j-1     ,k,XMAXPLUSFOUR)]
                                                   *(viscosity[ARRAYXY(j-1,k,XMAXPLUSFOUR)]-viscosity[ARRAYXY(j-1,k-1,XMAXPLUSFOUR)]));
    }
}
