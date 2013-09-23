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
 *  @brief OCL device-side flux kernel
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details The edge volume fluxes are calculated based on the velocity fields.
 */

#include "ocl_knls.h"

__kernel void flux_calc_ocl_kernel(
    const double dt,
    __global double *xarea,
    __global double *xvel0,
    __global double *xvel1,
    __global double *vol_flux_x,
    __global double *yarea,
    __global double *yvel0,
    __global double *yvel1,
    __global double *vol_flux_y)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2) && (j<=XMAXPLUSTWO) && (k>=2) && (k<=YMAXPLUSONE) ) {

	    vol_flux_x[ARRAYXY(j, k, XMAXPLUSFIVE)] = 0.25*dt*xarea[ARRAYXY(j, k, XMAXPLUSFIVE)]
	                                                           *( xvel0[ARRAYXY(j, k  , XMAXPLUSFIVE)]
				                                                 +xvel0[ARRAYXY(j, k+1, XMAXPLUSFIVE)]
				                                                 +xvel1[ARRAYXY(j, k  , XMAXPLUSFIVE)]
				                                                 +xvel1[ARRAYXY(j, k+1, XMAXPLUSFIVE)]);
    }

    if ( (j>=2) && (j<=XMAXPLUSONE) && (k>=2) && (k<=YMAXPLUSTWO) ) {

	   vol_flux_y[ARRAYXY(j, k, XMAXPLUSFOUR)] = 0.25*dt*yarea[ARRAYXY(j, k, XMAXPLUSFOUR)]
	                                                          *( yvel0[ARRAYXY(j,   k, XMAXPLUSFIVE)]
			                                                    +yvel0[ARRAYXY(j+1, k, XMAXPLUSFIVE)]
			                                                    +yvel1[ARRAYXY(j,   k, XMAXPLUSFIVE)]
			                                                    +yvel1[ARRAYXY(j+1, k, XMAXPLUSFIVE)]);
    }
}

