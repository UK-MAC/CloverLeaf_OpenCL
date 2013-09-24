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
 *  @brief OCL device-side update halo kernels
 *  @author Andrew Mallinson, David Beckingsale, Wayne Gaudin
 *  @details Updates halo cells for the required fields at the required depth
 *  for any halo cells that lie on an external boundary. The location and type
 *  of data governs how this is carried out. External boundaries are always
 *  reflective.
 */

__kernel void update_halo_bottom_cell_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2-depth) && (j<=XMAXPLUSONE+depth) ) {

        field[ (YMIN - k)*XMAXPLUSFOUR + j] = field[ (YMINPLUSONE+k)*XMAXPLUSFOUR+j ];

    }
}

__kernel void update_halo_bottom_vel_ocl_kernel(
    const int depth,
    __global double * restrict field,
    const int multiplier)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2-depth) && (j<=XMAXPLUSTWO+depth) ) {

        field[ (YMIN - k)*XMAXPLUSFIVE + j] = multiplier*field[ (YMINPLUSTWO+k)*XMAXPLUSFIVE+j ];

    }
}

__kernel void update_halo_bottom_flux_x_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2-depth) && (j<=XMAXPLUSTWO+depth) ) {

        field[ (YMIN - k)*XMAXPLUSFIVE + j] = field[ (YMINPLUSTWO+k)*XMAXPLUSFIVE+j ];

    }
}

__kernel void update_halo_bottom_flux_y_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ( (j>=2-depth) && (j<=XMAXPLUSONE+depth) ) {

        field[ (YMIN - k)*XMAXPLUSFOUR + j] = -1*field[ (YMINPLUSTWO+k)*XMAXPLUSFOUR+j ];

    }
}





__kernel void update_halo_top_cell_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2-depth) && (j<=XMAXPLUSONE+depth)) {

        field[ (YMAXPLUSTWO + k)*XMAXPLUSFOUR + j ] = field[ (YMAXPLUSONE - k)*XMAXPLUSFOUR + j ];
    }

}

__kernel void update_halo_top_vel_ocl_kernel(
    const int depth,
    __global double * restrict field,
    const int multiplier)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2-depth) && (j<=XMAXPLUSTWO+depth)) {

        field[ (YMAXPLUSTHREE + k)*XMAXPLUSFIVE + j ] = multiplier*field[ (YMAXPLUSONE - k)*XMAXPLUSFIVE + j ];
    }

}

__kernel void update_halo_top_flux_x_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2-depth) && (j<=XMAXPLUSTWO+depth)) {

        field[ (YMAXPLUSTWO + k)*XMAXPLUSFIVE + j ] = field[ (YMAX - k)*XMAXPLUSFIVE + j ];
    }

}

__kernel void update_halo_top_flux_y_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((j>=2-depth) && (j<=XMAXPLUSONE+depth)) {

        field[ (YMAXPLUSTHREE + k)*XMAXPLUSFOUR + j ] = -1*field[ (YMAXPLUSONE - k)*XMAXPLUSFOUR + j ];
    }

}





__kernel void update_halo_left_cell_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSONE+depth)) {

        field[ (k*XMAXPLUSFOUR)+1-j  ] = field[ (k*XMAXPLUSFOUR)+2+j  ];
    }

}

__kernel void update_halo_left_vel_ocl_kernel(
    const int depth,
    __global double * restrict field,
    const int multiplier)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSTWO+depth)) {

        field[ (k*XMAXPLUSFIVE)+1-j  ] = multiplier*field[ (k*XMAXPLUSFIVE)+3+j  ];
    }

}

__kernel void update_halo_left_flux_x_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSONE+depth)) {

        field[ (k*XMAXPLUSFIVE)+1-j  ] = -1*field[ (k*XMAXPLUSFIVE)+3+j  ];
    }

}

__kernel void update_halo_left_flux_y_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSTWO+depth)) {

        field[ (k*XMAXPLUSFOUR)+1-j  ] = field[ (k*XMAXPLUSFOUR)+3+j  ];
    }

}




__kernel void update_halo_right_cell_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSONE+depth)) {

        field[ k*XMAXPLUSFOUR+XMAXPLUSTWO+j ] = field[ k*XMAXPLUSFOUR+XMAXPLUSONE-j ];
    }
}

__kernel void update_halo_right_vel_ocl_kernel(
    const int depth,
    __global double * restrict field,
    const int multiplier)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSTWO+depth)) {

        field[ k*XMAXPLUSFIVE+XMAXPLUSTHREE+j ] = multiplier*field[ k*XMAXPLUSFIVE+XMAXPLUSONE-j ];
    }
}

__kernel void update_halo_right_flux_x_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSONE+depth)) {

        field[ k*XMAXPLUSFIVE+XMAXPLUSTHREE+j ] = -1*field[ k*XMAXPLUSFIVE+XMAXPLUSONE-j ];
    }
}

__kernel void update_halo_right_flux_y_ocl_kernel(
    const int depth,
    __global double * restrict field)
{
    int k = get_global_id(1);
    int j = get_global_id(0);

    if ((k>=2-depth) && (k<=YMAXPLUSTWO+depth)) {

        field[ k*XMAXPLUSFOUR+XMAXPLUSTWO+j ] = field[ k*XMAXPLUSFOUR+XMAX-j ];
    }
}
