#include <kernel_files/macros_cl.cl>

__kernel void advec_mom_vol
(int mom_sweep,
 __global       double* __restrict const post_vol,
 __global       double* __restrict const pre_vol,
 __global const double* __restrict const volume,
 __global const double* __restrict const vol_flux_x,
 __global const double* __restrict const vol_flux_y)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) - 2 &&*/ row <= (y_max + 1) + 2
    && /*column >= (x_min + 1) - 2 &&*/ column <= (x_max + 1) + 2)
    {
        if(mom_sweep == 1)
        {
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)]
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)];
            pre_vol[THARR2D(0, 0, 1)] = post_vol[THARR2D(0, 0, 1)]
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)];
        }
        else if(mom_sweep == 2)
        {
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)]
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)];
            pre_vol[THARR2D(0, 0, 1)] = post_vol[THARR2D(0, 0, 1)]
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)];
        }
        else if(mom_sweep == 3)
        {
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)];
            pre_vol[THARR2D(0, 0, 1)] = post_vol[THARR2D(0, 0, 1)]
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)];
        }
        else if(mom_sweep == 4)
        {
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)];
            pre_vol[THARR2D(0, 0, 1)] = post_vol[THARR2D(0, 0, 1)]
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)];
        }
    }
}

////////////////////////////////////////////////////////////
//x kernels

__kernel void advec_mom_node_flux_post_x_1
(__global       double* __restrict const node_flux,
 __global const double* __restrict const mass_flux_x)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) - 2 &&*/ column <= (x_max + 1) + 2)
    {
        node_flux[THARR2D(0, 0, 1)] = 0.25
            * (mass_flux_x[THARR2D(0, -1, 1)] + mass_flux_x[THARR2D(0, 0, 1)]
            +  mass_flux_x[THARR2D(1, -1, 1)] + mass_flux_x[THARR2D(1, 0, 1)]);
    }
}

__kernel void advec_mom_node_flux_post_x_2
(__global       double* __restrict const node_mass_post,
 __global const double* __restrict const post_vol,
 __global const double* __restrict const density1)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) - 1 &&*/ column <= (x_max + 1) + 2)
    {
        node_mass_post[THARR2D(0, 0, 1)] = 0.25
            *(density1[THARR2D(0, -1, 0)]  * post_vol[THARR2D(0, -1, 1)]
            + density1[THARR2D(0, 0, 0)]   * post_vol[THARR2D(0, 0, 1)]
            + density1[THARR2D(-1, -1, 0)] * post_vol[THARR2D(-1, -1, 1)]
            + density1[THARR2D(-1, 0, 0)]  * post_vol[THARR2D(-1, 0, 1)]);
    }
}

__kernel void advec_mom_node_pre_x
(__global const double* __restrict const node_flux,
 __global const double* __restrict const node_mass_post,
 __global       double* __restrict const node_mass_pre)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) - 1 &&*/ column <= (x_max + 1) + 2)
    {
        node_mass_pre[THARR2D(0, 0, 1)] = node_mass_post[THARR2D(0, 0, 1)]
            - node_flux[THARR2D(-1, 0, 1)] + node_flux[THARR2D(0, 0, 1)];
    }
}

__kernel void advec_mom_flux_x
(__global const double* __restrict const node_flux,
 __global const double* __restrict const node_mass_post,
 __global const double* __restrict const node_mass_pre,
 __global const double* __restrict const xvel1,
 __global const double* __restrict const celldx,
 __global       double* __restrict const mom_flux)
{
    __kernel_indexes;

    int upwind, donor, downwind, dif;
    double advec_vel;
    double sigma, width, vdiffuw, vdiffdw, limiter;
    double auw, adw, wind;

    if(row >= (y_min + 1) && row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) - 1 &&*/ column <= (x_max + 1) + 1)
    {
        if(node_flux[THARR2D(0, 0, 1)] < 0.0)
        {
            upwind = 2;
            donor = 1;
            downwind = 0;
            dif = donor;
        }
        else
        {
            upwind = -1;
            donor = 0;
            downwind = 1;
            dif = upwind;
        }

        sigma = fabs(node_flux[THARR2D(0, 0, 1)]) / node_mass_pre[THARR2D(donor, 0, 1)];
        vdiffuw = xvel1[THARR2D(donor, 0, 1)] - xvel1[THARR2D(upwind, 0, 1)];
        vdiffdw = xvel1[THARR2D(downwind, 0, 1)] - xvel1[THARR2D(donor, 0, 1)];
        limiter = 0.0;

        if(vdiffdw * vdiffuw > 0.0)
        {
            auw = fabs(vdiffuw);
            adw = fabs(vdiffdw);
            wind = (vdiffdw <= 0.0) ? -1.0 : 1.0;
            width = celldx[column];
            limiter = wind * MIN(width * ((2.0 - sigma) * adw / width 
                + (1.0 + sigma) * auw / celldx[column + dif]) / 6.0, 
                MIN(auw, adw));
        }

        advec_vel = xvel1[THARR2D(donor, 0, 1)] + (1.0 - sigma) * limiter;
        mom_flux[THARR2D(0, 0, 1)] = advec_vel * node_flux[THARR2D(0, 0, 1)];
    }
}

__kernel void advec_mom_xvel
(__global const double* __restrict const node_mass_post,
 __global const double* __restrict const node_mass_pre,
 __global const double* __restrict const mom_flux,
 __global       double* __restrict const xvel1)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)
    {
        xvel1[THARR2D(0, 0, 1)] = (xvel1[THARR2D(0, 0, 1)]
            * node_mass_pre[THARR2D(0, 0, 1)] + mom_flux[THARR2D(-1, 0, 1)]
            - mom_flux[THARR2D(0, 0, 1)]) / node_mass_post[THARR2D(0, 0, 1)];
    }
}

////////////////////////////////////////////////////////////
//y kernels

__kernel void advec_mom_node_flux_post_y_1
(__global       double* __restrict const node_flux,
 __global const double* __restrict const mass_flux_y)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) - 2 &&*/ row <= (y_max + 1) + 2
    && /*column >= (x_min + 1) &&*/column <= (x_max + 1) + 1)
    {
        node_flux[THARR2D(0, 0, 1)] = 0.25
            * (mass_flux_y[THARR2D(-1, 0, 0)] + mass_flux_y[THARR2D(0, 0, 0)]
            +  mass_flux_y[THARR2D(-1, 1, 0)] + mass_flux_y[THARR2D(0, 1, 0)]);
    }
}

__kernel void advec_mom_node_flux_post_y_2
(__global       double* __restrict const node_mass_post,
 __global const double* __restrict const post_vol,
 __global const double* __restrict const density1)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) - 1 &&*/ row <= (y_max + 1) + 2
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)
    {
        node_mass_post[THARR2D(0, 0, 1)] = 0.25
            * (density1[THARR2D(0, -1, 0)] * post_vol[THARR2D(0, -1, 1)]
            + density1[THARR2D(0, 0, 0)]   * post_vol[THARR2D(0, 0, 1)]
            + density1[THARR2D(-1, -1, 0)] * post_vol[THARR2D(-1, -1, 1)]
            + density1[THARR2D(-1, 0, 0)]  * post_vol[THARR2D(-1, 0, 1)]);
    }
}

__kernel void advec_mom_node_pre_y
(__global const double* __restrict const node_flux,
 __global const double* __restrict const node_mass_post,
 __global       double* __restrict const node_mass_pre)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) - 1 &&*/ row <= (y_max + 1) + 2
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)
    {
        node_mass_pre[THARR2D(0, 0, 1)] = node_mass_post[THARR2D(0, 0, 1)]
            - node_flux[THARR2D(0, -1, 1)] + node_flux[THARR2D(0, 0, 1)];
    }
}

__kernel void advec_mom_flux_y
(__global const double* __restrict const node_flux,
 __global const double* __restrict const node_mass_post,
 __global const double* __restrict const node_mass_pre,
 __global const double* __restrict const yvel1,
 __global const double* __restrict const celldy,
 __global       double* __restrict const mom_flux)
{
    __kernel_indexes;

    int upwind, donor, downwind, dif;
    double advec_vel;
    double sigma, width, vdiffuw, vdiffdw, limiter;
    double auw, adw, wind;

    if(/*row >= (y_min + 1) - 1 &&*/ row <= (y_max + 1) + 1
    && column >= (x_min + 1) && column <= (x_max + 1) + 1)
    {
        if(node_flux[THARR2D(0, 0, 1)] < 0.0)
        {
            upwind = 2;
            donor = 1;
            downwind = 0;
            dif = donor;
        }
        else
        {
            upwind = -1;
            donor = 0;
            downwind = 1;
            dif = upwind;
        }

        sigma = fabs(node_flux[THARR2D(0, 0, 1)]) / node_mass_pre[THARR2D(0, donor, 1)];
        vdiffuw = yvel1[THARR2D(0, donor, 1)] - yvel1[THARR2D(0, upwind, 1)];
        vdiffdw = yvel1[THARR2D(0, downwind, 1)] - yvel1[THARR2D(0, donor, 1)];
        limiter = 0.0;

        if(vdiffdw * vdiffuw > 0.0)
        {
            auw = fabs(vdiffuw);
            adw = fabs(vdiffdw);
            wind = (vdiffdw <= 0.0) ? -1.0 : 1.0;
            width = celldy[row];
            limiter = wind * MIN(width * ((2.0 - sigma) * adw / width 
                + (1.0 + sigma) * auw / celldy[row + dif]) / 6.0, 
                MIN(auw, adw));
        }

        advec_vel = yvel1[THARR2D(0, donor, 1)] + (1.0 - sigma) * limiter;
        mom_flux[THARR2D(0, 0, 1)] = advec_vel * node_flux[THARR2D(0, 0, 1)];

    }
}

__kernel void advec_mom_yvel
(__global const double* __restrict const node_mass_post,
 __global const double* __restrict const node_mass_pre,
 __global const double* __restrict const mom_flux,
 __global       double* __restrict const yvel1)
{
    __kernel_indexes;

    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)
    {
        yvel1[THARR2D(0, 0, 1)] = (yvel1[THARR2D(0, 0, 1)]
            * node_mass_pre[THARR2D(0, 0, 1)] + mom_flux[THARR2D(0, -1, 1)]
            - mom_flux[THARR2D(0, 0, 1)]) / node_mass_post[THARR2D(0, 0, 1)];
    }
}

