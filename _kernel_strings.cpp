#include "ocl_kernel_hdr.hpp"
const std::string src_accelerate_cl("#include <kernel_files/macros_cl.cl> \n\
__kernel void accelerate\n\
(double dbyt,\n\
 __global const double * const __restrict xarea,\n\
 __global const double * const __restrict yarea,\n\
 __global const double * const __restrict volume,\n\
 __global const double * const __restrict density0,\n\
 __global const double * const __restrict pressure,\n\
 __global const double * const __restrict viscosity,\n\
 __global const double * const __restrict xvel0,\n\
 __global const double * const __restrict yvel0,\n\
 __global       double * const __restrict xvel1,\n\
 __global       double * const __restrict yvel1)\n\
{\n\
    __kernel_indexes;\n\
\n\
    double nodal_mass, step_by_mass;\n\
\n\
    // prevent writing to *vel1, then read from it, then write to it again\n\
    double xvel_temp, yvel_temp;\n\
\n\
    if(/*row >= (x_min + 1) &&*/ row <= (y_max + 1) + 1\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)\n\
    {\n\
        nodal_mass =\n\
            (density0[THARR2D(-1, -1, 0)] * volume[THARR2D(-1, -1, 0)]\n\
            + density0[THARR2D(0, -1, 0)] * volume[THARR2D(0, -1, 0)]\n\
            + density0[THARR2D(0, 0, 0)] * volume[THARR2D(0, 0, 0)]\n\
            + density0[THARR2D(-1, 0, 0)] * volume[THARR2D(-1, 0, 0)])\n\
            * 0.25;\n\
\n\
        step_by_mass = 0.5 * dbyt / nodal_mass;\n\
\n\
        // x velocities\n\
        xvel_temp = xvel0[THARR2D(0, 0, 1)] - step_by_mass\n\
            * (xarea[THARR2D(0, 0, 1)] * (pressure[THARR2D(0, 0, 0)]  - pressure[THARR2D(-1, 0, 0)])\n\
            + xarea[THARR2D(0, -1, 1)] * (pressure[THARR2D(0, -1, 0)] - pressure[THARR2D(-1, -1, 0)]));\n\
\n\
        xvel1[THARR2D(0, 0, 1)] = xvel_temp - step_by_mass\n\
            * (xarea[THARR2D(0, 0, 1)] * (viscosity[THARR2D(0, 0, 0)]  - viscosity[THARR2D(-1, 0, 0)])\n\
            + xarea[THARR2D(0, -1, 1)] * (viscosity[THARR2D(0, -1, 0)] - viscosity[THARR2D(-1, -1, 0)]));\n\
\n\
        // y velocities\n\
        yvel_temp = yvel0[THARR2D(0, 0, 1)] - step_by_mass\n\
            * (yarea[THARR2D(0, 0, 0)] * (pressure[THARR2D(0, 0, 0)]  - pressure[THARR2D(0, -1, 0)])\n\
            + yarea[THARR2D(-1, 0, 0)] * (pressure[THARR2D(-1, 0, 0)] - pressure[THARR2D(-1, -1, 0)]));\n\
\n\
        yvel1[THARR2D(0, 0, 1)] = yvel_temp - step_by_mass\n\
            * (yarea[THARR2D(0, 0, 0)] * (viscosity[THARR2D(0, 0, 0)]  - viscosity[THARR2D(0, -1, 0)])\n\
            + yarea[THARR2D(-1, 0, 0)] * (viscosity[THARR2D(-1, 0, 0)] - viscosity[THARR2D(-1, -1, 0)]));\n\
\n\
    }\n\
    \n\
}\n\
");
const std::string src_advec_cell_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
#define _SHARED_KERNEL_ARGS_                    \\\n\
    const int swp_nmbr,                         \\\n\
    __global const double* __restrict const volume,      \\\n\
    __global const double* __restrict const vol_flux_x,  \\\n\
    __global const double* __restrict const vol_flux_y,  \\\n\
    __global const double* __restrict const pre_vol,     \\\n\
    __global       double* __restrict const density1,    \\\n\
    __global       double* __restrict const energy1,     \\\n\
    __global       double* __restrict const ener_flux\n\
\n\
__kernel void advec_cell_pre_vol_x\n\
(const int swp_nmbr,\n\
 __global       double* __restrict const pre_vol,\n\
 __global       double* __restrict const post_vol,\n\
 __global const double* __restrict const volume,\n\
 __global const double* __restrict const vol_flux_x,\n\
 __global const double* __restrict const vol_flux_y)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) - 2 &&*/ row <= (y_max + 1) + 2\n\
    && /*column >= (x_min + 1) - 2 &&*/ column <= (x_max + 1) + 2)\n\
    {\n\
        if(swp_nmbr == 1)\n\
        {\n\
            pre_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)]\n\
                +(vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)]\n\
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)]);\n\
            post_vol[THARR2D(0, 0, 1)] = pre_vol[THARR2D(0, 0, 1)]\n\
                - (vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)]);\n\
        }\n\
        else\n\
        {\n\
            pre_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)]\n\
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)];\n\
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)];\n\
        }\n\
    }\n\
}\n\
\n\
__kernel void advec_cell_ener_flux_x\n\
(_SHARED_KERNEL_ARGS_,\n\
 __global const double* __restrict const vertexdx,\n\
 __global       double* __restrict const mass_flux_x)\n\
{\n\
    __kernel_indexes;\n\
\n\
    double sigmat, sigmam, sigma3, sigma4, diffuw, diffdw, limiter;\n\
    int upwind, donor, downwind, dif;\n\
    const double one_by_six = 1.0/6.0;\n\
\n\
    //\n\
    //  if cell is within x area:\n\
    //  +++++++++++++++++++++\n\
    //  +++++++++++++++++++++\n\
    //  ++xxxxxxxxxxxxxxxxxxx\n\
    //  +++++++++++++++++++++\n\
    //  +++++++++++++++++++++\n\
    //\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1)\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 2)\n\
    {\n\
        // if flowing right\n\
        if(vol_flux_x[THARR2D(0, 0, 1)] > 0.0)\n\
        {\n\
            upwind = -2;\n\
            donor = -1;\n\
            downwind = 0;\n\
            dif = donor;\n\
        }\n\
        else\n\
        {\n\
            //  tries to get from below, unless it would be reading from a cell\n\
            //  which would be off the right, in which case read from cur cell\n\
            upwind = (column == (x_max + 1) + 2) ? 0 : 1;\n\
            donor = 0;\n\
            downwind = -1;\n\
            dif = upwind;\n\
        }\n\
\n\
        sigmat = fabs(vol_flux_x[THARR2D(0, 0, 1)]) / pre_vol[THARR2D(donor, 0, 1)];\n\
        sigma3 = (1.0 + sigmat) * (vertexdx[column] / vertexdx[column + dif]);\n\
        sigma4 = 2.0 - sigmat;\n\
\n\
        diffuw = density1[THARR2D(donor, 0, 0)] - density1[THARR2D(upwind, 0, 0)];\n\
        diffdw = density1[THARR2D(downwind, 0, 0)] - density1[THARR2D(donor, 0, 0)];\n\
\n\
        if(diffuw * diffdw > 0.0)\n\
        {\n\
            limiter = (1.0 - sigmat) * SIGN(1.0, diffdw)\n\
                * MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six\n\
                * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));\n\
        }\n\
        else\n\
        {\n\
            limiter = 0.0;\n\
        }\n\
\n\
        mass_flux_x[THARR2D(0, 0, 1)] = vol_flux_x[THARR2D(0, 0, 1)]\n\
            * (density1[THARR2D(donor, 0, 0)] + limiter);\n\
\n\
        sigmam = fabs(mass_flux_x[THARR2D(0, 0, 1)])\n\
            / (density1[THARR2D(donor, 0, 0)] * pre_vol[THARR2D(donor, 0, 1)]);\n\
        diffuw = energy1[THARR2D(donor, 0, 0)] - energy1[THARR2D(upwind, 0, 0)];\n\
        diffdw = energy1[THARR2D(downwind, 0, 0)] - energy1[THARR2D(donor, 0, 0)];\n\
\n\
        if(diffuw * diffdw > 0.0)\n\
        {\n\
            limiter = (1.0 - sigmam) * SIGN(1.0, diffdw)\n\
                * MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six\n\
                * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));\n\
        }\n\
        else\n\
        {\n\
            limiter = 0.0;\n\
        }\n\
\n\
        ener_flux[THARR2D(0, 0, 0)] = mass_flux_x[THARR2D(0, 0, 1)]\n\
            * (energy1[THARR2D(donor, 0, 0)] + limiter);\n\
    }\n\
}\n\
\n\
__kernel void advec_cell_x\n\
(_SHARED_KERNEL_ARGS_,\n\
 __global const double* __restrict const mass_flux_x)\n\
{\n\
    __kernel_indexes;\n\
\n\
    double pre_mass, post_mass, advec_vol, post_ener;\n\
\n\
    //\n\
    //  if cell is within x area:\n\
    //  +++++++++++++++++++++\n\
    //  +++++++++++++++++++++\n\
    //  ++xxxxxxxxxxxxxxxxx++\n\
    //  +++++++++++++++++++++\n\
    //  +++++++++++++++++++++\n\
    //\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1)\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))\n\
    {\n\
        pre_mass = density1[THARR2D(0, 0, 0)] * pre_vol[THARR2D(0, 0, 1)];\n\
\n\
        post_mass = pre_mass + mass_flux_x[THARR2D(0, 0, 1)]\n\
            - mass_flux_x[THARR2D(1, 0, 1)];\n\
\n\
        post_ener = (energy1[THARR2D(0, 0, 0)] * pre_mass\n\
            + ener_flux[THARR2D(0, 0, 0)] - ener_flux[THARR2D(1, 0, 0)])\n\
            / post_mass;\n\
\n\
        advec_vol = pre_vol[THARR2D(0, 0, 1)] + vol_flux_x[THARR2D(0, 0, 1)]\n\
            - vol_flux_x[THARR2D(1, 0, 1)];\n\
\n\
        density1[THARR2D(0, 0, 0)] = post_mass / advec_vol;\n\
        energy1[THARR2D(0, 0, 0)] = post_ener;\n\
    }\n\
}\n\
\n\
//////////////////////////////////////////////////////////////////////////\n\
//y kernels\n\
\n\
__kernel void advec_cell_pre_vol_y\n\
(const int swp_nmbr,\n\
 __global       double* __restrict const pre_vol,\n\
 __global       double* __restrict const post_vol,\n\
 __global const double* __restrict const volume,\n\
 __global const double* __restrict const vol_flux_x,\n\
 __global const double* __restrict const vol_flux_y)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) - 2 &&*/ row <= (y_max + 1) + 2\n\
    && /*column >= (x_min + 1) - 2 &&*/ column <= (x_max + 1) + 2)\n\
    {\n\
        if(swp_nmbr == 1)\n\
        {\n\
            pre_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)]\n\
                +(vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)]\n\
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)]);\n\
            post_vol[THARR2D(0, 0, 1)] = pre_vol[THARR2D(0, 0, 1)]\n\
                - (vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 1)]);\n\
        }\n\
        else\n\
        {\n\
            pre_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)]\n\
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)];\n\
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)];\n\
        }\n\
    }\n\
}\n\
\n\
__kernel void advec_cell_ener_flux_y\n\
(_SHARED_KERNEL_ARGS_,\n\
 __global const double* __restrict const vertexdy,\n\
 __global       double* __restrict const mass_flux_y)\n\
{\n\
    __kernel_indexes;\n\
\n\
    double sigmat, sigmam, sigma3, sigma4, diffuw, diffdw, limiter;\n\
    int upwind, donor, downwind, dif;\n\
    const double one_by_six = 1.0/6.0;\n\
\n\
    //\n\
    //  if cell is within x area:\n\
    //  +++++++++++++++++++++\n\
    //  +++++++++++++++++++++\n\
    //  ++xxxxxxxxxxxxxxxxx++\n\
    //  ++xxxxxxxxxxxxxxxxx++\n\
    //\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 2\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))\n\
    {\n\
        // if flowing up\n\
        if(vol_flux_y[THARR2D(0, 0, 0)] > 0.0)\n\
        {\n\
            upwind = -2;\n\
            donor = -1;\n\
            downwind = 0;\n\
            dif = donor;\n\
        }\n\
        else\n\
        {\n\
            //\n\
            //  tries to get from below, unless it would be reading from a cell\n\
            //  which would be off the bottom, in which case read from cur cell\n\
            //\n\
            upwind = (row == (y_max + 1) + 2) ? 0 : 1;\n\
            donor = 0;\n\
            downwind = -1;\n\
            dif = downwind;\n\
        }\n\
\n\
        sigmat = fabs(vol_flux_y[THARR2D(0, 0, 0)]) / pre_vol[THARR2D(0, donor, 1)];\n\
        sigma3 = (1.0 + sigmat) * (vertexdy[row] / vertexdy[row + dif]);\n\
        sigma4 = 2.0 - sigmat;\n\
\n\
        diffuw = density1[THARR2D(0, donor, 0)] - density1[THARR2D(0, upwind, 0)];\n\
        diffdw = density1[THARR2D(0, downwind, 0)] - density1[THARR2D(0, donor, 0)];\n\
\n\
        if(diffuw * diffdw > 0.0)\n\
        {\n\
            limiter = (1.0 - sigmat) * SIGN(1.0, diffdw)\n\
                * MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six\n\
                * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));\n\
        }\n\
        else\n\
        {\n\
            limiter = 0.0;\n\
        }\n\
\n\
        mass_flux_y[THARR2D(0, 0, 0)] = vol_flux_y[THARR2D(0, 0, 0)]\n\
            * (density1[THARR2D(0, donor, 0)] + limiter);\n\
\n\
        sigmam = fabs(mass_flux_y[THARR2D(0, 0, 0)])\n\
            / (density1[THARR2D(0, donor, 0)] * pre_vol[THARR2D(0, donor, 1)]);\n\
        diffuw = energy1[THARR2D(0, donor, 0)] - energy1[THARR2D(0, upwind, 0)];\n\
        diffdw = energy1[THARR2D(0, downwind, 0)] - energy1[THARR2D(0, donor, 0)];\n\
\n\
        if(diffuw * diffdw > 0.0)\n\
        {\n\
            limiter = (1.0 - sigmam) * SIGN(1.0, diffdw)\n\
                * MIN(fabs(diffuw), MIN(fabs(diffdw), one_by_six\n\
                * (sigma3 * fabs(diffuw) + sigma4 * fabs(diffdw))));\n\
        }\n\
        else\n\
        {\n\
            limiter = 0.0;\n\
        }\n\
\n\
        ener_flux[THARR2D(0, 0, 0)] = mass_flux_y[THARR2D(0, 0, 0)]\n\
            * (energy1[THARR2D(0, donor, 0)] + limiter);\n\
    }\n\
}\n\
\n\
__kernel void advec_cell_y\n\
(_SHARED_KERNEL_ARGS_,\n\
 __global const double* __restrict const mass_flux_y)\n\
{\n\
    __kernel_indexes;\n\
\n\
    double pre_mass, post_mass, advec_vol, post_ener;\n\
\n\
    //\n\
    //  if cell is within x area:\n\
    //  +++++++++++++++++++++\n\
    //  +++++++++++++++++++++\n\
    //  ++xxxxxxxxxxxxxxxxx++\n\
    //  +++++++++++++++++++++\n\
    //  +++++++++++++++++++++\n\
    //\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1)\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))\n\
    {\n\
        pre_mass = density1[THARR2D(0, 0, 0)] * pre_vol[THARR2D(0, 0, 1)];\n\
\n\
        post_mass = pre_mass + mass_flux_y[THARR2D(0, 0, 0)]\n\
            - mass_flux_y[THARR2D(0, 1, 0)];\n\
\n\
        post_ener = (energy1[THARR2D(0, 0, 0)] * pre_mass\n\
            + ener_flux[THARR2D(0, 0, 0)] - ener_flux[THARR2D(0, 1, 0)])\n\
            / post_mass;\n\
\n\
        advec_vol = pre_vol[THARR2D(0, 0, 1)] + vol_flux_y[THARR2D(0, 0, 0)]\n\
            - vol_flux_y[THARR2D(0, 1, 0)];\n\
\n\
        density1[THARR2D(0, 0, 0)] = post_mass / advec_vol;\n\
        energy1[THARR2D(0, 0, 0)] = post_ener;\n\
    }\n\
}\n\
");
const std::string src_advec_mom_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void advec_mom_vol\n\
(int mom_sweep,\n\
 __global       double* __restrict const post_vol,\n\
 __global       double* __restrict const pre_vol,\n\
 __global const double* __restrict const volume,\n\
 __global const double* __restrict const vol_flux_x,\n\
 __global const double* __restrict const vol_flux_y)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) - 2 &&*/ row <= (y_max + 1) + 2\n\
    && /*column >= (x_min + 1) - 2 &&*/ column <= (x_max + 1) + 2)\n\
    {\n\
        if(mom_sweep == 1)\n\
        {\n\
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)]\n\
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)];\n\
            pre_vol[THARR2D(0, 0, 1)] = post_vol[THARR2D(0, 0, 1)]\n\
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)];\n\
        }\n\
        else if(mom_sweep == 2)\n\
        {\n\
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)]\n\
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)];\n\
            pre_vol[THARR2D(0, 0, 1)] = post_vol[THARR2D(0, 0, 1)]\n\
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)];\n\
        }\n\
        else if(mom_sweep == 3)\n\
        {\n\
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)];\n\
            pre_vol[THARR2D(0, 0, 1)] = post_vol[THARR2D(0, 0, 1)]\n\
                + vol_flux_y[THARR2D(0, 1, 0)] - vol_flux_y[THARR2D(0, 0, 0)];\n\
        }\n\
        else if(mom_sweep == 4)\n\
        {\n\
            post_vol[THARR2D(0, 0, 1)] = volume[THARR2D(0, 0, 0)];\n\
            pre_vol[THARR2D(0, 0, 1)] = post_vol[THARR2D(0, 0, 1)]\n\
                + vol_flux_x[THARR2D(1, 0, 1)] - vol_flux_x[THARR2D(0, 0, 1)];\n\
        }\n\
    }\n\
}\n\
\n\
////////////////////////////////////////////////////////////\n\
//x kernels\n\
\n\
__kernel void advec_mom_node_flux_post_x\n\
(__global       double* __restrict const node_flux,\n\
 __global       double* __restrict const node_mass_post,\n\
 __global const double* __restrict const mass_flux_x,\n\
 __global const double* __restrict const post_vol,\n\
 __global const double* __restrict const density1)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1\n\
    && /*column >= (x_min + 1) - 1 &&*/ column <= (x_max + 1) + 2)\n\
    {\n\
        node_flux[THARR2D(0, 0, 1)] = 0.25\n\
            * (mass_flux_x[THARR2D(0, -1, 1)] + mass_flux_x[THARR2D(0, 0, 1)]\n\
            + mass_flux_x[THARR2D(1, -1, 1)] + mass_flux_x[THARR2D(1, 0, 1)]);\n\
/*\n\
    }\n\
\n\
    if(row >= (y_min + 1) && row <= (y_max + 1) + 1\n\
    && column >= (x_min + 1) - 1 && column <= (x_max + 1) + 2)\n\
    {\n\
*/\n\
            node_mass_post[THARR2D(0, 0, 1)] = 0.25\n\
                *(density1[THARR2D(0, -1, 0)]  * post_vol[THARR2D(0, -1, 1)]\n\
                + density1[THARR2D(0, 0, 0)]   * post_vol[THARR2D(0, 0, 1)]\n\
                + density1[THARR2D(-1, -1, 0)] * post_vol[THARR2D(-1, -1, 1)]\n\
                + density1[THARR2D(-1, 0, 0)]  * post_vol[THARR2D(-1, 0, 1)]);\n\
    }\n\
}\n\
\n\
__kernel void advec_mom_node_pre_x\n\
(__global const double* __restrict const node_flux,\n\
 __global const double* __restrict const node_mass_post,\n\
 __global       double* __restrict const node_mass_pre)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1\n\
    && /*column >= (x_min + 1) - 1 &&*/ column <= (x_max + 1) + 2)\n\
    {\n\
        node_mass_pre[THARR2D(0, 0, 1)] = node_mass_post[THARR2D(0, 0, 1)]\n\
            - node_flux[THARR2D(-1, 0, 1)] + node_flux[THARR2D(0, 0, 1)];\n\
    }\n\
}\n\
\n\
__kernel void advec_mom_flux_x\n\
(__global const double* __restrict const node_flux,\n\
 __global const double* __restrict const node_mass_post,\n\
 __global const double* __restrict const node_mass_pre,\n\
 __global const double* __restrict const xvel1,\n\
 __global const double* __restrict const celldx,\n\
 __global       double* __restrict const mom_flux)\n\
{\n\
    __kernel_indexes;\n\
\n\
    int upwind, donor, downwind, dif;\n\
    double advec_vel;\n\
    double sigma, width, vdiffuw, vdiffdw, limiter;\n\
    double auw, adw, wind;\n\
\n\
    if(row >= (y_min + 1) && row <= (y_max + 1) + 1\n\
    && /*column >= (x_min + 1) - 1 &&*/ column <= (x_max + 1) + 1)\n\
    {\n\
        if(node_flux[THARR2D(0, 0, 1)] < 0.0)\n\
        {\n\
            upwind = 2;\n\
            donor = 1;\n\
            downwind = 0;\n\
            dif = donor;\n\
        }\n\
        else\n\
        {\n\
            upwind = -1;\n\
            donor = 0;\n\
            downwind = 1;\n\
            dif = upwind;\n\
        }\n\
\n\
        sigma = fabs(node_flux[THARR2D(0, 0, 1)]) / node_mass_pre[THARR2D(donor, 0, 1)];\n\
        vdiffuw = xvel1[THARR2D(donor, 0, 1)] - xvel1[THARR2D(upwind, 0, 1)];\n\
        vdiffdw = xvel1[THARR2D(downwind, 0, 1)] - xvel1[THARR2D(donor, 0, 1)];\n\
        limiter = 0.0;\n\
\n\
        if(vdiffdw * vdiffuw > 0.0)\n\
        {\n\
            auw = fabs(vdiffuw);\n\
            adw = fabs(vdiffdw);\n\
            wind = (vdiffdw <= 0.0) ? -1.0 : 1.0;\n\
            width = celldx[column];\n\
            limiter = wind * MIN(width * ((2.0 - sigma) * adw / width \n\
                + (1.0 + sigma) * auw / celldx[column + dif]) / 6.0, \n\
                MIN(auw, adw));\n\
        }\n\
\n\
        advec_vel = xvel1[THARR2D(donor, 0, 1)] + (1.0 - sigma) * limiter;\n\
        mom_flux[THARR2D(0, 0, 1)] = advec_vel * node_flux[THARR2D(0, 0, 1)];\n\
    }\n\
}\n\
\n\
__kernel void advec_mom_xvel\n\
(__global const double* __restrict const node_mass_post,\n\
 __global const double* __restrict const node_mass_pre,\n\
 __global const double* __restrict const mom_flux,\n\
 __global       double* __restrict const xvel1)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)\n\
    {\n\
        xvel1[THARR2D(0, 0, 1)] = (xvel1[THARR2D(0, 0, 1)]\n\
            * node_mass_pre[THARR2D(0, 0, 1)] + mom_flux[THARR2D(-1, 0, 1)]\n\
            - mom_flux[THARR2D(0, 0, 1)]) / node_mass_post[THARR2D(0, 0, 1)];\n\
    }\n\
}\n\
\n\
////////////////////////////////////////////////////////////\n\
//y kernels\n\
\n\
__kernel void advec_mom_node_flux_post_y\n\
(__global       double* __restrict const node_flux,\n\
 __global       double* __restrict const node_mass_post,\n\
 __global const double* __restrict const mass_flux_y,\n\
 __global const double* __restrict const post_vol,\n\
 __global const double* __restrict const density1)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) - 2 &&*/ row <= (y_max + 1) + 2\n\
    && /*column >= (x_min + 1) &&*/column <= (x_max + 1) + 1)\n\
    {\n\
        node_flux[THARR2D(0, 0, 1)] = 0.25\n\
            * (mass_flux_y[THARR2D(-1, 0, 0)] + mass_flux_y[THARR2D(0, 0, 0)]\n\
            + mass_flux_y[THARR2D(-1, 1, 0)] + mass_flux_y[THARR2D(0, 1, 0)]);\n\
/*\n\
    }\n\
\n\
    if(row >= (y_min + 1) - 1 && row <= (y_max + 1) + 2\n\
    && column >= (x_min + 1) && column <= (x_max + 1) + 1)\n\
    {\n\
*/\n\
        node_mass_post[THARR2D(0, 0, 1)] = 0.25\n\
            * (density1[THARR2D(0, -1, 0)] * post_vol[THARR2D(0, -1, 1)]\n\
            + density1[THARR2D(0, 0, 0)]   * post_vol[THARR2D(0, 0, 1)]\n\
            + density1[THARR2D(-1, -1, 0)] * post_vol[THARR2D(-1, -1, 1)]\n\
            + density1[THARR2D(-1, 0, 0)]  * post_vol[THARR2D(-1, 0, 1)]);\n\
    }\n\
}\n\
\n\
__kernel void advec_mom_node_pre_y\n\
(__global const double* __restrict const node_flux,\n\
 __global const double* __restrict const node_mass_post,\n\
 __global       double* __restrict const node_mass_pre)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) - 1 &&*/ row <= (y_max + 1) + 2\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)\n\
    {\n\
        node_mass_pre[THARR2D(0, 0, 1)] = node_mass_post[THARR2D(0, 0, 1)]\n\
            - node_flux[THARR2D(0, -1, 1)] + node_flux[THARR2D(0, 0, 1)];\n\
    }\n\
}\n\
\n\
__kernel void advec_mom_flux_y\n\
(__global const double* __restrict const node_flux,\n\
 __global const double* __restrict const node_mass_post,\n\
 __global const double* __restrict const node_mass_pre,\n\
 __global const double* __restrict const yvel1,\n\
 __global const double* __restrict const celldy,\n\
 __global       double* __restrict const mom_flux)\n\
{\n\
    __kernel_indexes;\n\
\n\
    int upwind, donor, downwind, dif;\n\
    double advec_vel;\n\
    double sigma, width, vdiffuw, vdiffdw, limiter;\n\
    double auw, adw, wind;\n\
\n\
    if(/*row >= (y_min + 1) - 1 &&*/ row <= (y_max + 1) + 1\n\
    && column >= (x_min + 1) && column <= (x_max + 1) + 1)\n\
    {\n\
        if(node_flux[THARR2D(0, 0, 1)] < 0.0)\n\
        {\n\
            upwind = 2;\n\
            donor = 1;\n\
            downwind = 0;\n\
            dif = donor;\n\
        }\n\
        else\n\
        {\n\
            upwind = -1;\n\
            donor = 0;\n\
            downwind = 1;\n\
            dif = upwind;\n\
        }\n\
\n\
        sigma = fabs(node_flux[THARR2D(0, 0, 1)]) / node_mass_pre[THARR2D(0, donor, 1)];\n\
        vdiffuw = yvel1[THARR2D(0, donor, 1)] - yvel1[THARR2D(0, upwind, 1)];\n\
        vdiffdw = yvel1[THARR2D(0, downwind, 1)] - yvel1[THARR2D(0, donor, 1)];\n\
        limiter = 0.0;\n\
\n\
        if(vdiffdw * vdiffuw > 0.0)\n\
        {\n\
            auw = fabs(vdiffuw);\n\
            adw = fabs(vdiffdw);\n\
            wind = (vdiffdw <= 0.0) ? -1.0 : 1.0;\n\
            width = celldy[row];\n\
            limiter = wind * MIN(width * ((2.0 - sigma) * adw / width \n\
                + (1.0 + sigma) * auw / celldy[row + dif]) / 6.0, \n\
                MIN(auw, adw));\n\
        }\n\
\n\
        advec_vel = yvel1[THARR2D(0, donor, 1)] + (1.0 - sigma) * limiter;\n\
        mom_flux[THARR2D(0, 0, 1)] = advec_vel * node_flux[THARR2D(0, 0, 1)];\n\
\n\
    }\n\
}\n\
\n\
__kernel void advec_mom_yvel\n\
(__global const double* __restrict const node_mass_post,\n\
 __global const double* __restrict const node_mass_pre,\n\
 __global const double* __restrict const mom_flux,\n\
 __global       double* __restrict const yvel1)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)\n\
    {\n\
        yvel1[THARR2D(0, 0, 1)] = (yvel1[THARR2D(0, 0, 1)]\n\
            * node_mass_pre[THARR2D(0, 0, 1)] + mom_flux[THARR2D(0, -1, 1)]\n\
            - mom_flux[THARR2D(0, 0, 1)]) / node_mass_post[THARR2D(0, 0, 1)];\n\
    }\n\
}\n\
\n\
");
const std::string src_calc_dt_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void calc_dt\n\
(const double g_small,\n\
 const double g_big,\n\
 const double dtmin,\n\
 const double dtc_safe,\n\
 const double dtu_safe,\n\
 const double dtv_safe,\n\
 const double dtdiv_safe,\n\
\n\
 __global const double * __restrict const xarea,\n\
 __global const double * __restrict const yarea,\n\
 __global const double * __restrict const celldx,\n\
 __global const double * __restrict const celldy,\n\
 __global const double * __restrict const volume,\n\
 __global const double * __restrict const density0,\n\
 __global const double * __restrict const viscosity,\n\
 __global const double * __restrict const soundspeed,\n\
 __global const double * __restrict const xvel0,\n\
 __global const double * __restrict const yvel0,\n\
\n\
 __global       double * __restrict const jk_ctrl_out,\n\
 __global       double * __restrict const dt_min_out)\n\
{\n\
    __kernel_indexes;\n\
\n\
    double dsx, dsy;\n\
    double cc;\n\
    double dtct;\n\
    double div;\n\
    double dv1;\n\
    double dv2;\n\
    double dtut;\n\
    double dtvt;\n\
    double dtdivt;\n\
\n\
    //reduced\n\
    double dt_min_val = g_big;\n\
    double jk_control = 0.0;\n\
\n\
#if defined(NO_KERNEL_REDUCTIONS)\n\
    dt_min_out[gid] = dt_min_val;\n\
    jk_ctrl_out[gid] = jk_control;\n\
#else\n\
    __local double dt_min_shared[BLOCK_SZ];\n\
    __local double jk_ctrl_shared[BLOCK_SZ];\n\
    dt_min_shared[lid] = dt_min_val;\n\
    jk_ctrl_shared[lid] = jk_control;\n\
#endif\n\
\n\
    if(row >= (y_min + 1) && row <= (y_max + 1)\n\
    && column >= (x_min + 1) && column <= (x_max + 1))\n\
    {\n\
        dsx = celldx[column];\n\
        dsy = celldy[row];\n\
\n\
        cc = soundspeed[THARR2D(0, 0, 0)] * soundspeed[THARR2D(0, 0, 0)];\n\
        cc += 2.0 * viscosity[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)];\n\
        cc = MAX(SQRT(cc), g_small);\n\
\n\
        dtct = dtc_safe * MIN(dsx, dsy)/cc;\n\
\n\
        div = 0.0;\n\
\n\
        //x\n\
        dv1 = (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)])\n\
            * xarea[THARR2D(0, 0, 1)];\n\
        dv2 = (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)])\n\
            * xarea[THARR2D(1, 0, 1)];\n\
\n\
        div += dv2 - dv1;\n\
\n\
        dtut = dtu_safe * 2.0 * volume[THARR2D(0, 0, 0)]\n\
            / MAX(g_small*volume[THARR2D(0, 0, 0)], \n\
            MAX(fabs(dv1), fabs(dv2)));\n\
\n\
        //y\n\
        dv1 = (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)])\n\
            * yarea[THARR2D(0, 0, 0)];\n\
        dv2 = (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)])\n\
            * yarea[THARR2D(0, 1, 0)];\n\
\n\
        div += dv2 - dv1;\n\
\n\
        dtvt = dtv_safe * 2.0 * volume[THARR2D(0, 0, 0)]\n\
            / MAX(g_small*volume[THARR2D(0, 0, 0)], \n\
            MAX(fabs(dv1), fabs(dv2)));\n\
\n\
        //\n\
        div /= (2.0 * volume[THARR2D(0, 0, 0)]);\n\
\n\
        dtdivt = (div < (-g_small)) ? dtdiv_safe * (-1.0/div) : g_big;\n\
\n\
#if defined(NO_KERNEL_REDUCTIONS)\n\
        dt_min_out[gid] = MIN(dtdivt, MIN(dtvt, MIN(dtct, dtut)));\n\
        jk_ctrl_out[gid] = (column + (x_max * (row - 1))) + 0.4;\n\
#else\n\
        dt_min_shared[lid] = MIN(dtdivt, MIN(dtvt, MIN(dtct, dtut)));\n\
        jk_ctrl_shared[lid] = (column + (x_max * (row - 1))) + 0.4;\n\
#endif\n\
    }\n\
\n\
#if !defined(NO_KERNEL_REDUCTIONS)\n\
    REDUCTION(dt_min_shared, dt_min_out, MIN)\n\
    REDUCTION(jk_ctrl_shared, jk_ctrl_out, MAX)\n\
#endif\n\
}\n\
\n\
");
const std::string src_field_summary_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void field_summary\n\
(__global const double * __restrict const volume,\n\
 __global const double * __restrict const density0,\n\
 __global const double * __restrict const energy0,\n\
 __global const double * __restrict const pressure,\n\
 __global const double * __restrict const xvel0,\n\
 __global const double * __restrict const yvel0,\n\
\n\
 __global       double * __restrict const vol,\n\
 __global       double * __restrict const mass,\n\
 __global       double * __restrict const ie,\n\
 __global       double * __restrict const ke,\n\
 __global       double * __restrict const press)\n\
{\n\
    __kernel_indexes;\n\
\n\
#if defined(NO_KERNEL_REDUCTIONS)\n\
    vol[gid] = 0.0;\n\
    mass[gid] = 0.0;\n\
    ie[gid] = 0.0;\n\
    ke[gid] = 0.0;\n\
    press[gid] = 0.0;\n\
#else\n\
    __local double vol_shared[BLOCK_SZ];\n\
    __local double mass_shared[BLOCK_SZ];\n\
    __local double ie_shared[BLOCK_SZ];\n\
    __local double ke_shared[BLOCK_SZ];\n\
    __local double press_shared[BLOCK_SZ];\n\
    vol_shared[lid] = 0.0;\n\
    mass_shared[lid] = 0.0;\n\
    ie_shared[lid] = 0.0;\n\
    ke_shared[lid] = 0.0;\n\
    press_shared[lid] = 0.0;\n\
#endif\n\
\n\
    if(row >= (y_min + 1) && row <= (y_max + 1)\n\
    && column >= (x_min + 1) && column <= (x_max + 1))\n\
    {\n\
        double vsqrd = 0.0;\n\
\n\
        //unrolled do loop\n\
        vsqrd += 0.25 * (xvel0[THARR2D(0, 0, 1)] * xvel0[THARR2D(0, 0, 1)]\n\
                        +yvel0[THARR2D(0, 0, 1)] * yvel0[THARR2D(0, 0, 1)]);\n\
\n\
        vsqrd += 0.25 * (xvel0[THARR2D(1, 0, 1)] * xvel0[THARR2D(1, 0, 1)]\n\
                        +yvel0[THARR2D(1, 0, 1)] * yvel0[THARR2D(1, 0, 1)]);\n\
\n\
        vsqrd += 0.25 * (xvel0[THARR2D(0, 1, 1)] * xvel0[THARR2D(0, 1, 1)]\n\
                        +yvel0[THARR2D(0, 1, 1)] * yvel0[THARR2D(0, 1, 1)]);\n\
\n\
        vsqrd += 0.25 * (xvel0[THARR2D(1, 1, 1)] * xvel0[THARR2D(1, 1, 1)]\n\
                        +yvel0[THARR2D(1, 1, 1)] * yvel0[THARR2D(1, 1, 1)]);\n\
\n\
        const double cell_vol = volume[THARR2D(0, 0, 0)];\n\
        const double cell_mass = cell_vol * density0[THARR2D(0, 0, 0)];\n\
\n\
#if defined(NO_KERNEL_REDUCTIONS)\n\
        vol[gid] = cell_vol;\n\
        mass[gid] = cell_mass;\n\
        ie[gid] = cell_mass * energy0[THARR2D(0, 0, 0)];\n\
        ke[gid] = cell_mass * 0.5 * vsqrd;\n\
        press[gid] = cell_vol * pressure[THARR2D(0, 0, 0)];\n\
#else\n\
        vol_shared[lid] = cell_vol;\n\
        mass_shared[lid] = cell_mass;\n\
        ie_shared[lid] = cell_mass * energy0[THARR2D(0, 0, 0)];\n\
        ke_shared[lid] = cell_mass * 0.5 * vsqrd;\n\
        press_shared[lid] = cell_vol * pressure[THARR2D(0, 0, 0)];\n\
#endif\n\
    }\n\
\n\
#if !defined(NO_KERNEL_REDUCTIONS)\n\
    REDUCTION(vol_shared, vol, SUM)\n\
    REDUCTION(mass_shared, mass, SUM)\n\
    REDUCTION(ie_shared, ie, SUM)\n\
    REDUCTION(ke_shared, ke, SUM)\n\
    REDUCTION(press_shared, press, SUM)\n\
#endif\n\
}\n\
\n\
");
const std::string src_flux_calc_cl("#include <kernel_files/macros_cl.cl> \n\
__kernel void flux_calc\n\
(double dt,\n\
 __global const double * __restrict const xarea,\n\
 __global const double * __restrict const yarea,\n\
 __global const double * __restrict const xvel0,\n\
 __global const double * __restrict const yvel0,\n\
 __global const double * __restrict const xvel1,\n\
 __global const double * __restrict const yvel1,\n\
 __global       double * __restrict const vol_flux_x,\n\
 __global       double * __restrict const vol_flux_y)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1)\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)\n\
    {\n\
        vol_flux_x[THARR2D(0, 0, 1)] = 0.25 * dt * xarea[THARR2D(0, 0, 1)]\n\
            * (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]\n\
            + xvel1[THARR2D(0, 0, 1)] + xvel1[THARR2D(0, 1, 1)]);\n\
    }\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))\n\
    {\n\
        vol_flux_y[THARR2D(0, 0, 0)] = 0.25 * dt * yarea[THARR2D(0, 0, 0)]\n\
            * (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]\n\
            + yvel1[THARR2D(0, 0, 1)] + yvel1[THARR2D(1, 0, 1)]);\n\
    }\n\
\n\
}\n\
\n\
");
const std::string src_generate_chunk_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void generate_chunk\n\
(__global const double * __restrict const vertexx,\n\
 __global const double * __restrict const vertexy,\n\
 __global const double * __restrict const cellx,\n\
 __global const double * __restrict const celly,\n\
 __global       double * __restrict const density0,\n\
 __global       double * __restrict const energy0,\n\
 __global       double * __restrict const xvel0,\n\
 __global       double * __restrict const yvel0,\n\
\n\
 __global const double * __restrict const state_density,\n\
 __global const double * __restrict const state_energy,\n\
 __global const double * __restrict const state_xvel,\n\
 __global const double * __restrict const state_yvel,\n\
 __global const double * __restrict const state_xmin,\n\
 __global const double * __restrict const state_xmax,\n\
 __global const double * __restrict const state_ymin,\n\
 __global const double * __restrict const state_ymax,\n\
 __global const double * __restrict const state_radius,\n\
 __global const int    * __restrict const state_geometry,\n\
\n\
 const int g_rect,\n\
 const int g_circ,\n\
 const int g_point,\n\
\n\
 const int state)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if (/*row >= (y_min + 1) - 2 &&*/ row <= (y_max + 1) + 2\n\
    && /*column >= (x_min + 1) - 2 &&*/ column <= (x_max + 1) + 2)\n\
    {\n\
        const double x_cent = state_xmin[state];\n\
        const double y_cent = state_ymin[state];\n\
\n\
        if (g_rect == state_geometry[state])\n\
        {\n\
            if (vertexx[1 + column] >= state_xmin[state]\n\
            && vertexx[column] <  state_xmax[state]\n\
            && vertexy[1 + row]    >= state_ymin[state]\n\
            && vertexy[row]    <  state_ymax[state])\n\
            {\n\
                energy0[THARR2D(0, 0, 0)] = state_energy[state];\n\
                density0[THARR2D(0, 0, 0)] = state_density[state];\n\
\n\
                //unrolled do loop\n\
                xvel0[THARR2D(0, 0, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(0, 0, 1)] = state_yvel[state];\n\
\n\
                xvel0[THARR2D(1, 0, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(1, 0, 1)] = state_yvel[state];\n\
\n\
                xvel0[THARR2D(0, 1, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(0, 1, 1)] = state_yvel[state];\n\
\n\
                xvel0[THARR2D(1, 1, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(1, 1, 1)] = state_yvel[state];\n\
            }\n\
        }\n\
        else if (state_geometry[state] == g_circ)\n\
        {\n\
            double x_pos = cellx[column]-x_cent;\n\
            double y_pos = celly[row]-y_cent;\n\
            double radius = SQRT(x_pos*x_pos + y_pos*y_pos);\n\
\n\
            if (radius <= state_radius[state])\n\
            {\n\
                energy0[THARR2D(0, 0, 0)] = state_energy[state];\n\
                density0[THARR2D(0, 0, 0)] = state_density[state];\n\
\n\
                //unrolled do loop\n\
                xvel0[THARR2D(0, 0, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(0, 0, 1)] = state_yvel[state];\n\
\n\
                xvel0[THARR2D(1, 0, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(1, 0, 1)] = state_yvel[state];\n\
\n\
                xvel0[THARR2D(0, 1, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(0, 1, 1)] = state_yvel[state];\n\
\n\
                xvel0[THARR2D(1, 1, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(1, 1, 1)] = state_yvel[state];\n\
            }\n\
        }\n\
        else if (state_geometry[state] == g_point)\n\
        {\n\
            if (vertexx[column] == x_cent && vertexy[row] == y_cent)\n\
            {\n\
                energy0[THARR2D(0, 0, 0)] = state_energy[state];\n\
                density0[THARR2D(0, 0, 0)] = state_density[state];\n\
\n\
                //unrolled do loop\n\
                xvel0[THARR2D(0, 0, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(0, 0, 1)] = state_yvel[state];\n\
\n\
                xvel0[THARR2D(1, 0, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(1, 0, 1)] = state_yvel[state];\n\
\n\
                xvel0[THARR2D(0, 1, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(0, 1, 1)] = state_yvel[state];\n\
\n\
                xvel0[THARR2D(1, 1, 1)] = state_xvel[state];\n\
                yvel0[THARR2D(1, 1, 1)] = state_yvel[state];\n\
            }\n\
        }\n\
    }\n\
}\n\
\n\
__kernel void generate_chunk_init\n\
(__global       double * density0,\n\
 __global       double * energy0,\n\
 __global       double * xvel0,\n\
 __global       double * yvel0,\n\
 __global const double * state_density,\n\
 __global const double * state_energy,\n\
 __global const double * state_xvel,\n\
 __global const double * state_yvel)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) - 2 &&*/ row <= (y_max + 1) + 2\n\
    && /*column >= (x_min + 1) - 2 &&*/ column <= (x_max + 1) + 2)\n\
    {\n\
        energy0[THARR2D(0, 0, 0)] = state_energy[0];\n\
        density0[THARR2D(0, 0, 0)] = state_density[0];\n\
        xvel0[THARR2D(0, 0, 1)] = state_xvel[0];\n\
        yvel0[THARR2D(0, 0, 1)] = state_yvel[0];\n\
    }\n\
}\n\
\n\
");
const std::string src_halo_bare_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void update_halo_bottom_BARE\n\
(int depth,\n\
 __global double * __restrict const cur_array)\n\
{\n\
    __kernel_indexes;\n\
\n\
    // offset by 1 if it is anything but a CELL grid\n\
    int b_offset = (GRID_TYPE != CELL_DATA) ? 1 : 0;\n\
\n\
    if (column >= 2 - depth && column <= (x_max + 1) + X_EXTRA + depth)\n\
    {\n\
        if (row < depth)\n\
        {\n\
            const int offset = 2 + b_offset;\n\
\n\
            /*\n\
             * 1 - 2 * row means that row 0 services row 1, and vice versa\n\
             * this means that it can be dispatched with 'depth' rows only\n\
             */\n\
            cur_array[THARR2D(0, 1 - (2 * row), X_EXTRA)] =\n\
                Y_INVERT * cur_array[THARR2D(0, offset, X_EXTRA)];\n\
        }\n\
    }\n\
}\n\
\n\
__kernel void update_halo_top_BARE\n\
(int depth,\n\
 __global double * __restrict const cur_array)\n\
{\n\
    __kernel_indexes;\n\
\n\
    // if x face data, offset source/dest by - 1\n\
    int x_f_offset = (X_FACE) ? 1 : 0;\n\
\n\
    if (column >= 2 - depth && column <= (x_max + 1) + X_EXTRA + depth)\n\
    {\n\
        if (row < depth)\n\
        {\n\
            const int offset = (- row) * 2 - 1 - x_f_offset;\n\
\n\
            cur_array[THARR2D(0, Y_EXTRA + y_max + 2, X_EXTRA)] =\n\
                Y_INVERT * cur_array[THARR2D(0, y_max + 2 + offset, X_EXTRA)];\n\
        }\n\
    }\n\
}\n\
\n\
__kernel void update_halo_left_BARE\n\
(int depth,\n\
 __global double * __restrict const cur_array)\n\
{\n\
    // offset by 1 if it is anything but a CELL grid\n\
    int l_offset = (GRID_TYPE != CELL_DATA) ? 1 : 0;\n\
\n\
    // special indexes for specific depth\n\
    //const int glob_id = threadIdx.x + blockIdx.x * blockDim.x;\n\
    //const int row = glob_id / depth;\n\
    //const int column = glob_id % depth;\n\
    __kernel_indexes;\n\
\n\
    if (row >= 2 - depth && row <= (y_max + 1) + Y_EXTRA + depth)\n\
    {\n\
        // first in row\n\
        const int row_begin = row * (x_max + 4 + X_EXTRA);\n\
\n\
        cur_array[row_begin + (1 - column)] = X_INVERT * cur_array[row_begin + 2 + column + l_offset];\n\
    }\n\
}\n\
\n\
__kernel void update_halo_right_BARE\n\
(int depth,\n\
 __global double * __restrict const cur_array)\n\
{\n\
    // offset source by -1 if its a y face\n\
    int y_f_offset = (Y_FACE) ? 1 : 0;\n\
\n\
    //const int glob_id = threadIdx.x + blockIdx.x * blockDim.x;\n\
    //const int row = glob_id / depth;\n\
    //const int column = glob_id % depth;\n\
    __kernel_indexes;\n\
\n\
    if (row >= 2 - depth && row <= (y_max + 1) + Y_EXTRA + depth)\n\
    {\n\
        const int row_begin = row * (x_max + 4 + X_EXTRA);\n\
\n\
        cur_array[row_begin + x_max + 2 + X_EXTRA + column] = X_INVERT * cur_array[row_begin + x_max + 1 - (column + y_f_offset)];\n\
    }\n\
}\n\
\n\
");
const std::string src_ideal_gas_cl("#include <kernel_files/macros_cl.cl> \n\
__kernel void ideal_gas\n\
(__global const double * __restrict const density,\n\
 __global const double * __restrict const energy,\n\
 __global       double * __restrict const pressure,\n\
 __global       double * __restrict const soundspeed)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if (/*row >= (y_min + 1) &&*/ row <= (y_max + 1)\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))\n\
    {\n\
        double v, pres_by_ener, pres_by_vol, ss_sq;\n\
\n\
        v = 1.0/density[THARR2D(0,0,0)];\n\
\n\
        pressure[THARR2D(0,0,0)] = (1.4 - 1.0)\n\
            *density[THARR2D(0,0,0)]*energy[THARR2D(0,0,0)];\n\
\n\
        pres_by_ener = (1.4 - 1.0)*density[THARR2D(0,0,0)];\n\
\n\
        pres_by_vol = - density[THARR2D(0,0,0)]*pressure[THARR2D(0,0,0)];\n\
\n\
        ss_sq = v*v*(pressure[THARR2D(0,0,0)]*pres_by_ener - pres_by_vol);\n\
\n\
        soundspeed[THARR2D(0,0,0)] = SQRT(ss_sq);\n\
    }\n\
}\n\
\n\
");
const std::string src_initialise_chunk_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void initialise_chunk_first\n\
(const double d_xmin,\n\
 const double d_ymin,\n\
 const double d_dx,\n\
 const double d_dy,\n\
 __global       double * __restrict const vertexx,\n\
 __global       double * __restrict const vertexdx,\n\
 __global       double * __restrict const vertexy,\n\
 __global       double * __restrict const vertexdy)\n\
{\n\
    __kernel_indexes;\n\
\n\
    // fill out x arrays\n\
    if (row == 0 && column <= (x_max + 1) + 3)\n\
    {\n\
        vertexx[column] = d_xmin +\n\
            d_dx*(double)((((int)column) - 1) - x_min);\n\
        vertexdx[column] = d_dx;\n\
    }\n\
\n\
    // fill out y arrays\n\
    if (column == 0 && row <= (y_max + 1) + 3)\n\
    {\n\
        vertexy[row] = d_ymin +\n\
            d_dy*(double)((((int)row) - 1) - y_min);\n\
        vertexdy[row] = d_dy;\n\
    }\n\
}\n\
\n\
__kernel void initialise_chunk_second\n\
(const double d_xmin,\n\
 const double d_ymin,\n\
 const double d_dx,\n\
 const double d_dy,\n\
 __global const double * __restrict const vertexx,\n\
 __global const double * __restrict const vertexdx,\n\
 __global const double * __restrict const vertexy,\n\
 __global const double * __restrict const vertexdy,\n\
 __global       double * __restrict const cellx,\n\
 __global       double * __restrict const celldx,\n\
 __global       double * __restrict const celly,\n\
 __global       double * __restrict const celldy,\n\
 __global       double * __restrict const volume, \n\
 __global       double * __restrict const xarea, \n\
 __global       double * __restrict const yarea)\n\
{\n\
    __kernel_indexes;\n\
\n\
    //fill x arrays\n\
    if (row == 0 && column <= (x_max + 1) + 2)\n\
    {\n\
        cellx[column] = 0.5 * (vertexx[column] + vertexx[column + 1]);\n\
        celldx[column] = d_dx;\n\
    }\n\
\n\
    //fill y arrays\n\
    if (column == 0 && row <= (y_max + 1) + 2)\n\
    {\n\
        celly[row] = 0.5 * (vertexy[row] + vertexy[row + 1]);\n\
        celldy[row] = d_dy;\n\
    }\n\
\n\
    if (/*row >= (y_min + 1) - 2 &&*/ row <= (y_max + 1) + 2\n\
    && /*column >= (x_min + 1) - 2 &&*/ column <= (x_max + 1) + 2)\n\
    {\n\
        volume[THARR2D(0, 0, 0)] = d_dx * d_dy;\n\
        xarea[THARR2D(0, 0, 1)] = d_dy;\n\
        yarea[THARR2D(0, 0, 0)] = d_dx;\n\
    }\n\
}\n\
\n\
");
const std::string src_macros_cl("#include <kernel_files/macros_cl.cl> \n\
#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\
\n\
#ifdef ONED_KERNEL_LAUNCHES\n\
    #define __kernel_indexes                            \\\n\
        const size_t glob_id = get_global_id(0); \\\n\
        const size_t row = glob_id / (x_max + 4);  \\\n\
        const size_t column = glob_id % (x_max + 4); \\\n\
        const size_t lid = get_local_id(0);\n\
#else\n\
    #define __kernel_indexes                            \\\n\
        const size_t column = get_global_id(0);			\\\n\
        const size_t row = get_global_id(1);				\\\n\
        const size_t loc_column = get_local_id(0);			\\\n\
        const size_t loc_row = get_local_id(1);			\\\n\
        const size_t lid = loc_row*LOCAL_X + loc_column;	\\\n\
        const size_t gid = row*get_global_size(0) + column;\n\
#endif\n\
\n\
#define THARR2D(x_offset, y_offset, big_row)        \\\n\
    (                                               \\\n\
      column                      /* horizontal  */ \\\n\
    + row*(x_max + 4)             /* vertical    */ \\\n\
    + (x_offset)                  /* horz offset */ \\\n\
    + (y_offset)*(x_max + 4)      /* vert offset */ \\\n\
    + (big_row)*(row + (y_offset))/* big row   */   \\\n\
    )\n\
\n\
#ifdef CLOVER_NO_BUILTINS\n\
    #define MAX(a,b) (a<b?a:b)\n\
    #define MIN(a,b) (a>b?a:b)\n\
    #define SUM(a,b) (a+b)\n\
    #define SIGN(a,b) (((b) <  (0) && (a > (0))||((b) > (0) && ((a)<(0)))) ? (-a) : (a))\n\
    #define SQRT(a) sqrt(convert_float(a))\n\
#else\n\
    #define MAX(a,b) max(a,b)\n\
    #define MIN(a,b) min(a,b)\n\
    #define SUM(a,b) ((a)+(b))\n\
    #define SIGN(a,b) copysign(a,b)\n\
    #define SQRT(a) sqrt(a)\n\
#endif\n\
\n\
// TODO probably can optimise reductions somehow\n\
#if defined(CL_DEVICE_TYPE_GPU)\n\
\n\
    // binary tree reduction\n\
    #define REDUCTION(in, out, operation)                    \\\n\
        barrier(CLK_LOCAL_MEM_FENCE);                               \\\n\
        for (int offset = BLOCK_SZ / 2; offset > 0; offset /= 2) \\\n\
        {                                                           \\\n\
            if (lid < offset)                                       \\\n\
            {                                                       \\\n\
                in[lid] = operation(in[lid],                        \\\n\
                                    in[lid + offset]);              \\\n\
            }                                                       \\\n\
            barrier(CLK_LOCAL_MEM_FENCE);                           \\\n\
        }                                                           \\\n\
        if(!lid)                                                    \\\n\
        {                                                           \\\n\
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \\\n\
        }\n\
\n\
#elif defined(CL_DEVICE_TYPE_CPU)\n\
\n\
    // loop in first thread\n\
    #define REDUCTION(in, out, operation)                    \\\n\
        barrier(CLK_LOCAL_MEM_FENCE);                               \\\n\
        if (0 == lid)                                               \\\n\
        {                                                           \\\n\
            for (int offset = 1; offset < BLOCK_SZ; offset++)    \\\n\
            {                                                       \\\n\
                in[0] = operation(in[0], in[offset]);               \\\n\
            }                                                       \\\n\
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \\\n\
        }\n\
\n\
#elif defined(CL_DEVICE_TYPE_ACCELERATOR)\n\
\n\
    //#warning Using CPU style reduction for xeon phi - better performance may be obtained by using the NO_KERNEL_REDUCTIONS option which removes the need for the barrier\n\
\n\
    // loop in first thread\n\
    #define REDUCTION(in, out, operation)                    \\\n\
        barrier(CLK_LOCAL_MEM_FENCE);                               \\\n\
        if (0 == lid)                                               \\\n\
        {                                                           \\\n\
            for (int offset = 1; offset < BLOCK_SZ; offset++)    \\\n\
            {                                                       \\\n\
                in[0] = operation(in[0], in[offset]);               \\\n\
            }                                                       \\\n\
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \\\n\
        }\n\
#if 0\n\
\n\
    /*\n\
     *  TODO\n\
     *  \n\
     *  8/16 wide vector units\n\
     *  4 cores per thing\n\
     *  57-61 cpus\n\
     */\n\
\n\
    #if 0\n\
    #define REDUCTION(in, out, operation)                    \\\n\
        barrier(CLK_LOCAL_MEM_FENCE);                               \\\n\
        for (size_t offset = BLOCK_SZ / 2; offset > 0; offset /= 2) \\\n\
        {                                                           \\\n\
            if (lid < offset)                                       \\\n\
            {                                                       \\\n\
                in[lid] = operation(in[lid],                        \\\n\
                                    in[lid + offset]);              \\\n\
            }                                                       \\\n\
            barrier(CLK_LOCAL_MEM_FENCE);                           \\\n\
        }                                                           \\\n\
        if(!lid)                                                    \\\n\
        {                                                           \\\n\
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \\\n\
        }\n\
    #else\n\
    #define REDUCTION(in, out, operation)                    \\\n\
    { \\\n\
        barrier(CLK_LOCAL_MEM_FENCE);                               \\\n\
        const size_t vecsz = 512/(sizeof(in[0])*8);  \\\n\
        const size_t redsz = vecsz*2; \\\n\
        size_t remain = BLOCK_SZ ; \\\n\
        do \\\n\
        { \\\n\
            if (!(lid % redsz) && lid < remain)                \\\n\
            {                                                           \\\n\
                /*for (size_t offset = 1; offset < redsz; offset++)    \\\n\
                {                                                       \\\n\
                    in[lid] = operation(in[lid], in[lid + offset]);               \\\n\
                }*/                                                       \\\n\
                /*in[0 + lid] = operation(in[0 + lid], in[0 + lid+vecsz]); \\\n\
                in[1 + lid] = operation(in[1 + lid], in[1 + lid+vecsz]); \\\n\
                in[2 + lid] = operation(in[2 + lid], in[2 + lid+vecsz]); \\\n\
                in[3 + lid] = operation(in[3 + lid], in[3 + lid+vecsz]); \\\n\
                in[4 + lid] = operation(in[4 + lid], in[4 + lid+vecsz]); \\\n\
                in[5 + lid] = operation(in[5 + lid], in[5 + lid+vecsz]); \\\n\
                in[6 + lid] = operation(in[6 + lid], in[6 + lid+vecsz]); \\\n\
                in[7 + lid] = operation(in[7 + lid], in[7 + lid+vecsz]);*/ \\\n\
                for (size_t offset = 0; offset < vecsz; offset++)    \\\n\
                {                                                       \\\n\
                    in[offset + lid] = operation(in[offset + lid], in[offset + lid+vecsz]); \\\n\
                }                                                       \\\n\
                barrier(CLK_LOCAL_MEM_FENCE);                               \\\n\
                in[lid/redsz] = in[lid]; \\\n\
            } \\\n\
            else \\\n\
            { \\\n\
                barrier(CLK_LOCAL_MEM_FENCE);                               \\\n\
                break;\\\n\
            } \\\n\
        } while ((remain = remain/redsz) >= redsz); \\\n\
        out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \\\n\
        /*if (0 == lid)                                               \\\n\
        {                                                           \\\n\
            for (size_t offset = 1; offset < BLOCK_SZ/vecsz; offset++)    \\\n\
            {                                                       \\\n\
                in[0] = operation(in[0], in[offset]);               \\\n\
            }                                                       \\\n\
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \\\n\
        }*/ \\\n\
        /*for (size_t offset = BLOCK_SZ / (2*vecsz); offset > 0; offset /= 2) \\\n\
        {                                                           \\\n\
            if (lid < offset)                                       \\\n\
            {                                                       \\\n\
                in[lid] = operation(in[lid],                        \\\n\
                                    in[lid + offset]);              \\\n\
            }                                                       \\\n\
            barrier(CLK_LOCAL_MEM_FENCE);                           \\\n\
        }                                                           \\\n\
        if(!lid)                                                    \\\n\
        {                                                           \\\n\
            out[get_group_id(1)*get_num_groups(0) + get_group_id(0)] = in[0]; \\\n\
        }*/ \\\n\
    }\n\
    #endif\n\
#endif\n\
\n\
#else\n\
\n\
    #error No device type specified - don't know which reduction to use\n\
\n\
#endif\n\
\n\
");
const std::string src_PdV_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void PdV_predict\n\
(double dt,\n\
 __global          int * __restrict const error_condition,\n\
 __global const double * __restrict const xarea,\n\
 __global const double * __restrict const yarea,\n\
 __global const double * __restrict const volume,\n\
 __global const double * __restrict const density0,\n\
 __global       double * __restrict const density1,\n\
 __global const double * __restrict const energy0,\n\
 __global       double * __restrict const energy1,\n\
 __global const double * __restrict const pressure,\n\
 __global const double * __restrict const viscosity,\n\
 __global const double * __restrict const xvel0,\n\
 __global const double * __restrict const yvel0,\n\
 __global const double * __restrict const xvel1,\n\
 __global const double * __restrict const yvel1)\n\
{\n\
    __kernel_indexes;\n\
\n\
#if defined(NO_KERNEL_REDUCTIONS)\n\
    error_condition[gid] = 0;\n\
#else\n\
    __local int err_cond_kernel[BLOCK_SZ];\n\
    err_cond_kernel[lid] = 0;\n\
#endif\n\
\n\
    double volume_change;\n\
    double recip_volume, energy_change, min_cell_volume,\n\
        right_flux, left_flux, top_flux, bottom_flux, total_flux;\n\
    \n\
    if (/*row >= (y_min + 1) &&*/ row <= (y_max + 1)\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))\n\
    {\n\
        left_flux   = (xarea[THARR2D(0, 0, 1)]\n\
            * (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)] \n\
            + xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]))\n\
            * 0.25 * dt * 0.5;\n\
        right_flux  = (xarea[THARR2D(1, 0, 1)]\n\
            * (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)] \n\
            + xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)]))\n\
            * 0.25 * dt * 0.5;\n\
\n\
        bottom_flux = (yarea[THARR2D(0, 0, 0)]\n\
            * (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)] \n\
            + yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]))\n\
            * 0.25 * dt * 0.5;\n\
        top_flux    = (yarea[THARR2D(0, 1, 0)]\n\
            * (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)] \n\
            + yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)]))\n\
            * 0.25 * dt * 0.5;\n\
\n\
        total_flux = right_flux - left_flux + top_flux - bottom_flux;\n\
\n\
        volume_change = volume[THARR2D(0, 0, 0)]\n\
            / (volume[THARR2D(0, 0, 0)] + total_flux);\n\
\n\
        //minimum of total, horizontal, and vertical flux\n\
        min_cell_volume = \n\
            MIN(volume[THARR2D(0, 0, 0)] + total_flux,\n\
            MIN(volume[THARR2D(0, 0, 0)] + right_flux - left_flux,\n\
                volume[THARR2D(0, 0, 0)] + top_flux - bottom_flux));\n\
\n\
#if defined(NO_KERNEL_REDUCTIONS)\n\
        if(volume_change <= 0.0)\n\
        {\n\
            error_condition[gid] = 1;\n\
        }\n\
        if(min_cell_volume <= 0.0)\n\
        {\n\
            error_condition[gid] = 2;\n\
        }\n\
#else\n\
        if(volume_change <= 0.0)\n\
        {\n\
            err_cond_kernel[lid] = 1;\n\
        }\n\
        if(min_cell_volume <= 0.0)\n\
        {\n\
            err_cond_kernel[lid] = 2;\n\
        }\n\
#endif\n\
\n\
        recip_volume = 1.0/volume[THARR2D(0, 0, 0)];\n\
\n\
        energy_change = ((pressure[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)])\n\
            + (viscosity[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)]))\n\
            * total_flux * recip_volume;\n\
\n\
        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)] - energy_change;\n\
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)] * volume_change;\n\
    }\n\
    \n\
#if !defined(NO_KERNEL_REDUCTIONS)\n\
    REDUCTION(err_cond_kernel, error_condition, MAX)\n\
#endif\n\
}\n\
\n\
__kernel void PdV_not_predict\n\
(double dt,\n\
 __global          int * __restrict const error_condition,\n\
 __global const double * __restrict const xarea,\n\
 __global const double * __restrict const yarea,\n\
 __global const double * __restrict const volume,\n\
 __global const double * __restrict const density0,\n\
 __global       double * __restrict const density1,\n\
 __global const double * __restrict const energy0,\n\
 __global       double * __restrict const energy1,\n\
 __global const double * __restrict const pressure,\n\
 __global const double * __restrict const viscosity,\n\
 __global const double * __restrict const xvel0,\n\
 __global const double * __restrict const yvel0,\n\
 __global const double * __restrict const xvel1,\n\
 __global const double * __restrict const yvel1)\n\
{\n\
    __kernel_indexes;\n\
\n\
#if defined(NO_KERNEL_REDUCTIONS)\n\
    error_condition[gid] = 0;\n\
#else\n\
    __local int err_cond_kernel[BLOCK_SZ];\n\
    err_cond_kernel[lid] = 0;\n\
#endif\n\
\n\
    double volume_change;\n\
    double recip_volume, energy_change, min_cell_volume,\n\
        right_flux, left_flux, top_flux, bottom_flux, total_flux;\n\
    \n\
    if (/*row >= (y_min + 1) &&*/ row <= (y_max + 1)\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))\n\
    {\n\
        left_flux   = (xarea[THARR2D(0, 0, 1)]\n\
            * (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)] \n\
            + xvel1[THARR2D(0, 0, 1)] + xvel1[THARR2D(0, 1, 1)]))\n\
            * 0.25 * dt;\n\
        right_flux  = (xarea[THARR2D(1, 0, 1)]\n\
            * (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)] \n\
            + xvel1[THARR2D(1, 0, 1)] + xvel1[THARR2D(1, 1, 1)]))\n\
            * 0.25 * dt;\n\
\n\
        bottom_flux = (yarea[THARR2D(0, 0, 0)]\n\
            * (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)] \n\
            + yvel1[THARR2D(0, 0, 1)] + yvel1[THARR2D(1, 0, 1)]))\n\
            * 0.25 * dt;\n\
        top_flux    = (yarea[THARR2D(0, 1, 0)]\n\
            * (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)] \n\
            + yvel1[THARR2D(0, 1, 1)] + yvel1[THARR2D(1, 1, 1)]))\n\
            * 0.25 * dt;\n\
\n\
        total_flux = right_flux - left_flux + top_flux - bottom_flux;\n\
\n\
        volume_change = volume[THARR2D(0, 0, 0)]\n\
            / (volume[THARR2D(0, 0, 0)] + total_flux);\n\
\n\
        min_cell_volume =\n\
            MIN(volume[THARR2D(0, 0, 0)] + total_flux,\n\
            MIN(volume[THARR2D(0, 0, 0)] + right_flux - left_flux,\n\
                volume[THARR2D(0, 0, 0)] + top_flux - bottom_flux));\n\
\n\
#if defined(NO_KERNEL_REDUCTIONS)\n\
        if(volume_change <= 0.0)\n\
        {\n\
            error_condition[gid] = 1;\n\
        }\n\
        if(min_cell_volume <= 0.0)\n\
        {\n\
            error_condition[gid] = 2;\n\
        }\n\
#else\n\
        if(volume_change <= 0.0)\n\
        {\n\
            err_cond_kernel[lid] = 1;\n\
        }\n\
        if(min_cell_volume <= 0.0)\n\
        {\n\
            err_cond_kernel[lid] = 2;\n\
        }\n\
#endif\n\
\n\
        recip_volume = 1.0/volume[THARR2D(0, 0, 0)];\n\
\n\
        energy_change = ((pressure[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)])\n\
            + (viscosity[THARR2D(0, 0, 0)] / density0[THARR2D(0, 0, 0)]))\n\
            * total_flux * recip_volume;\n\
\n\
        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)] - energy_change;\n\
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)] * volume_change;\n\
\n\
    }\n\
\n\
#if !defined(NO_KERNEL_REDUCTIONS)\n\
    REDUCTION(err_cond_kernel, error_condition, MAX)\n\
#endif\n\
}\n\
\n\
");
const std::string src_reduction_cl("#include <kernel_files/macros_cl.cl> \n\
#ifdef cl_khr_fp64\n\
#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\
#elif defined(cl_amd_fp64)\n\
#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n\
#endif\n\
\n\
/*\n\
 *  Need to be defined:\n\
 *  reduce_t = array type to reduce (double or int)\n\
 *  REDUCE = operation to do\n\
 *  INIT_RED_VAL = initial value before loading - eg, 0 for a sum\n\
 *  LOCAL_SZ = local work group size, 1D\n\
 */\n\
\n\
#if defined(red_sum)\n\
inline reduce_t REDUCE\n\
(reduce_t a, reduce_t b)\n\
{\n\
    return SUM(a,b);\n\
}\n\
#elif defined (red_min)\n\
inline reduce_t REDUCE\n\
(reduce_t a, reduce_t b)\n\
{\n\
    return MIN(a,b);\n\
}\n\
#elif defined (red_max)\n\
inline reduce_t REDUCE\n\
(reduce_t a, reduce_t b)\n\
{\n\
    return MAX(a,b);\n\
}\n\
#else\n\
    #error No definition for reduction\n\
#endif\n\
\n\
__kernel void reduction\n\
(__global       reduce_t * const __restrict input)\n\
{\n\
    const int lid = get_local_id(0);\n\
    const int gid = get_global_id(0);\n\
\n\
    __local reduce_t scratch[LOCAL_SZ];\n\
\n\
    // initialises to some initial value - different for different reductions\n\
    scratch[lid] = INIT_RED_VAL;\n\
\n\
    /*\n\
     *  Read and write to two opposite halves of the reduction buffer so that\n\
     *  there are no data races with reduction. eg, first stages reads from\n\
     *  first half of buffer and writes results into second half, then second\n\
     *  stage of reduction reads from second half and writes back into first,\n\
     *  etc.\n\
     */\n\
    size_t dest_offset;\n\
    size_t src_offset;\n\
\n\
    if (!(RED_STAGE % 2))\n\
    {\n\
        src_offset = ORIG_ELEMS_TO_REDUCE;\n\
        dest_offset = 0;\n\
    }\n\
    else\n\
    {\n\
        src_offset = 0;\n\
        dest_offset = ORIG_ELEMS_TO_REDUCE;\n\
    }\n\
\n\
    /*\n\
     *  If the number of elements to reduce is not a power of 2 then 2 values\n\
     *  can be loaded for an initial reduction for some threads but not for\n\
     *  others, defined by the threshold corresponding to the difference between\n\
     *  the next power of 2 up from the number to reduce\n\
     *\n\
     *  if there are 900 values to reduce with a 256 local size, then launch 2\n\
     *  groups of total thread count 512, and load 2 values to reduce on load in\n\
     *  the first 250 of these threads\n\
     */\n\
    if (0&&gid < RED_LOAD_THRESHOLD)\n\
    // FIXME this isn't working properly for now - just load one per thread\n\
    {\n\
        // TODO when this is fixed then do it in a vector for xeon phi things?\n\
        // load + reduce at the same time\n\
        scratch[lid] = REDUCE(input[src_offset + gid],\n\
            input[src_offset + gid + GLOBAL_SZ]);\n\
    }\n\
    else if (gid < ELEMS_TO_REDUCE)\n\
    {\n\
        // just load\n\
        scratch[lid] = input[src_offset + gid];\n\
    }\n\
\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
#if defined(CL_DEVICE_TYPE_GPU)\n\
\n\
    for (int offset = LOCAL_SZ / 2; offset > 0; offset /= 2)\n\
    {\n\
        if (lid < offset)\n\
        {\n\
            scratch[lid] = REDUCE(scratch[lid], scratch[lid + offset]);\n\
        }\n\
        barrier(CLK_LOCAL_MEM_FENCE);\n\
    }\n\
\n\
#elif defined(CL_DEVICE_TYPE_CPU)\n\
\n\
    if (0 == lid)\n\
    {\n\
        for (int offset = 1; offset < LOCAL_SZ; offset++)\n\
        {\n\
            scratch[0] = REDUCE(scratch[0], scratch[offset]);\n\
        }\n\
    }\n\
\n\
#elif defined(CL_DEVICE_TYPE_ACCELERATOR)\n\
\n\
    // TODO special reductions for xeon phi in some fashion\n\
    if (0 == lid)\n\
    {\n\
        for (int offset = 1; offset < LOCAL_SZ; offset++)\n\
        {\n\
            scratch[0] = REDUCE(scratch[0], scratch[offset]);\n\
        }\n\
    }\n\
\n\
#else\n\
\n\
    #error No device type specified for reduction\n\
\n\
#endif\n\
\n\
    if (0 == lid)\n\
    {\n\
        if (LOCAL_SZ == GLOBAL_SZ)\n\
        {\n\
            // last stage - write back into 0 - no chance of data race\n\
            input[0] = scratch[0];\n\
        }\n\
        else\n\
        {\n\
            input[dest_offset + get_group_id(0)] = scratch[0];\n\
        }\n\
    }\n\
}\n\
\n\
");
const std::string src_reset_field_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void reset_field\n\
(__global       double* __restrict const density0,\n\
 __global const double* __restrict const density1,\n\
 __global       double* __restrict const energy0,\n\
 __global const double* __restrict const energy1,\n\
 __global       double* __restrict const xvel0,\n\
 __global const double* __restrict const xvel1,\n\
 __global       double* __restrict const yvel0,\n\
 __global const double* __restrict const yvel1)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1) + 1\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1) + 1)\n\
    {\n\
        xvel0[THARR2D(0, 0, 1)] = xvel1[THARR2D(0, 0, 1)];\n\
        yvel0[THARR2D(0, 0, 1)] = yvel1[THARR2D(0, 0, 1)];\n\
\n\
        //if(row <= (y_max + 1)\n\
        //&& column <= (x_max + 1))\n\
        {\n\
            density0[THARR2D(0, 0, 0)] = density1[THARR2D(0, 0, 0)];\n\
            energy0[THARR2D(0, 0, 0)]  = energy1[THARR2D(0, 0, 0)];\n\
        }\n\
    }\n\
}\n\
\n\
");
const std::string src_revert_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void revert\n\
(__global const double * __restrict const density0,\n\
 __global       double * __restrict const density1,\n\
 __global const double * __restrict const energy0,\n\
 __global       double * __restrict const energy1)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(/*row >= (y_min + 1) &&*/ row <= (y_max + 1)\n\
    && /*column >= (x_min + 1) &&*/ column <= (x_max + 1))\n\
    {\n\
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)];\n\
        energy1[THARR2D(0, 0, 0)] = energy0[THARR2D(0, 0, 0)];\n\
    }\n\
}\n\
\n\
");
const std::string src_set_field_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void set_field\n\
(__global const double* __restrict const density0,\n\
 __global       double* __restrict const density1,\n\
 __global const double* __restrict const energy0,\n\
 __global       double* __restrict const energy1)\n\
{\n\
    __kernel_indexes;\n\
\n\
    if(row >= (y_min + 1) && row <= (y_max + 1)\n\
    && column >= (x_min + 1) && column <= (x_max + 1))\n\
    {\n\
        density1[THARR2D(0, 0, 0)] = density0[THARR2D(0, 0, 0)];\n\
        energy1[THARR2D(0, 0, 0)]  = energy0[THARR2D(0, 0, 0)];\n\
    }\n\
}\n\
\n\
");
const std::string src_update_halo_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void update_halo_bottom\n\
(int x_extra,   int y_extra,\n\
 int x_invert,  int y_invert,\n\
 int x_face,    int y_face,\n\
 int grid_type, int depth, \n\
 __global double * __restrict const cur_array)\n\
{\n\
    __kernel_indexes;\n\
\n\
    // offset by 1 if it is anything but a CELL grid\n\
    int b_offset = (grid_type != CELL_DATA) ? 1 : 0;\n\
\n\
    if (column >= 2 - depth && column <= (x_max + 1) + x_extra + depth)\n\
    {\n\
        if (row < depth)\n\
        {\n\
            const int offset = 2 + b_offset;\n\
\n\
            /*\n\
             * 1 - 2 * row means that row 0 services row 1, and vice versa\n\
             * this means that it can be dispatched with 'depth' rows only\n\
             */\n\
            cur_array[THARR2D(0, 1 - (2 * row), x_extra)] =\n\
                y_invert * cur_array[THARR2D(0, offset, x_extra)];\n\
        }\n\
    }\n\
}\n\
\n\
__kernel void update_halo_top\n\
(int x_extra, int y_extra,\n\
 int x_invert, int y_invert,\n\
 int x_face, int y_face,\n\
 int grid_type, int depth, \n\
 __global double * __restrict const cur_array)\n\
{\n\
    __kernel_indexes;\n\
\n\
    // if x face data, offset source/dest by - 1\n\
    int x_f_offset = (x_face) ? 1 : 0;\n\
\n\
    if (column >= 2 - depth && column <= (x_max + 1) + x_extra + depth)\n\
    {\n\
        if (row < depth)\n\
        {\n\
            const int offset = (- row) * 2 - 1 - x_f_offset;\n\
\n\
            cur_array[THARR2D(0, y_extra + y_max + 2, x_extra)] =\n\
                y_invert * cur_array[THARR2D(0, y_max + 2 + offset, x_extra)];\n\
        }\n\
    }\n\
}\n\
\n\
__kernel void update_halo_left\n\
(int x_extra, int y_extra,\n\
 int x_invert, int y_invert,\n\
 int x_face, int y_face,\n\
 int grid_type, int depth, \n\
 __global double * __restrict const cur_array)\n\
{\n\
    // offset by 1 if it is anything but a CELL grid\n\
    int l_offset = (grid_type != CELL_DATA) ? 1 : 0;\n\
\n\
    __kernel_indexes;\n\
\n\
    if (row >= 2 - depth && row <= (y_max + 1) + y_extra + depth)\n\
    {\n\
        // first in row\n\
        const int row_begin = row * (x_max + 4 + x_extra);\n\
\n\
        cur_array[row_begin + (1 - column)] = x_invert * cur_array[row_begin + 2 + column + l_offset];\n\
    }\n\
}\n\
\n\
__kernel void update_halo_right\n\
(int x_extra, int y_extra,\n\
 int x_invert, int y_invert,\n\
 int x_face, int y_face,\n\
 int grid_type, int depth, \n\
 __global double * __restrict const cur_array)\n\
{\n\
    // offset source by -1 if its a y face\n\
    int y_f_offset = (y_face) ? 1 : 0;\n\
\n\
    __kernel_indexes;\n\
\n\
    if (row >= 2 - depth && row <= (y_max + 1) + y_extra + depth)\n\
    {\n\
        const int row_begin = row * (x_max + 4 + x_extra);\n\
\n\
        cur_array[row_begin + x_max + 2 + x_extra + column] = x_invert * cur_array[row_begin + x_max + 1 - (column + y_f_offset)];\n\
    }\n\
}\n\
\n\
");
const std::string src_viscosity_cl("#include <kernel_files/macros_cl.cl> \n\
\n\
__kernel void viscosity\n\
(__global const double * __restrict const celldx,\n\
 __global const double * __restrict const celldy,\n\
 __global const double * __restrict const density0,\n\
 __global const double * __restrict const pressure,\n\
 __global       double * __restrict const viscosity,\n\
 __global const double * __restrict const xvel0,\n\
 __global const double * __restrict const yvel0)\n\
{\n\
    __kernel_indexes;\n\
\n\
    double ugrad, vgrad, grad2, pgradx, pgrady, pgradx2, pgrady2,\n\
        grad, ygrad, pgrad, xgrad, div, strain2, limiter;\n\
\n\
    if(/*row >= (y_min + 1) && */ row <= (y_max + 1)\n\
    && /*column >= (x_min + 1) && */ column <= (x_max + 1))\n\
    {\n\
        ugrad = (xvel0[THARR2D(1, 0, 1)] + xvel0[THARR2D(1, 1, 1)])\n\
              - (xvel0[THARR2D(0, 0, 1)] + xvel0[THARR2D(0, 1, 1)]);\n\
\n\
        vgrad = (yvel0[THARR2D(0, 1, 1)] + yvel0[THARR2D(1, 1, 1)])\n\
              - (yvel0[THARR2D(0, 0, 1)] + yvel0[THARR2D(1, 0, 1)]);\n\
        \n\
        div = (celldx[column] * ugrad) + (celldy[row] * vgrad);\n\
\n\
        strain2 = 0.5 * (xvel0[THARR2D(0, 1, 1)] + xvel0[THARR2D(1, 1, 1)]\n\
                - xvel0[THARR2D(0, 0, 1)] - xvel0[THARR2D(1, 0, 1)])/celldy[row]\n\
                + 0.5 * (yvel0[THARR2D(1, 0, 1)] + yvel0[THARR2D(1, 1, 1)]\n\
                - yvel0[THARR2D(0, 0, 1)] - yvel0[THARR2D(0, 1, 1)])/celldx[column];\n\
\n\
        pgradx = (pressure[THARR2D(1, 0, 0)] - pressure[THARR2D(-1, 0, 0)])\n\
               / (celldx[column] + celldx[column + 1]);\n\
        pgrady = (pressure[THARR2D(0, 1, 0)] - pressure[THARR2D(0, -1, 0)])\n\
               / (celldy[row] + celldy[row + 1]);\n\
\n\
        pgradx2 = pgradx*pgradx;\n\
        pgrady2 = pgrady*pgrady;\n\
\n\
        limiter = ((0.5 * ugrad / celldx[column]) * pgradx2\n\
                + ((0.5 * vgrad / celldy[row]) * pgrady2)\n\
                + (strain2 * pgradx * pgrady))\n\
                / MAX(pgradx2 + pgrady2, 1.0e-16);\n\
\n\
\n\
        pgradx = SIGN(MAX(1.0e-16, fabs(pgradx)), pgradx);\n\
        pgrady = SIGN(MAX(1.0e-16, fabs(pgrady)), pgrady);\n\
        pgrad = SQRT((pgradx * pgradx) + (pgrady * pgrady));\n\
\n\
        xgrad = fabs(celldx[column] * pgrad / pgradx);\n\
        ygrad = fabs(celldy[row] * pgrad / pgrady);\n\
\n\
        grad = MIN(xgrad, ygrad);\n\
        grad2 = grad * grad;\n\
\n\
        if(limiter > 0 || div >= 0.0)\n\
        {\n\
            viscosity[THARR2D(0,0,0)] = 0.0;\n\
        }\n\
        else\n\
        {\n\
            viscosity[THARR2D(0,0,0)] = 2.0 * density0[THARR2D(0,0,0)] * grad2 * (limiter * limiter);\n\
        }\n\
    }\n\
}\n\
\n\
");
