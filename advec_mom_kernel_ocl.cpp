#include "ocl_common.hpp"

extern "C" void advec_mom_kernel_ocl_
(int *whch_vl, int *swp_nmbr, int *drctn)
{
    chunk.advec_mom_kernel(*whch_vl, *swp_nmbr, *drctn);
}

void CloverChunk::advec_mom_kernel
(int which_vel, int sweep_number, int direction)
{
    int mom_sweep = direction + (2 * (sweep_number - 1));

    advec_mom_vol_device.setArg(0, mom_sweep);

    //ENQUEUE(advec_mom_vol_device);
    ENQUEUE_OFFSET(advec_mom_vol_device);

    if (1 == which_vel)
    {
        advec_mom_flux_x_device.setArg(3, xvel1);
        advec_mom_xvel_device.setArg(3, xvel1);
        advec_mom_flux_y_device.setArg(3, xvel1);
        advec_mom_yvel_device.setArg(3, xvel1);
    }
    else
    {
        advec_mom_flux_x_device.setArg(3, yvel1);
        advec_mom_xvel_device.setArg(3, yvel1);
        advec_mom_flux_y_device.setArg(3, yvel1);
        advec_mom_yvel_device.setArg(3, yvel1);
    }

    if (1 == direction)
    {
        //ENQUEUE(advec_mom_node_flux_post_x_1_device);
        ENQUEUE_OFFSET(advec_mom_node_flux_post_x_1_device);
        //ENQUEUE(advec_mom_node_flux_post_x_2_device);
        ENQUEUE_OFFSET(advec_mom_node_flux_post_x_2_device);

        //ENQUEUE(advec_mom_node_pre_x_device);
        ENQUEUE_OFFSET(advec_mom_node_pre_x_device);

        //ENQUEUE(advec_mom_flux_x_device);
        ENQUEUE_OFFSET(advec_mom_flux_x_device);

        //ENQUEUE(advec_mom_xvel_device);
        ENQUEUE_OFFSET(advec_mom_xvel_device);
    }
    else if (2 == direction)
    {
        //ENQUEUE(advec_mom_node_flux_post_y_1_device);
        ENQUEUE_OFFSET(advec_mom_node_flux_post_y_1_device);
        //ENQUEUE(advec_mom_node_flux_post_y_2_device);
        ENQUEUE_OFFSET(advec_mom_node_flux_post_y_2_device);

        //ENQUEUE(advec_mom_node_pre_y_device);
        ENQUEUE_OFFSET(advec_mom_node_pre_y_device);

        //ENQUEUE(advec_mom_flux_y_device);
        ENQUEUE_OFFSET(advec_mom_flux_y_device);

        //ENQUEUE(advec_mom_yvel_device);
        ENQUEUE_OFFSET(advec_mom_yvel_device);
    }
}

