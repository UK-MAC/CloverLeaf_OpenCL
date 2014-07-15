#include "ocl_common.hpp"
#include "ocl_reduction.hpp"
extern CloverChunk chunk;

extern "C" void pdv_kernel_ocl_
(int *errorcondition, int *prdct, double *dtbyt)
{
    chunk.PdV_kernel(errorcondition, *prdct, *dtbyt);
}

void CloverChunk::PdV_kernel
(int* error_condition, int predict, double dt)
{
    if (1 == predict)
    {
        PdV_predict_device.setArg(0, dt);

        //ENQUEUE(PdV_predict_device)
        ENQUEUE_OFFSET(PdV_predict_device)
    }
    else
    {
        PdV_not_predict_device.setArg(0, dt);

        //ENQUEUE(PdV_not_predict_device)
        ENQUEUE_OFFSET(PdV_not_predict_device)
    }

    *error_condition = reduceValue<int>(max_red_kernels_int,
                                        PdV_reduce_buf);

    if (1 == *error_condition)
    {
        fprintf(stdout, "Negative volume in PdV kernel\n");
    }
    else if (2 == *error_condition)
    {
        fprintf(stdout, "Negative cell volume in PdV kernel\n");
    }
}
