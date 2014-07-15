#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void ideal_gas_kernel_nopredict_ocl_
(void)
{
    chunk.ideal_gas_kernel(0);
}

extern "C" void ideal_gas_kernel_predict_ocl_
(void)
{
    chunk.ideal_gas_kernel(1);
}

void CloverChunk::ideal_gas_kernel
(int predict)
{
    /*
* For this and similar kernels, could just launch it with a 960x960 global
* size and offset the global ids so that no range checking needs to be
* done, but it makes the kernels inconsistent so might as well leave them
* as they are.
*/
    if (1 == predict)
    {
        ideal_gas_device.setArg(0, density1);
        ideal_gas_device.setArg(1, energy1);

        //ENQUEUE(ideal_gas_device)
        ENQUEUE_OFFSET(ideal_gas_device)
    }
    else
    {
        ideal_gas_device.setArg(0, density0);
        ideal_gas_device.setArg(1, energy0);

        //ENQUEUE(ideal_gas_device)
        ENQUEUE_OFFSET(ideal_gas_device)
    }
}
