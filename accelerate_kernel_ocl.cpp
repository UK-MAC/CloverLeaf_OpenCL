#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void accelerate_kernel_ocl_
(double *dbyt)
{
    chunk.accelerate_kernel(*dbyt);
}

void CloverChunk::accelerate_kernel
(double dbyt)
{
    accelerate_device.setArg(0, dbyt);

    //ENQUEUE(accelerate_device)
    ENQUEUE_OFFSET(accelerate_device)
}
