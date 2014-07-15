#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void viscosity_kernel_ocl_
(void)
{
    chunk.viscosity_kernel();
}

void CloverChunk::viscosity_kernel
(void)
{
    //ENQUEUE(viscosity_device)
    ENQUEUE_OFFSET(viscosity_device)
}
