#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void revert_kernel_ocl_
(void)
{
    chunk.revert_kernel();
}

void CloverChunk::revert_kernel
(void)
{
    //ENQUEUE(revert_device)
    ENQUEUE_OFFSET(revert_device)
}
