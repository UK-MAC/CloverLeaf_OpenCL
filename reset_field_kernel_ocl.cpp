#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void reset_field_kernel_ocl_
(void)
{
    chunk.reset_field_kernel();
}

void CloverChunk::reset_field_kernel
(void)
{
    //ENQUEUE(reset_field_device)
    ENQUEUE_OFFSET(reset_field_device)
}
