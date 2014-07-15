#include "ocl_common.hpp"
extern CloverChunk chunk;

extern "C" void set_field_kernel_ocl_
(void)
{
    chunk.set_field_kernel();
}

void CloverChunk::set_field_kernel
(void)
{
    //ENQUEUE(set_field_device)
    ENQUEUE_OFFSET(set_field_device)
}
