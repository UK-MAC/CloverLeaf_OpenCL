#ifdef DONTFIXWGSIZE

    #define ENQUEUE_KERNEL_OOO_MACRO(kernel,x_num,y_num,x_wg_size,y_wg_size) \
        CloverCL::outoforder_queue.enqueueNDRangeKernel(kernel, cl::NullRange, \
                                                        cl::NDRange(x_num, y_num), \
                                                        cl::NullRange, \
                                                        NULL, NULL);

#else

    #define ENQUEUE_KERNEL_OOO_MACRO(kernel,x_num,y_num,x_wg_size,y_wg_size) \
        CloverCL::outoforder_queue.enqueueNDRangeKernel(kernel, cl::NullRange, \
                                                        cl::NDRange(x_num, y_num), \
                                                        cl::NDRange(x_wg_size,y_wg_size), \
                                                        NULL, NULL);
#endif
