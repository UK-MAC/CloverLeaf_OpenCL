#include <cstdio>
#include <string>
#include "ocl_common.hpp"

enum {AMD_PLAT, INTEL_PLAT, NVIDIA_PLAT, NO_PLAT, ANY_PLAT, LIST_PLAT};

/*
 *  reads file given and finds the platform vendor to be used
 */
int platformRead
(FILE* input);

/*
 *  reads file given and finds the platform type
 */
int typeRead
(FILE* input);

/*
 *  takes the platform name from clGetPlatformInfo and matches against know
 *  values to convert to the relevant enumerated value for the vendor
 */
int platformMatch
(std::string& plat_name);

/*
 *  Takes string of type of context and returns enumerated value
 */
int typeMatch
(std::string& type_name);

/*
 *  Takes cl_device_type and returns string (merge into above/bit in ocl_init TODO)
 */
std::string strType
(cl_device_type dtype);

/*
 *  Returns stringified device type
 */
std::string errToString
(cl_int err);

/*
 *  Find if tl_use_cg is in the input file
 */
bool paramEnabled
(FILE* input, const char* param);

/*
 *  Returns index of desired device, or -1 if some error occurs (none specified, invalid specification, etc)
 */
int preferredDevice
(FILE* input);

/*
 *  Find out the value of a parameter
 */
std::string matchParam
(FILE * input,
 const char* param_name);


