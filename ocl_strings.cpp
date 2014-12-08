#include "ocl_strings.hpp"

#include <algorithm>
#include <cstring>
#include <sstream>

std::string matchParam
(FILE * input,
 const char* param_name)
{
    std::string param_string;
    param_string = std::string("NO_SETTING");
    static char name_buf[101];
    rewind(input);
    /* read in line from file */
    while (NULL != fgets(name_buf, 100, input))
    {
        if (NULL != strstr(name_buf, "!")) continue;
        /* if it has the parameter name, its the line we want */
        if (NULL != strstr(name_buf, param_name))
        {
            if (NULL != strstr(name_buf, "="))
            {
                *(strstr(name_buf, "=")) = ' ';
                char param_buf[100];
                sscanf(name_buf, "%*s %s", param_buf);
                param_string = std::string(param_buf);
            }
            else
            {
                param_string = std::string(param_name);
            }
            break;
        }
    }

    return param_string;
}

std::string platformRead
(FILE* input)
{
    std::string plat_name = matchParam(input, "opencl_vendor");

    // convert to lower case
    std::transform(plat_name.begin(),
                   plat_name.end(),
                   plat_name.begin(),
                   tolower);

    return plat_name;
}

std::string typeRead
(FILE* input)
{
    std::string type_name = matchParam(input, "opencl_type");

    // convert to lower case
    std::transform(type_name.begin(),
                   type_name.end(),
                   type_name.begin(),
                   tolower);

    return type_name;
}

int typeMatch
(std::string& type_name)
{
    //fprintf(stderr, "Matching with %s\n", type_name.c_str());

    // match
    if (type_name.find("cpu") != std::string::npos)
    {
        return CL_DEVICE_TYPE_CPU;
    }
    else if (type_name.find("gpu") != std::string::npos)
    {
        return CL_DEVICE_TYPE_GPU;
    }
    else if (type_name.find("accelerator") != std::string::npos)
    {
        return CL_DEVICE_TYPE_ACCELERATOR;
    }
    else if (type_name.find("all") != std::string::npos)
    {
        return CL_DEVICE_TYPE_ALL;
    }
    else if (type_name.find("no_setting") != std::string::npos)
    {
        return CL_DEVICE_TYPE_ALL;
    }
    else
    {
        return 0;
    }
}

std::string strType
(cl_device_type dtype)
{
    switch (dtype)
    {
    case CL_DEVICE_TYPE_GPU :
        return std::string("GPU");
    case CL_DEVICE_TYPE_CPU :
        return std::string("CPU");
    case CL_DEVICE_TYPE_ACCELERATOR :
        return std::string("ACCELERATOR");
    default :
        return std::string("Device type does not match known values");
    }
}

bool paramEnabled
(FILE* input, const char* param)
{
    std::string param_string = matchParam(input, param);
    return (param_string.find("NO_SETTING") == std::string::npos);
}

int preferredDevice
(FILE* input)
{
    std::string param_string = matchParam(input, "opencl_device");

    int preferred_device;

    if (param_string.size() == 0)
    {
        // not found in file
        preferred_device = -1;
    }
    else
    {
        std::stringstream converter(param_string);

        if (!(converter >> preferred_device))
        {
            preferred_device = -1;
        }
    }

    return preferred_device;
}

