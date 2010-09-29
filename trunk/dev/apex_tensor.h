#ifndef _APEX_TENSOR_H_
#define _APEX_TENSOR_H_

/*!
 * \file apex_tensor.h
 * \brief header of all library
 * \author Tianqi Chen: tqchen@apex.sjtu.edu.cn
 */
#include "apex_tensor_config.h"

namespace apex_tensor{
    class CTensor1D;
    class CTensor2D;
    class CTensor3D;
    class CTensor4D;
    class GTensor1D;
    class GTensor2D;
    class GTensor3D;
    class GTensor4D;       
};

#include "apex_tensor_cpu.h"
#include "apex_tensor_gpu.h"

#define _APEX_TENSOR_CPU_DECL_
#include "apex_tensor_func_decl.h"
#undef  _APEX_TENSOR_CPU_DECL_

#define _APEX_TENSOR_GPU_DECL_
#include "apex_tensor_func_decl.h"
#undef  _APEX_TENSOR_GPU_DECL_

#endif
