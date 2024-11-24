/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2024-10-14T10:53:55+0300
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x9605a18cbddab48de98fbad313e8a130"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2024-10-14T10:53:55+0300"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2400, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_Transpose_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2400, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_initial_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  _relu_initial_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  _conv1B_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  _relu1B_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  _single_conv_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  _single_relu_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  _conv3B_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  _relu3B_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_0_GlobalAveragePool_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_1_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_3_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_4_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  _Mul_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  _Add_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  _layer3_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 23040, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  _relu3_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 23040, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_0_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_2_Relu_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_3_Tanh_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1920, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_4_Conv_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 23040, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_5_Sigmoid_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 23040, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  _Mul_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 23040, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceSum_output_0_reduce_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  _Pow_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  _Pow_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 23040, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  _Mul_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 23040, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  _ReduceSum_1_output_0_reduce_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  _Sub_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  _Pow_2_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  _Add_1_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  _Sqrt_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  _Concat_output_0_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1536, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  _fc6_Gemm_output_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 152, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_initial_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 25600, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_initial_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  _conv1B_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  _conv1B_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  _single_conv_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12288, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  _single_conv_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  _conv3B_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 4096, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  _conv3B_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_1_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_1_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 8192, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  _layer3_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  _layer3_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_0_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_0_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_4_Conv_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49152, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_4_Conv_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_2_output_0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_1_output_0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  _Constant_output_0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  _Add_1_output_0_scale_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  _Add_1_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  _fc6_Gemm_output_0_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 233472, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  _fc6_Gemm_output_0_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 152, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  _conv1_initial_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 400, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  _conv1B_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  _conv3B_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_1_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  _se_se_3_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 128, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  _layer3_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_0_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 768, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  _attention_attention_4_Conv_output_0_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 64, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  _Concat_output_0_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 1536, 1, 1), AI_STRIDE_INIT(4, 4, 4, 6144, 6144),
  1, &_Concat_output_0_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  _fc6_Gemm_output_0_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 152, 1, 1), AI_STRIDE_INIT(4, 4, 4, 608, 608),
  1, &_fc6_Gemm_output_0_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_4_Conv_output_0_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_attention_attention_4_Conv_output_0_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_initial_Conv_output_0_weights, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 80, 1, 5, 64), AI_STRIDE_INIT(4, 4, 320, 20480, 20480),
  1, &_conv1_initial_Conv_output_0_weights_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_initial_Conv_output_0_bias, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_initial_Conv_output_0_bias_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  _conv1B_Conv_output_0_weights, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_conv1B_Conv_output_0_weights_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  _conv1B_Conv_output_0_bias, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1B_Conv_output_0_bias_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  _single_conv_Conv_output_0_weights, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 3, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_single_conv_Conv_output_0_weights_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  _single_conv_Conv_output_0_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_single_conv_Conv_output_0_bias_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  _conv3B_Conv_output_0_weights, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 4, 256, 16384, 16384),
  1, &_conv3B_Conv_output_0_weights_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  _conv3B_Conv_output_0_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv3B_Conv_output_0_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_1_Conv_output_0_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 128), AI_STRIDE_INIT(4, 4, 256, 32768, 32768),
  1, &_se_se_1_Conv_output_0_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_1_Conv_output_0_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_se_se_1_Conv_output_0_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_3_Conv_output_0_weights, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 128, 1, 1, 64), AI_STRIDE_INIT(4, 4, 512, 32768, 32768),
  1, &_se_se_3_Conv_output_0_weights_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_3_Conv_output_0_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_se_se_3_Conv_output_0_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  _layer3_Conv_output_0_weights, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 768), AI_STRIDE_INIT(4, 4, 256, 196608, 196608),
  1, &_layer3_Conv_output_0_weights_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  _layer3_Conv_output_0_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_layer3_Conv_output_0_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_0_Conv_output_0_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 768, 1, 1, 64), AI_STRIDE_INIT(4, 4, 3072, 196608, 196608),
  1, &_attention_attention_0_Conv_output_0_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_0_Conv_output_0_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_attention_attention_0_Conv_output_0_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_4_Conv_output_0_weights, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 64, 1, 1, 768), AI_STRIDE_INIT(4, 4, 256, 196608, 196608),
  1, &_attention_attention_4_Conv_output_0_weights_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_4_Conv_output_0_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_attention_attention_4_Conv_output_0_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_2_output_0, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_Constant_2_output_0_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 30, 1, 80), AI_STRIDE_INIT(4, 4, 4, 120, 120),
  1, &input_output_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_1_output_0, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_Constant_1_output_0_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  _Constant_output_0, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &_Constant_output_0_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  input_Transpose_output, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 80, 1, 30), AI_STRIDE_INIT(4, 4, 4, 320, 320),
  1, &input_Transpose_output_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_initial_Conv_output_0_output, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1_initial_Conv_output_0_output_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  _Add_1_output_0_scale, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_Add_1_output_0_scale_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  _relu_initial_Relu_output_0_output, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_relu_initial_Relu_output_0_output_array, NULL)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  _Add_1_output_0_bias, AI_STATIC,
  29, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_Add_1_output_0_bias_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  _conv1B_Conv_output_0_output, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1B_Conv_output_0_output_array, NULL)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  _relu1B_Relu_output_0_output, AI_STATIC,
  31, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_relu1B_Relu_output_0_output_array, NULL)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  _fc6_Gemm_output_0_weights, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1536, 152, 1, 1), AI_STRIDE_INIT(4, 4, 6144, 933888, 933888),
  1, &_fc6_Gemm_output_0_weights_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  _single_conv_Conv_output_0_output, AI_STATIC,
  33, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_single_conv_Conv_output_0_output_array, NULL)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  _fc6_Gemm_output_0_bias, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 152, 1, 1), AI_STRIDE_INIT(4, 4, 4, 608, 608),
  1, &_fc6_Gemm_output_0_bias_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  _single_relu_Relu_output_0_output, AI_STATIC,
  35, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_single_relu_Relu_output_0_output_array, NULL)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  _conv3B_Conv_output_0_output, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv3B_Conv_output_0_output_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  _relu3B_Relu_output_0_output, AI_STATIC,
  37, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_relu3B_Relu_output_0_output_array, NULL)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_0_GlobalAveragePool_output_0_output, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_se_se_0_GlobalAveragePool_output_0_output_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  _conv1_initial_Conv_output_0_scratch0, AI_STATIC,
  39, 0x0,
  AI_SHAPE_INIT(4, 1, 80, 1, 5), AI_STRIDE_INIT(4, 4, 4, 320, 320),
  1, &_conv1_initial_Conv_output_0_scratch0_array, NULL)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_1_Conv_output_0_output, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_se_se_1_Conv_output_0_output_array, NULL)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_2_Relu_output_0_output, AI_STATIC,
  41, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_se_se_2_Relu_output_0_output_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_3_Conv_output_0_output, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_se_se_3_Conv_output_0_output_array, NULL)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  _conv1B_Conv_output_0_scratch0, AI_STATIC,
  43, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv1B_Conv_output_0_scratch0_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_4_Sigmoid_output_0_output, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_se_se_4_Sigmoid_output_0_output_array, NULL)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  _Mul_output_0_output, AI_STATIC,
  45, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Mul_output_0_output_array, NULL)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  _Add_output_0_output, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_Add_output_0_output_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  _layer3_Conv_output_0_output, AI_STATIC,
  47, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 30), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_layer3_Conv_output_0_output_array, NULL)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  _conv3B_Conv_output_0_scratch0, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_conv3B_Conv_output_0_scratch0_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  _relu3_Relu_output_0_output, AI_STATIC,
  49, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 30), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_relu3_Relu_output_0_output_array, NULL)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_0_Conv_output_0_output, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_attention_attention_0_Conv_output_0_output_array, NULL)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_2_Relu_output_0_output, AI_STATIC,
  51, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_attention_attention_2_Relu_output_0_output_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_3_Tanh_output_0_output, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 30), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_attention_attention_3_Tanh_output_0_output_array, NULL)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_1_Conv_output_0_scratch0, AI_STATIC,
  53, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_se_se_1_Conv_output_0_scratch0_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_4_Conv_output_0_output, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 30), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_attention_attention_4_Conv_output_0_output_array, NULL)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_5_Sigmoid_output_0_output, AI_STATIC,
  55, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 30), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_attention_attention_5_Sigmoid_output_0_output_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  _Mul_1_output_0_output, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 30), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_Mul_1_output_0_output_array, NULL)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  _se_se_3_Conv_output_0_scratch0, AI_STATIC,
  57, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &_se_se_3_Conv_output_0_scratch0_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceSum_output_0_reduce_output, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_ReduceSum_output_0_reduce_output_array, NULL)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  _Pow_1_output_0_output, AI_STATIC,
  59, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_Pow_1_output_0_output_array, NULL)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  _Pow_output_0_output, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 30), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_Pow_output_0_output_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  _Mul_2_output_0_output, AI_STATIC,
  61, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 30), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_Mul_2_output_0_output_array, NULL)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  _layer3_Conv_output_0_scratch0, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &_layer3_Conv_output_0_scratch0_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  _ReduceSum_1_output_0_reduce_output, AI_STATIC,
  63, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_ReduceSum_1_output_0_reduce_output_array, NULL)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  _Sub_output_0_output, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_Sub_output_0_output_array, NULL)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  _Pow_2_output_0_output, AI_STATIC,
  65, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_Pow_2_output_0_output_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  _attention_attention_0_Conv_output_0_scratch0, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_attention_attention_0_Conv_output_0_scratch0_array, NULL)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  _Add_1_output_0_output, AI_STATIC,
  67, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_Add_1_output_0_output_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  _Sqrt_output_0_output, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 768, 1, 1), AI_STRIDE_INIT(4, 4, 4, 3072, 3072),
  1, &_Sqrt_output_0_output_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  _fc6_Gemm_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_fc6_Gemm_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_fc6_Gemm_output_0_weights, &_fc6_Gemm_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _fc6_Gemm_output_0_layer, 41,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &_fc6_Gemm_output_0_chain,
  NULL, &_fc6_Gemm_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Concat_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_ReduceSum_output_0_reduce_output, &_Sqrt_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Concat_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Concat_output_0_layer, 38,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &_Concat_output_0_chain,
  NULL, &_fc6_Gemm_output_0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Sqrt_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Sqrt_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Sqrt_output_0_layer, 37,
  NL_TYPE, 0x0, NULL,
  nl, forward_sqrt,
  &_Sqrt_output_0_chain,
  NULL, &_Concat_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Pow_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Add_1_output_0_scale, &_Add_1_output_0_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_1_output_0_layer, 36,
  BN_TYPE, 0x0, NULL,
  bn, forward_bn,
  &_Add_1_output_0_chain,
  NULL, &_Sqrt_output_0_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Pow_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Sub_output_0_output, &_Constant_2_output_0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Pow_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Pow_2_output_0_layer, 34,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Pow_2_output_0_chain,
  NULL, &_Add_1_output_0_layer, AI_STATIC, 
  .operation = ai_pow, 
  .buffer_operation = ai_pow_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Sub_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_ReduceSum_1_output_0_reduce_output, &_Pow_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Sub_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Sub_output_0_layer, 32,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Sub_output_0_chain,
  NULL, &_Pow_2_output_0_layer, AI_STATIC, 
  .operation = ai_sub_f32, 
  .buffer_operation = ai_sub_buffer_f32, 
)


AI_STATIC_CONST ai_float _ReduceSum_1_output_0_reduce_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    _ReduceSum_1_output_0_reduce_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    _ReduceSum_1_output_0_reduce_neutral_value_data, _ReduceSum_1_output_0_reduce_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _ReduceSum_1_output_0_reduce_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceSum_1_output_0_reduce_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _ReduceSum_1_output_0_reduce_layer, 29,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &_ReduceSum_1_output_0_reduce_chain,
  NULL, &_Sub_output_0_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &_ReduceSum_1_output_0_reduce_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Mul_2_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Pow_output_0_output, &_attention_attention_5_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Mul_2_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Mul_2_output_0_layer, 28,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Mul_2_output_0_chain,
  NULL, &_ReduceSum_1_output_0_reduce_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Pow_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_relu3_Relu_output_0_output, &_Constant_output_0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Pow_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Pow_output_0_layer, 27,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Pow_output_0_chain,
  NULL, &_Mul_2_output_0_layer, AI_STATIC, 
  .operation = ai_pow, 
  .buffer_operation = ai_pow_buffer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Pow_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_ReduceSum_output_0_reduce_output, &_Constant_1_output_0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Pow_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Pow_1_output_0_layer, 31,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Pow_1_output_0_chain,
  NULL, &_Pow_output_0_layer, AI_STATIC, 
  .operation = ai_pow, 
  .buffer_operation = ai_pow_buffer, 
)


AI_STATIC_CONST ai_float _ReduceSum_output_0_reduce_neutral_value_data[] = { 0.0f };
AI_ARRAY_OBJ_DECLARE(
    _ReduceSum_output_0_reduce_neutral_value, AI_ARRAY_FORMAT_FLOAT,
    _ReduceSum_output_0_reduce_neutral_value_data, _ReduceSum_output_0_reduce_neutral_value_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  _ReduceSum_output_0_reduce_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Mul_1_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_ReduceSum_output_0_reduce_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _ReduceSum_output_0_reduce_layer, 25,
  REDUCE_TYPE, 0x0, NULL,
  reduce, forward_reduce,
  &_ReduceSum_output_0_reduce_chain,
  NULL, &_Pow_1_output_0_layer, AI_STATIC, 
  .operation = ai_sum, 
  .neutral_value = &_ReduceSum_output_0_reduce_neutral_value, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Mul_1_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_relu3_Relu_output_0_output, &_attention_attention_5_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Mul_1_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Mul_1_output_0_layer, 24,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Mul_1_output_0_chain,
  NULL, &_ReduceSum_output_0_reduce_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _attention_attention_5_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_attention_attention_4_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_attention_attention_5_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _attention_attention_5_Sigmoid_output_0_layer, 23,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_attention_attention_5_Sigmoid_output_0_chain,
  NULL, &_Mul_1_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _attention_attention_4_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_attention_attention_3_Tanh_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_attention_attention_4_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_attention_attention_4_Conv_output_0_weights, &_attention_attention_4_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_attention_attention_4_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _attention_attention_4_Conv_output_0_layer, 22,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_attention_attention_4_Conv_output_0_chain,
  NULL, &_attention_attention_5_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _attention_attention_3_Tanh_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_attention_attention_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_attention_attention_3_Tanh_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _attention_attention_3_Tanh_output_0_layer, 21,
  NL_TYPE, 0x0, NULL,
  nl, forward_tanh,
  &_attention_attention_3_Tanh_output_0_chain,
  NULL, &_attention_attention_4_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _attention_attention_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_attention_attention_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_attention_attention_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _attention_attention_2_Relu_output_0_layer, 20,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_attention_attention_2_Relu_output_0_chain,
  NULL, &_attention_attention_3_Tanh_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _attention_attention_0_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_relu3_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_attention_attention_0_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_attention_attention_0_Conv_output_0_weights, &_attention_attention_0_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_attention_attention_0_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _attention_attention_0_Conv_output_0_layer, 19,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_attention_attention_0_Conv_output_0_chain,
  NULL, &_attention_attention_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _relu3_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_layer3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_relu3_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _relu3_Relu_output_0_layer, 18,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_relu3_Relu_output_0_chain,
  NULL, &_attention_attention_0_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _layer3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_layer3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_layer3_Conv_output_0_weights, &_layer3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_layer3_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _layer3_Conv_output_0_layer, 17,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_layer3_Conv_output_0_chain,
  NULL, &_relu3_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Add_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_Mul_output_0_output, &_relu_initial_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Add_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Add_output_0_layer, 16,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Add_output_0_chain,
  NULL, &_layer3_Conv_output_0_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _Mul_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_relu3B_Relu_output_0_output, &_se_se_4_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_Mul_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _Mul_output_0_layer, 15,
  ELTWISE_TYPE, 0x0, NULL,
  eltwise, forward_eltwise,
  &_Mul_output_0_chain,
  NULL, &_Add_output_0_layer, AI_STATIC, 
  .operation = ai_mul_f32, 
  .buffer_operation = ai_mul_buffer_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_se_4_Sigmoid_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_se_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_se_4_Sigmoid_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _se_se_4_Sigmoid_output_0_layer, 14,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &_se_se_4_Sigmoid_output_0_chain,
  NULL, &_Mul_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_se_3_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_se_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_se_3_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_se_se_3_Conv_output_0_weights, &_se_se_3_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_se_se_3_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _se_se_3_Conv_output_0_layer, 13,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_se_se_3_Conv_output_0_chain,
  NULL, &_se_se_4_Sigmoid_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_se_2_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_se_1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_se_2_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _se_se_2_Relu_output_0_layer, 12,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_se_se_2_Relu_output_0_chain,
  NULL, &_se_se_3_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_se_1_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_se_0_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_se_1_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_se_se_1_Conv_output_0_weights, &_se_se_1_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_se_se_1_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _se_se_1_Conv_output_0_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_se_se_1_Conv_output_0_chain,
  NULL, &_se_se_2_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _se_se_0_GlobalAveragePool_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_relu3B_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_se_se_0_GlobalAveragePool_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _se_se_0_GlobalAveragePool_output_0_layer, 10,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &_se_se_0_GlobalAveragePool_output_0_chain,
  NULL, &_se_se_1_Conv_output_0_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 30), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 30), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _relu3B_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv3B_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_relu3B_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _relu3B_Relu_output_0_layer, 9,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_relu3B_Relu_output_0_chain,
  NULL, &_se_se_0_GlobalAveragePool_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv3B_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_single_relu_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv3B_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv3B_Conv_output_0_weights, &_conv3B_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_conv3B_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _conv3B_Conv_output_0_layer, 8,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_conv3B_Conv_output_0_chain,
  NULL, &_relu3B_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _single_relu_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_single_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_single_relu_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _single_relu_Relu_output_0_layer, 7,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_single_relu_Relu_output_0_chain,
  NULL, &_conv3B_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _single_conv_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_relu1B_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_single_conv_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_single_conv_Conv_output_0_weights, &_single_conv_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _single_conv_Conv_output_0_layer, 6,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32_group,
  &_single_conv_Conv_output_0_chain,
  NULL, &_single_relu_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 2), 
  .filter_pad = AI_SHAPE_INIT(4, 2, 0, 2, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _relu1B_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1B_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_relu1B_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _relu1B_Relu_output_0_layer, 5,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_relu1B_Relu_output_0_chain,
  NULL, &_single_conv_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv1B_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_relu_initial_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1B_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv1B_Conv_output_0_weights, &_conv1B_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_conv1B_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _conv1B_Conv_output_0_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_conv1B_Conv_output_0_chain,
  NULL, &_relu1B_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _relu_initial_Relu_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_initial_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_relu_initial_Relu_output_0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  _relu_initial_Relu_output_0_layer, 3,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &_relu_initial_Relu_output_0_chain,
  NULL, &_conv1B_Conv_output_0_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  _conv1_initial_Conv_output_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &_conv1_initial_Conv_output_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &_conv1_initial_Conv_output_0_weights, &_conv1_initial_Conv_output_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &_conv1_initial_Conv_output_0_scratch0, NULL)
)

AI_LAYER_OBJ_DECLARE(
  _conv1_initial_Conv_output_0_layer, 2,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_if32of32wf32,
  &_conv1_initial_Conv_output_0_chain,
  NULL, &_relu_initial_Relu_output_0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 2, 0, 2, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_Transpose_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_Transpose_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_Transpose_layer, 2,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &input_Transpose_chain,
  NULL, &_conv1_initial_Conv_output_0_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_HEIGHT, AI_SHAPE_WIDTH, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1788524, 1, 1),
    1788524, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 279552, 1, 1),
    279552, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &_fc6_Gemm_output_0_output),
  &input_Transpose_layer, 0x70c6cdf3, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 1788524, 1, 1),
      1788524, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 279552, 1, 1),
      279552, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &_fc6_Gemm_output_0_output),
  &input_Transpose_layer, 0x70c6cdf3, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_output_array.data = AI_PTR(g_network_activations_map[0] + 165568);
    input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 165568);
    input_Transpose_output_array.data = AI_PTR(g_network_activations_map[0] + 175168);
    input_Transpose_output_array.data_start = AI_PTR(g_network_activations_map[0] + 175168);
    _conv1_initial_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 165568);
    _conv1_initial_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 165568);
    _conv1_initial_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 167168);
    _conv1_initial_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 167168);
    _relu_initial_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 174848);
    _relu_initial_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 174848);
    _conv1B_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 165568);
    _conv1B_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 165568);
    _conv1B_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 167168);
    _conv1B_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 167168);
    _relu1B_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 167168);
    _relu1B_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 167168);
    _single_conv_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 166144);
    _single_conv_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 166144);
    _single_relu_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 166144);
    _single_relu_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 166144);
    _conv3B_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 165568);
    _conv3B_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 165568);
    _conv3B_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 165888);
    _conv3B_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 165888);
    _relu3B_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 165888);
    _relu3B_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 165888);
    _se_se_0_GlobalAveragePool_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 165568);
    _se_se_0_GlobalAveragePool_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 165568);
    _se_se_1_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 173568);
    _se_se_1_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 173568);
    _se_se_1_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 174336);
    _se_se_1_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 174336);
    _se_se_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 184256);
    _se_se_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 184256);
    _se_se_3_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 183744);
    _se_se_3_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 183744);
    _se_se_3_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 183488);
    _se_se_3_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 183488);
    _se_se_4_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 174592);
    _se_se_4_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 174592);
    _Mul_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 165888);
    _Mul_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 165888);
    _Add_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 182528);
    _Add_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 182528);
    _layer3_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 190208);
    _layer3_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 190208);
    _layer3_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 92160);
    _layer3_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 92160);
    _relu3_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 92160);
    _relu3_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 92160);
    _attention_attention_0_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 89088);
    _attention_attention_0_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 89088);
    _attention_attention_0_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 81408);
    _attention_attention_0_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 81408);
    _attention_attention_2_Relu_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 73728);
    _attention_attention_2_Relu_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 73728);
    _attention_attention_3_Tanh_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 66048);
    _attention_attention_3_Tanh_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 66048);
    _attention_attention_4_Conv_output_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 65792);
    _attention_attention_4_Conv_output_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 65792);
    _attention_attention_4_Conv_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 184320);
    _attention_attention_4_Conv_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 184320);
    _attention_attention_5_Sigmoid_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 184320);
    _attention_attention_5_Sigmoid_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 184320);
    _Mul_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _Mul_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _ReduceSum_output_0_reduce_output_array.data = AI_PTR(g_network_activations_map[0] + 276480);
    _ReduceSum_output_0_reduce_output_array.data_start = AI_PTR(g_network_activations_map[0] + 276480);
    _Pow_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _Pow_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _Pow_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 92160);
    _Pow_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 92160);
    _Mul_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 92160);
    _Mul_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 92160);
    _ReduceSum_1_output_0_reduce_output_array.data = AI_PTR(g_network_activations_map[0] + 3072);
    _ReduceSum_1_output_0_reduce_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072);
    _Sub_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 6144);
    _Sub_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6144);
    _Pow_2_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _Pow_2_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _Add_1_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3072);
    _Add_1_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072);
    _Sqrt_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _Sqrt_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    _Concat_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 3072);
    _Concat_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 3072);
    _fc6_Gemm_output_0_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    _fc6_Gemm_output_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    _conv1_initial_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv1_initial_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    _conv1_initial_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    _conv1_initial_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv1_initial_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 102400);
    _conv1_initial_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 102400);
    _conv1B_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv1B_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 102656);
    _conv1B_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 102656);
    _conv1B_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv1B_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 119040);
    _conv1B_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 119040);
    _single_conv_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _single_conv_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 119296);
    _single_conv_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 119296);
    _single_conv_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _single_conv_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 168448);
    _single_conv_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 168448);
    _conv3B_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _conv3B_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 168704);
    _conv3B_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 168704);
    _conv3B_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _conv3B_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 185088);
    _conv3B_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 185088);
    _se_se_1_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _se_se_1_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 185344);
    _se_se_1_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 185344);
    _se_se_1_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _se_se_1_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 218112);
    _se_se_1_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 218112);
    _se_se_3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _se_se_3_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 218624);
    _se_se_3_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 218624);
    _se_se_3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _se_se_3_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 251392);
    _se_se_3_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 251392);
    _layer3_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _layer3_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 251648);
    _layer3_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 251648);
    _layer3_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _layer3_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 448256);
    _layer3_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 448256);
    _attention_attention_0_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _attention_attention_0_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 451328);
    _attention_attention_0_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 451328);
    _attention_attention_0_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _attention_attention_0_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 647936);
    _attention_attention_0_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 647936);
    _attention_attention_4_Conv_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _attention_attention_4_Conv_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 648192);
    _attention_attention_4_Conv_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 648192);
    _attention_attention_4_Conv_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _attention_attention_4_Conv_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 844800);
    _attention_attention_4_Conv_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 844800);
    _Constant_2_output_0_array.format |= AI_FMT_FLAG_CONST;
    _Constant_2_output_0_array.data = AI_PTR(g_network_weights_map[0] + 847872);
    _Constant_2_output_0_array.data_start = AI_PTR(g_network_weights_map[0] + 847872);
    _Constant_1_output_0_array.format |= AI_FMT_FLAG_CONST;
    _Constant_1_output_0_array.data = AI_PTR(g_network_weights_map[0] + 847876);
    _Constant_1_output_0_array.data_start = AI_PTR(g_network_weights_map[0] + 847876);
    _Constant_output_0_array.format |= AI_FMT_FLAG_CONST;
    _Constant_output_0_array.data = AI_PTR(g_network_weights_map[0] + 847880);
    _Constant_output_0_array.data_start = AI_PTR(g_network_weights_map[0] + 847880);
    _Add_1_output_0_scale_array.format |= AI_FMT_FLAG_CONST;
    _Add_1_output_0_scale_array.data = AI_PTR(g_network_weights_map[0] + 847884);
    _Add_1_output_0_scale_array.data_start = AI_PTR(g_network_weights_map[0] + 847884);
    _Add_1_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _Add_1_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 850956);
    _Add_1_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 850956);
    _fc6_Gemm_output_0_weights_array.format |= AI_FMT_FLAG_CONST;
    _fc6_Gemm_output_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 854028);
    _fc6_Gemm_output_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 854028);
    _fc6_Gemm_output_0_bias_array.format |= AI_FMT_FLAG_CONST;
    _fc6_Gemm_output_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 1787916);
    _fc6_Gemm_output_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1787916);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 6697544,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x70c6cdf3,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 6697544,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x70c6cdf3,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

