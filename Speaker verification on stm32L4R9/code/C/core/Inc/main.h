/**
  ******************************************************************************
  * @file    DFSDM/DFSDM_AudioRecord/Inc/main.h
  * @author  MCD Application Team
  * @brief   Header for main.c module
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2017 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

/* Includes ------------------------------------------------------------------*/
#include "stm32l4xx_hal.h"
#include "stm32l4r9i_discovery.h"
#include "audio.h"
#include "cs42l51.h"
/* Exported types ------------------------------------------------------------*/
/* Exported constants --------------------------------------------------------*/
/* Exported macro ------------------------------------------------------------*/
/* Exported functions ------------------------------------------------------- */
void Error_Handler(void);
void Preprocessing_Init(void);
//void pre_emphasis(const float* input, float* output, int len);
void reflect_pad(int32_t *original, int16_t original_size, int16_t pad_size, int32_t *padded);
void AudioPreprocessing_Run( int32_t *pInSignal, float *pOutMfcc, int signal_len);
void arm_cosine_distance_f32( float *pA, float *pB, uint32_t blockSize, float *pOutD);
#endif /* __MAIN_H */

