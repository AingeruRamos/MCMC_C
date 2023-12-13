// Copyright Notice ===========================================================
//
// constants.h, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#ifndef _CONST_H_
#define _CONST_H_

// CUDA DECORATOR
#ifdef __CUDACC__ //* IF IS COMPILED WITH NVCC
#   define _DEVICE_ __device__
#   define _HOST_ __host__
#else
#   define _DEVICE_
#   define _HOST_
#endif

// SIMULATION CONSTANTS
#ifndef N_ITERATIONS
    #define N_ITERATIONS 50
#endif

#ifndef SWAP_ACTIVE
    #define SWAP_ACTIVE 0
#endif

#ifndef INIT_TEMP
    #define INIT_TEMP 0.1
#endif

#ifndef END_TEMP
    #define END_TEMP 4.1
#endif

#ifndef TEMP_STEP
    #define TEMP_STEP 1.5
#endif

#define AUX ((END_TEMP-INIT_TEMP)/TEMP_STEP)
#define TOTAL_REPLICAS ((int) AUX + ( ((double)(int) AUX < AUX) ? 1 : 0)) //* Ceil trick

// MODEL CONSTANTS
#ifndef N_ROW
    #define N_ROW 50
#endif

#ifndef N_COL
    #define N_COL 50
#endif

#ifndef SPIN_PLUS_PERCENTAGE
    #define SPIN_PLUS_PERCENTAGE 0.75
#endif

#endif