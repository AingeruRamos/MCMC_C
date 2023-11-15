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
#   define _CUDA_DECOR_ __host__ __device__
#else
#   define _CUDA_DECOR_
#endif

// SIMULATION CONSTANTS
#define N_ITERATIONS 250000
#define SWAP_ACTIVE 0

#define INIT_TEMP 0.1
#define END_TEMP 3.1
#define TEMP_STEP 0.5
#define TOTAL_REPLICAS (int) (((END_TEMP-INIT_TEMP)/TEMP_STEP)+0.5) //* 0.5 is part of a truncation trick

// MODEL CONSTANTS
#define N_ROW 200
#define N_COL 200
#define SPIN_PLUS_PERCENTAGE 0.75

#endif