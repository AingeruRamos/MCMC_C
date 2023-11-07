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

// SIMULATION CONSTANTS
#define N_ITERATIONS 5
#define SWAP_ACTIVE 1

#define INIT_TEMP 0.1
#define END_TEMP 0.2
#define TEMP_STEP 0.1
const int TOTAL_REPLICAS = (int) (((END_TEMP-INIT_TEMP)/TEMP_STEP)+0.5); //* 0.5 is part of a truncation trick

// MODEL CONSTANTS
#define N_ROW 50
#define N_COL 50
#define SPIN_PLUS_PERCENTAGE 0.75

#endif