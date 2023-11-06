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

// DEBUG OPTIONS
int DEBUG_FLOW = 0;
int DEBUG_RESULTS = 0;

// SIMULATION CONSTANTS
int N_ITERATIONS = 5;
int SWAP_ACTIVE = 1;
int N_THREADS = 1;

double INIT_TEMP = 0.1;
double END_TEMP = 0.2;
double TEMP_STEP = 0.1;
int TOTAL_REPLICAS = (int) (((END_TEMP-INIT_TEMP)/TEMP_STEP)+0.5); //* 0.5 is part of a truncation trick

// MODEL CONSTANTS
int N_ROW = 50;
int N_COL = 50;
double SPIN_PLUS_PERCENTAGE = 0.75;

#endif