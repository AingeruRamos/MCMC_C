// Copyright Notice ===========================================================
//
// rand.cpp, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#include "./headers/rand.h"

#include <stdlib.h>

double rand_uniform() {
    return (double)rand()/(double)RAND_MAX;
}

double rand_uniform(double start, double end) {
    return (end-start)*rand_uniform()+start;
}