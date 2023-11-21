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
#include <time.h>

RandGen::RandGen() {
    _rand_state = time(NULL);
}

void RandGen::set_state(unsigned int seed) {
    _rand_state = seed;
}

double RandGen::rand_uniform() {
    return (double)rand_r(&_rand_state)/(double)RAND_MAX;
}

double RandGen::rand_uniform(double start, double end) {
    double r = (double)rand_r(&_rand_state)/(double)RAND_MAX;
    return (end-start)*r+start;
}