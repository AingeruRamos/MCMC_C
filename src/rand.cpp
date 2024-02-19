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

#include "../header/rand.h"

#include <stdlib.h>

_DEVICE_ void RandGen::set_state(unsigned int seed) {
    #ifdef __CUDACC__ //* IF IS COMPILED WITH NVCC
        curand_init(seed, 0, 0, &_rand_state);
    #else
        _rand_state = seed;
    #endif
}

_DEVICE_ double RandGen::rand_uniform() {
    #ifdef __CUDACC__ //* IF IS COMPILED WITH NVCC
        return (double)curand_uniform(&_rand_state);
    #else
        return (double)rand_r(&_rand_state)/(double)RAND_MAX;
    #endif
}

_DEVICE_ double RandGen::rand_uniform(double start, double end) {
    #ifdef __CUDACC__ //* IF IS COMPILED WITH NVCC
        double r = (double)curand_uniform(&_rand_state);
    #else
        double r = (double)rand_r(&_rand_state)/(double)RAND_MAX;
    #endif

    return (end-start)*r+start;
}

_DEVICE_ double RandGen::rand_normal() {
    double r1, r2;

    #ifdef __CUDACC__ //* IF IS COMPILED WITH NVCC
        r1 = (double)curand_uniform(&_rand_state);
        r2 = (double)curand_uniform(&_rand_state);
    #else
        r1 = (double)rand_r(&_rand_state)/(double)RAND_MAX;
        r2 = (double)rand_r(&_rand_state)/(double)RAND_MAX;
    #endif

    return sqrt(-2*log(r1)) * cos(2*M_PI*r2);
}


_DEVICE_ double RandGen::rand_normal(double mu, double std) {
    double r1, r2;

    #ifdef __CUDACC__ //* IF IS COMPILED WITH NVCC
        r1 = (double)curand_uniform(&_rand_state);
        r2 = (double)curand_uniform(&_rand_state);
    #else
        r1 = (double)rand_r(&_rand_state)/(double)RAND_MAX;
        r2 = (double)rand_r(&_rand_state)/(double)RAND_MAX;
    #endif

    double n_rand = sqrt(-2*log(r1)) * cos(2*M_PI*r2);
    return (n_rand*std)+mu;
}
