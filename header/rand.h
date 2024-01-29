// Copyright Notice ===========================================================
//
// rand.h, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#ifndef RAND_H
#define RAND_H

#include "constants.h"

#ifdef __CUDACC__ //* IF IS COMPILED WITH NVCC
#   include <curand.h>
#   include <curand_kernel.h>
#   define rand_gen_t curandState
#else
#   define rand_gen_t unsigned int
#endif

#include <math.h>

/**
 * @class RandGen
 * @param _rand_state The actual state of the random generator
 * @brief
 * * Instances of this class are used to generate random numbers.
 * * The class is designed to be used in OpenMP and CUDA version.
*/
class RandGen {
    private:
        rand_gen_t _rand_state;
    
    public:

        /**
         * @name set_state
         * @param _seed Initial seed of the generator
         * @brief
         * * Sets the initial state of the generator
        */
        _DEVICE_ void set_state(unsigned int _seed);

        /**
         * 
         * @name rand_uniform
         * @return A random value
         * @brief
         * * Calculates a random in the raqnge of [0, 1]
        */
        _DEVICE_ double rand_uniform();

        /**
         * @name rand_uniform
         * @overload
         * @return A random value
         * @brief
         * * Calculates a random in the range of [start, end]
        */
        _DEVICE_ double rand_uniform(double start, double end);

        /**
         * @name rand_normal
         * @overload
         * @return A random value
         * @brief
         * * Calculates a random in the range of [start, end]
        */
        _DEVICE_ double rand_normal();

};

#endif