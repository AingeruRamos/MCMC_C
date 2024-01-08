// Copyright Notice ===========================================================
//
// model_template.h, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>

#include "constants.h"
#include "stack.h"

/**
 * @class ReplicaIterationResult
 * @brief
 * * Instances of this class saves the state generated 
 * * by one iteration (in time t) of the MCMC.
*/
class ReplicaIterationResult {
    public:
        /**
         * @name ReplicaIterationResult
         * @remark constructor
        */
        _HOST_ _DEVICE_ ReplicaIterationResult();
};

/**
 * @name print_chain
 * @param chain A chain of a Replica
 * @param fp The file pointer where write the chain
 * @brief
 * * Writes the chain into a file
*/
_DEVICE_ void print_chain(Stack<ReplicaIterationResult, N_ITERATIONS>* result, FILE* fp);

/**
 * @class ReplicaTrial
 * @param _accepted Flag of accepted trial
 * @brief
 * * Instances of this class represents a trial on the MCMC
 * * algorithm.
*/
class ReplicaTrial {
    public:
        char _accepted;
};

/**
 * @class Replica
 * @param _rand_gen A random generator
 * @param _trial State of the trial
 * @param _results Stack to store the generated chain
 * @brief
 * * Instances of this class represent a replica in
 * * the MCMC algorithm.
*/
class Replica {
    public:
        RandGen _rand_gen;
        SpinGlassTrial _trial;
        Stack<ReplicaIterationResult, N_ITERATIONS> _results;

        /**
         * @name init
         * @brief
         * * Initializes the replica
        */
        _DEVICE_ void init();

        /**
         * @name trial
         * @return A trial
         * @brief
         * * Generates a trial of the replica
        */
        _DEVICE_ void* trial();

        /**
         * @name delta
         * @return Value of the difference of the actual replica and 
         * the replica result of applying the trial
         * @brief
         * * Calculates the effect of acceptance of the trial
        */
        _DEVICE_ double delta();

        /**
         * @name move
         * @brief
         * * Apply the trial to the replica
        */
        _DEVICE_ void move();

        /**
         * @name save
         * @brief
         * * Save the state of the replica
        */
        _DEVICE_ void save();
};

//-----------------------------------------------------------------------------|
//                             AUXILIARY FUNCTIONS                             |
//-----------------------------------------------------------------------------|

#define MODEL_NAME Replica
#define MODEL_ITER ReplicaIterationResult
#define MODEL_RESULTS Stack<ReplicaIterationResult, N_ITERATIONS>

#endif