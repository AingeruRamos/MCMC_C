// Copyright Notice ===========================================================
//
// gauss.h, Copyright (c) 2023 Aingeru Ramos
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
#include "rand.h"
#include "stack.h"

/**
 * @class SpinGlass2DIterationResult
 * @param _energy Energy of the model at time t
 * @param _averasge_spin Average spin value of the model at time t
 * @brief
 * * Instances of this class saves the state generated 
 * * by one iteration (in time t) of the MCMC. Specifically, the state generateed
 * * by a replica with a behaviour of a SpinGlass2D model
*/
class Gauss1DIterationResult {
    public:
        float _energy;

        /**
         * @name SpinGlass2DIterationResult
         * @remark constructor
        */
        _HOST_ _DEVICE_ Gauss1DIterationResult();
};

/**
 * @name print_chain
 * @param chain A chain of a Replica with behaviour of SpinGlass2D
 * @param fp The file pointer where write the chain
 * @brief
 * * Writes the chain into a file
*/
_HOST_ void print_chain(Stack<Gauss1DIterationResult, N_ITERATIONS>* chain, FILE* fp);

/**
 * @class SpinGlass2DTrial
 * @param _accepted Flag of accepted trial
 * @param _row_index Row index selected for the trial
 * @param _col_index Collumn index selected for the trial
 * @brief
 * * Instances of this class represents a trial on the MCMC
 * * algorithm. Specifically, a trial of a SpinGlass2D model
*/
class Gauss1DTrial {
    public:
        char _accepted;
	float _x;
};

/**
 * @class SpinGlass2D
 * @param _rand_gen A random generator
 * @param _trial State of the trial
 * @param _chain Stack to store the generated chain
 * @param _kernel_cross Convolution kernel in cross shape
 * @param _kernel_semicross Convolution kernel in semi-cross shape (Bottom and Right)
 * @param _sample State of the SpinGlass model
 * @param _last_delta The last delta calculated with 'delta() method'
 * @param _last_spin The value of the spin moved in the trial
 * @brief
 * * Instances of this class represent a replica in
 * * the MCMC algorithm. Specifically, this replica
 * * has the behaviour of a SpinGlass2D
*/
class Gauss1D {
    public:
        RandGen _rand_gen;
        Gauss1DTrial _trial;
        Stack<Gauss1DIterationResult, N_ITERATIONS>* _chain;

	float _x;
        float _last_delta;

        /**
         * @name init
         * @brief
         * * Initializes the replica
        */
        _DEVICE_ void init();

        /**
         * @name trial
         * @brief
         * * Generates a trial of the replica
        */
        _DEVICE_ void trial();

        /**
         * @name delta
         * @return Value of the difference of the actual state of the replica and 
         * the replica result of applying the trial
         * @brief
         * * Calculates the effect of acceptance of the trial
        */
        _DEVICE_ double delta();

        /**
         * @name move
         * @brief
         * * Applies the trial to the replica
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

/**
 * @name is_index_in_matrix
 * @param mat Matrix to use
 * @param n_row Number of rows of the matrix
 * @param n_col Number of collumns of the matrix
 * @param row Row of the index to check
 * @param col Collumn of the index to check
 * @return A value indicating if the index is inside 
 * the matrix boundaries
 * @brief
 * * Return 1 if index is inside the matrix, else 0
*/
_DEVICE_ float dnorm(float x);

#define MODEL_NAME Gauss1D
#define MODEL_ITER Gauss1DIterationResult
#define MODEL_CHAIN Stack<Gauss1DIterationResult, N_ITERATIONS>

#endif
