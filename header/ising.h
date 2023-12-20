// Copyright Notice ===========================================================
//
// ising.h, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#ifndef _MODEL_H_
#define _MODEL_H_

#include "constants.h"
#include "rand.h"
#include "stack.h"

/**
 * @class SpinGlassIterationResult
 * @param _energy Energy of the model
 * @param _averasge_spin Average spin of the model
 * @brief
 * * Instances of this class saves the state generated 
 * * by one iteration of the MCMC
*/
class SpinGlassIterationResult {
    public:
        float _energy;
        int _average_spin;

        /**
         * @name SpinGlassIterationResult
         * @remark constructor
        */
        _HOST_ _DEVICE_ SpinGlassIterationResult();
};

_HOST_ _DEVICE_ void print_result(Stack<SpinGlassIterationResult, N_ITERATIONS>* result);

/**
 * @class SpinGlassTrial
 * @param _accepted If the trial is accepted
 * @param _row_index Row index selected for the trial
 * @param _col_index Col index selected for the trial
 * @brief
 * * Instances of this class represents a trial on the MCMC
 * * algorithm. Specifucally, a trial of a SpinGlass model
*/
class SpinGlassTrial {
    public:
        char _accepted;
        int _row_index, _col_index;
};

/**
 * @class SpinGlass
 * @param _rand_gen A random generator
 * @param _trial State of the trial
 * @param _results Stack of iteration results
 * @param _kernel_cross Convolution kernel in cross shape
 * @param _kernel_semicross Convolution kernel in semi-cross shape (Bottom and Right)
 * @param _sample State of the SpinGlass
 * @param _last_delta The last delta calculated with 'delta()'
 * @param _last_spin The value of the spin moved in the trial
 * @brief
 * * Instances of this class represent a replica in
 * * the MCMC algorithm. Specifically, this replica
 * * has the behaviour of a SpinGlass
*/
class SpinGlass {
    public:
        RandGen _rand_gen;
        SpinGlassTrial _trial;
        Stack<SpinGlassIterationResult, N_ITERATIONS>* _results;
        
        int _kernel_cross[9];
        int _kernel_semicross[9];

        char _sample[N_ROW*N_COL];

        double _last_delta;
        char _last_spin;

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
_DEVICE_ int is_index_in_matrix(char* mat, int n_row, int n_col, int row, int col);

/**
 * @name single_convolve
 * @param mat Matrix to use
 * @param n_row Number of rows of the matrix
 * @param n_col Number of collumns of the matrix
 * @param index Index of the matrix where do the convolution
 * @param kernel Kernel of the convolution
 * @param kernel_size Size of the kernel
 * @return The value of the convolution
 * @brief
 * * Applies the kernel in the specified position of the matrix
 * @note
 * The kernel size is assumed to be (kernel_size x kernel_size). 
 * Those values that, when applying the kernel, are outside the matrix, it will assumed the value 0 
*/
_DEVICE_ int apply_kernel(char* mat, int n_row, int n_col, int index, int* kernel, int kernel_size);


#define MODEL_NAME SpinGlass
#define MODEL_ITER SpinGlassIterationResult
#define MODEL_RESULTS Stack<SpinGlassIterationResult, N_ITERATIONS>

#endif