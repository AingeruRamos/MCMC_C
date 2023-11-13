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

#ifndef _ISING_H_
#define _ISING_H_

#include "constants.h"
#include "stack.h"

class SpinGlassIterationResult {
    public:
        double _energy;
        int _average_spin;

        _CUDA_DECOR_ SpinGlassIterationResult();
        _CUDA_DECOR_ SpinGlassIterationResult(double energy, double average_spin);

        _CUDA_DECOR_ SpinGlassIterationResult* copy();
};

class SpinGlass {
    public:
        int _kernel_cross[9];
        int _kernel_semicross[9];

        char _sample[N_ROW*N_COL];

        double _last_delta;
        char _last_spin;
        
        Stack<SpinGlassIterationResult*, N_ITERATIONS> _results;

        _CUDA_DECOR_ void init();
        _CUDA_DECOR_ void* trial();
        _CUDA_DECOR_ double eval();
        _CUDA_DECOR_ double delta(void* trial);
        _CUDA_DECOR_ void move(void* trial);
        _CUDA_DECOR_ void save(void* trial);
};

/**
 * @name is_index_in_matrix
 * @param mat Matrix to use
 * @param n_row Number of rows of the matrix
 * @param n_col Number of collumns of the matrix
 * @param row Row of the index to check
 * @param col Collumn of the index to check
 * @brief
 * * Return 1 if index is inside the matrix, else 0
*/
_CUDA_DECOR_ int is_index_in_matrix(char* mat, int n_row, int n_col, int row, int col);

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
_CUDA_DECOR_ int apply_kernel(char* mat, int n_row, int n_col, int index, int* kernel, int kernel_size);

#endif