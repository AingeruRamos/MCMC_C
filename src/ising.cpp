// Copyright Notice ===========================================================
//
// ising.cpp, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#include "../header/ising.h"

#include <stdio.h>
#include <stdlib.h>

#include "../header/constants.h"
#include "../header/rand.h"

_HOST_ _DEVICE_ SpinGlass2DIterationResult::SpinGlass2DIterationResult() {
    _energy = 0;
    _average_spin = 0;
}

_HOST_ void print_chain(Stack<SpinGlass2DIterationResult, N_ITERATIONS>* chain, FILE* fp) {
    SpinGlass2DIterationResult* sp_it;

    for(int i=0; i<N_ITERATIONS; i++) {
        sp_it = (SpinGlass2DIterationResult*) chain->get(i);
        fwrite(&sp_it->_energy, sizeof(float), 1, fp);
    }
    for(int i=0; i<N_ITERATIONS; i++) {
        sp_it = (SpinGlass2DIterationResult*) chain->get(i);
        fwrite(&sp_it->_average_spin, sizeof(int), 1, fp);
    }
}

_DEVICE_ void SpinGlass2D::init() {

    // Convolution kernel assigment
    for(int index=0; index<9; index++) {
        _kernel_cross[index] = 0;
        _kernel_semicross[index] = 0;
    }
    _kernel_cross[1] = 1;
    _kernel_cross[3] = 1;
    _kernel_cross[5] = 1;
    _kernel_cross[7] = 1;

    _kernel_semicross[5] = 1;
    _kernel_semicross[7] = 1;

    // Initialize sample
    for(int i=0; i<(N_ROW*N_COL); i++) {
        _sample[i] = (_rand_gen.rand_uniform() <= SPIN_PLUS_PERCENTAGE) ? 1 : -1;
    }

    // Initialize ancillary variables
    _last_delta = 0.0;
    _last_spin = 0;

    // Calculate initial iteration
    SpinGlass2DIterationResult sp_it;

    /// Calculate initial energy and average spin
    for(int index=0; index<(N_ROW*N_COL); index++) {
        sp_it._energy += (double) apply_kernel(_sample, N_ROW, N_COL, index, _kernel_semicross, 3);
        sp_it._average_spin += (int) _sample[index];
    }
    
    _chain->push(sp_it);
}

_DEVICE_ void SpinGlass2D::trial() {
    _trial._accepted = 1;
    _trial._row_index = (int) (_rand_gen.rand_uniform()*N_ROW);
    _trial._col_index = (int) (_rand_gen.rand_uniform()*N_COL);
}

_DEVICE_ double SpinGlass2D::delta() {
    int index = _trial._row_index*N_ROW+_trial._col_index;
    int sum = apply_kernel(_sample, N_ROW, N_COL, index, _kernel_cross, 3); 
    int si = _sample[index];
    _last_delta = 2.0*si*sum;
    return _last_delta;
}

_DEVICE_ void SpinGlass2D::move() {
    int index = _trial._row_index*N_ROW+_trial._col_index;
    _last_spin = _sample[index];
    _sample[index] *= -1;
}

_DEVICE_ void SpinGlass2D::save() {
    SpinGlass2DIterationResult* sp_last_it = (SpinGlass2DIterationResult*) _chain->top();
    SpinGlass2DIterationResult sp_it;
    sp_it._energy = sp_last_it->_energy;
    sp_it._average_spin = sp_last_it->_average_spin;

    if(_trial._accepted) { //* If trial has been accepted
        sp_it._energy += _last_delta;
        sp_it._average_spin -= 2*_last_spin;
    }

    _chain->push(sp_it);
}

//-----------------------------------------------------------------------------|
//                             AUXILIARY FUNCTIONS                             |
//-----------------------------------------------------------------------------|

_DEVICE_ int is_index_in_matrix(char* mat, int n_row, int n_col, int row, int col) {
    if((row >= 0) && (row < n_row) && (col >= 0) && (col < n_col)) {
        return 1;
    }
    return 0;
}

_DEVICE_ int apply_kernel(char* mat, int n_row, int n_col, int index, int* kernel, int kernel_size) {
    int sum=0;

    int center_row = (int) index/n_row;
    int center_col = index%n_row;

    int mat_index, kernel_index = 0;
    int act_row, act_col = 0;

    int padding = (kernel_size-1)/2;
    for(int drow=-padding; drow<=padding; drow++) {
        for(int dcol=-padding; dcol<=padding; dcol++) {

            act_row = center_row+drow;
            act_col = center_col+dcol;

            kernel_index = (drow+padding)*kernel_size+(dcol+padding);
            mat_index = act_row*n_row+act_col;

            if(is_index_in_matrix(mat, n_row, n_col, act_row, act_col)) {
                sum += kernel[kernel_index]*mat[mat_index];
            }
        }
    }
    return sum;
}