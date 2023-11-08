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

#include "./headers/ising.h"

#include <stdio.h>
#include <stdlib.h>

#include "./headers/constants.h"
#include "./headers/tools.h"

// SPIN_GLASS_ITERATION_RESULT DEFS.

SpinGlassIterationResult::SpinGlassIterationResult() {}

SpinGlassIterationResult::SpinGlassIterationResult(double energy, double average_spin) {
    _energy = energy;
    _average_spin = average_spin;
}

IterationResult* SpinGlassIterationResult::copy() {
    return new SpinGlassIterationResult(_energy, _average_spin);
}

// SPIN_GLASS DEFS.

int SpinGlass::_kernel_cross[] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
int SpinGlass::_kernel_semicross[] = {0, 0, 0, 0, 0, 1, 0, 1, 0};

void SpinGlass::init() {

    // Initialize sample
    for(int i=0; i<(N_ROW*N_COL); i++) {
        if(rand_uniform() <= SPIN_PLUS_PERCENTAGE) { _sample[i] = 1; }
        else { _sample[i] = -1; }
    }

    // Initialize ancillary variables
    _last_delta = 0.0;

    // Calculate initial iteration
    SpinGlassIterationResult* sp_it = new SpinGlassIterationResult();

    /// Calculate initial energy
    sp_it->_energy = calc_energy(_sample); //* ISING MODEL!!!

    /// Calculate initial average spin
    sp_it->_average_spin = calc_average_spin(_sample);

    _results->push(sp_it);
}

void* SpinGlass::trial() {
    int* arr = (int*) malloc(2*sizeof(int));
    arr[0] = (int) (rand_uniform()*N_ROW);
    arr[1] = (int) (rand_uniform()*N_COL);
    return arr;
}

double SpinGlass::eval() {
    SpinGlassIterationResult* sp_it = (SpinGlassIterationResult*) _results->top();
    return sp_it->_energy;
}

double SpinGlass::delta(void* trial) {
    int* trial_int = (int*) trial;
    int index = trial_int[0]*N_ROW+trial_int[1];
    int sum = apply_kernel(_sample, N_ROW, N_COL, index, SpinGlass::_kernel_cross, 3); 
    int si = _sample[trial_int[0]*N_ROW+trial_int[1]];
    _last_delta = 2.0*si*sum;
    return _last_delta;
}

void SpinGlass::move(void* trial) {
    int* trial_int = (int*) trial;
    int index = trial_int[0]*N_ROW+trial_int[1];
    _sample[index] *= -1;
}

void SpinGlass::save(void* trial) {
    SpinGlassIterationResult* sp_last_it = (SpinGlassIterationResult*) _results->top();
    SpinGlassIterationResult* sp_it = (SpinGlassIterationResult*) sp_last_it->copy();

    if(trial != nullptr) { //* If trial has been accepted
        sp_it->_energy += _last_delta;
    }

    sp_it->_average_spin = calc_average_spin(_sample);

    _results->push(sp_it);
}

//

int is_index_in_matrix(char* mat, int n_row, int n_col, int row, int col) {
    if((row >= 0) && (row < n_row) && (col >= 0) && (col < n_col)) {
        return 1;
    }
    return 0;
}

int apply_kernel(char* mat, int n_row, int n_col, int index, int* kernel, int kernel_size) {
    int sum=0;

    int center_row = (int) index/n_row;
    int center_col = index%n_row;

    int mat_index, kernel_index = 0;
    int act_row, act_col = 0;

    int drow, dcol;
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

double calc_energy(char* sample) {
    double sum = 0;
    for(int index=0; index<(N_ROW*N_COL); index++) {
        sum += (double) apply_kernel(sample, N_ROW, N_COL, index, SpinGlass::_kernel_semicross, 3);
    }
    return sum;
}

int calc_average_spin(char* sample) {
    int sum = 0;
    for(int index=0; index<(N_ROW*N_COL); index++) {
        sum += (int) sample[index];
    }
    return sum;
}