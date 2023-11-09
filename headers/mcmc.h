// Copyright Notice ===========================================================
//
// mcmc.h, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#ifndef _MCMC_H_
#define _MCMC_H_

#include <math.h>
#include <stdlib.h>

#include "tools.h"

/**
 * @class Swap
 * @param _accepted Flag of accepted swap
 * @param _swap_candidate_1 Index of the first swap candidate
 * @param _swap_candidate_2 Index of the second swap candidate
 * @brief
 * * Instances of this class saves the planned swap between two replicas
*/
class Swap {
    public:
        bool _accepted;
        int _swap_candidate_1;
        int _swap_candidate_2;

        /**
         * @name Swap
         * @remark constructor
         * @param sw_cand_1 Index of the first swap candidate
         * @param sw_cand_2 Index of the second swap candidate
        */
        Swap(int sw_cand_1, int sw_cand_2) {
            _accepted = false;
            _swap_candidate_1 = sw_cand_1;
            _swap_candidate_2 = sw_cand_2;
        }
};

/**
 * @name MCMC_iteration
 * @param model A Replica
 * @param temp Temperature to use for the simulation
 * @brief
 * * Does one iteration in MCMC algorithm in one Replica
*/
template <typename T>
void MCMC_iteration(T* model, double temp) {

    // Get a trial
    void* trial = model->trial();
    // Calculate the acceptance probability
    double delta_energy = model->delta(trial);

    double acc_p = 0;
    if(delta_energy <= 0) { acc_p = 1.0; }
    else { acc_p = exp(-delta_energy/temp); }

    // Change state
    double ranf =  rand_uniform();
    if(ranf < acc_p) { model->move(trial); } //* Trial is accepted
    else {
        free(trial);
        trial = nullptr; //* Used as flag of rejected move
    }

    // Save actual state of the model
    model->save(trial);
    free(trial);
}

/**
 * @name get_swap_prob
 * @param models Array of Replica-s
 * @param temps Array of temperatures
 * @return Probabilty to accept the swap
 * @brief
 * * Calculates the probability of accepting the swap
*/
template <typename T>
double get_swap_prob(Swap* sw, T* models, double* temps) {
    int sw_cand_1 = sw->_swap_candidate_1;
    int sw_cand_2 = sw->_swap_candidate_2;

    // Get the evals
    double evals[2];
    evals[0] = models[sw_cand_1].eval();
    evals[1] = models[sw_cand_2].eval();

    // Calculate the swap probability
    double temp_diff = (1/temps[sw_cand_2])-(1/temps[sw_cand_1]);
    double energy_diff = evals[1]-evals[0];
    double aux = exp(temp_diff*energy_diff);
    double swap_prob = aux/(aux+1);

    return swap_prob;
}

#endif