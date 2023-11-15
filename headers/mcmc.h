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

#include "constants.h"
#include "rand.h"

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
        _CUDA_DECOR_ Swap(int sw_cand_1, int sw_cand_2) {
            _accepted = false;
            _swap_candidate_1 = sw_cand_1;
            _swap_candidate_2 = sw_cand_2;
        }
};

/**
 * @name MCMC_iteration
 * @remark template
 * @param replica A Replica
 * @param temp Temperature to use for the simulation
 * @brief
 * * Does one iteration in MCMC algorithm in one Replica
 * @note
 * This function is a template. T is the type of replica to use
*/
template <typename T>
_CUDA_DECOR_ void MCMC_iteration(T* replica, double temp) {

    // Get a trial
    void* trial = replica->trial();

    // Calculate the acceptance probability
    double delta_energy = replica->delta(trial);

    double acc_p = 0;
    if(delta_energy <= 0) { acc_p = 1.0; }
    else { acc_p = exp(-delta_energy/temp); }

    // Change state
    double ranf =  rand_uniform();
    if(ranf < acc_p) { replica->move(trial); } //* Trial is accepted
    else {
        free(trial);
        trial = nullptr; //* Used as flag of rejected move
    }

    // Save actual state of the replica
    replica->save(trial);
    free(trial);
}

/**
 * @name get_swap_prob
 * @remark template
 * @param replicas Array of replicas
 * @param temps Array of temperatures
 * @return Probabilty to accept the swap
 * @brief
 * * Calculates the probability of accepting the swap
 * @note
 * This function is a template. T is the type of replica to use
*/
template <typename T>
_CUDA_DECOR_ double get_swap_prob(Swap* sw, T* replicas, double* temps) {
    int sw_cand_1 = sw->_swap_candidate_1;
    int sw_cand_2 = sw->_swap_candidate_2;

    // Get the evals
    double evals[2];
    evals[0] = replicas[sw_cand_1]._results.top()->_energy;
    evals[1] = replicas[sw_cand_2]._results.top()->_energy;

    // Calculate the swap probability
    double temp_diff = (1/temps[sw_cand_2])-(1/temps[sw_cand_1]);
    double energy_diff = evals[1]-evals[0];
    double aux = exp(temp_diff*energy_diff);
    double swap_prob = aux/(aux+1);

    return swap_prob;
}

#endif