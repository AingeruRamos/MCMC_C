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
        _DEVICE_ Swap(int sw_cand_1, int sw_cand_2) {
            _accepted = false;
            _swap_candidate_1 = sw_cand_1;
            _swap_candidate_2 = sw_cand_2;
        }
};

/***/
_HOST_ void print_swap_list(Swap* swap_planning, int offset, int n_swaps, FILE* fp) {
    int i_print;

    for(int j=0; j<n_swaps; j++) {
        Swap* sw = &swap_planning[offset+j];
        if(sw->_accepted) {
            fwrite(&sw->_swap_candidate_1, sizeof(int), 1, fp);
            fwrite(&sw->_swap_candidate_2, sizeof(int), 1, fp);
        }
    }
    fwrite(&(i_print=-1), sizeof(int), 1, fp);
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

template <typename T>
_DEVICE_ void init_chain(T* chain, int replica_id) {
    chain[replica_id].clean();
}

template <typename T, typename M>
_DEVICE_ void init_replica(T* replicas, int* rands, M* chains, int replica_id) {
    T* replica = &replicas[replica_id];
    replica->_rand_gen.set_state(rands[replica_id]);
    replica->_chain = &chains[replica_id];
    replica->init();
}

_DEVICE_ void init_temp(double* temps, int replica_id) {
    temps[replica_id] = INIT_TEMP+(replica_id*TEMP_STEP);
}

_HOST_ _DEVICE_ void init_swap_list_offsets(int* swap_list_offsets) {
    swap_list_offsets[0] = 0;
    int aux = (int) (TOTAL_REPLICAS)/2;

    for(int i=1; i<=N_ITERATIONS; i++) {
        swap_list_offsets[i] = aux+swap_list_offsets[i-1];
        if((i-1) % 2 != 0) {
            swap_list_offsets[i] -= 1;
        }
    }
}

_HOST_ _DEVICE_ void init_swap_planning(int* swap_list_offsets, Swap* swap_planning) {
    for(int iteration=0; iteration<N_ITERATIONS; iteration++) {
        
        int sw_cand_1 = 0; //* Defining the starting point
        if(iteration % 2 != 0) {
            sw_cand_1 = 1;
        }

        int offset = swap_list_offsets[iteration];
        int n_swaps = swap_list_offsets[iteration+1]-offset;
        for(int j=0; j<n_swaps; j++) {
            swap_planning[offset+j]._accepted = false;
            swap_planning[offset+j]._swap_candidate_1 = sw_cand_1;
            swap_planning[offset+j]._swap_candidate_2 = (sw_cand_1+1);
            sw_cand_1 += 2;
        }
    }
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

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
_DEVICE_ void MCMC_iteration(T* replica, double temp) {

    // Get a trial
    replica->trial();

    // Calculate the acceptance probability
    double delta_energy = replica->delta();

    double acc_p = (delta_energy <= 0) ? 1.0 : exp(-delta_energy/temp);

    // Change state
    double ranf =  replica->_rand_gen.rand_uniform();
    if(ranf < acc_p) { replica->move(); } //* Trial is accepted
    else {
        replica->_trial._accepted = 0; //* Used as flag of rejected move
    }

    // Save actual state of the replica
    replica->save();
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
_DEVICE_ double get_swap_prob(Swap* sw, T* replicas, double* temps) {
    int sw_cand_1 = sw->_swap_candidate_1;
    int sw_cand_2 = sw->_swap_candidate_2;

    // Get the evals
    double evals[2];
    evals[0] = replicas[sw_cand_1]._chain->top()->_energy;
    evals[1] = replicas[sw_cand_2]._chain->top()->_energy;

    // Calculate the swap probability
    double temp_diff = (1/temps[sw_cand_2])-(1/temps[sw_cand_1]);
    double energy_diff = evals[1]-evals[0];
    double aux = exp(temp_diff*energy_diff);
    double swap_prob = aux/(aux+1);

    return swap_prob;
}

template <typename T>
_DEVICE_ void doSwap(double* temps, T* replicas, Swap* sw) {
    int replica_id1 = sw->_swap_candidate_1;
    int replica_id2 = sw->_swap_candidate_2;

    double aux_temp = temps[replica_id1];
    temps[replica_id1] = temps[replica_id2];
    temps[replica_id2] = aux_temp;

    MODEL_CHAIN* aux_results = replicas[replica_id1]._chain;
    replicas[replica_id1]._chain = replicas[replica_id2]._chain;
    replicas[replica_id2]._chain = aux_results;

    sw->_accepted = true;
}

#endif