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

#ifndef MCMC_H
#define MCMC_H

#include <math.h>
#include <stdlib.h>

#include "constants.h"

/**
 * @name init_chain
 * @remark template
 * @param chains List of all chains
 * @param replica_id The replica_id of the target chain
 * @brief
 * * Initializes the chain of the Replica with replica_id as index
 * @note
 * This function is a template. T is the type of chain to use
*/
template <typename T>
_DEVICE_ void init_chain(T* chains, int replica_id) {
    chains[replica_id].clean();
}

/**
 * @name init_replica
 * @remark template
 * @param replicas List of all Replicas
 * @param rands List of random numbers
 * @param chains List of all chains
 * @param replica_id The replica_id of the target chain
 * @brief
 * * Initializes the Replica with replica_id as index
 * @note
 * This function is a template. T is the type of Replica to use.
 * M is the type of chains to use
*/
template <typename T, typename M>
_DEVICE_ void init_replica(T* replicas, int* rands, M* chains, int replica_id) {
    T* replica = &replicas[replica_id];
    replica->_rand_gen.set_state(rands[replica_id]);
    replica->_chain = &chains[replica_id];
    replica->init();
}

/**
 * @name init_temp
 * @param temps List of all temps
 * @param replica_id The replica_id of the target chain
 * @brief
 * * Initializes the temperature of the Replica with replica_id as index
*/
_DEVICE_ void init_temp(double* temps, int replica_id) {
    temps[replica_id] = INIT_TEMP+(replica_id*TEMP_STEP);
}

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
_DEVICE_ double get_swap_prob(int sw_cand_1, int sw_cand_2, T* replicas, double* temps) {

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

/**
 * @name doSwap
 * @remark template
 * @param temps List of all temps
 * @param replicas List of all Replicas
 * @param sw Swap to be executed
 * @brief
 * * Executes a swap between two Replicas
 * @note
 * This function is a template. T is the type of Replica to use
*/
template <typename T>
_DEVICE_ void doSwap(int replica_id1, int replica_id2, int interval_counter, char* swap_planning, T* replicas, double* temps) {
    double aux_temp = temps[replica_id1];
    temps[replica_id1] = temps[replica_id2];
    temps[replica_id2] = aux_temp;

    MODEL_CHAIN* aux_results = replicas[replica_id1]._chain;
    replicas[replica_id1]._chain = replicas[replica_id2]._chain;
    replicas[replica_id2]._chain = aux_results;

    swap_planning[replica_id1*N_INTERVALS+interval_counter] = 1;
}

#endif