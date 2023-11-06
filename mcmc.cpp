// Copyright Notice ===========================================================
//
// mcmc.cpp, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#include "./headers/mcmc.h"

#include <math.h>
#include <stdlib.h>

#include "./headers/mcmc.h"
#include "./headers/tools.h"

// REPLICA_RESULT DEFS.

ReplicaResult::ReplicaResult(int n_iterations) {
    _n_iterations = n_iterations;
    _act_iteration = 0;
    _iteration_list = (IterationResult**) malloc(_n_iterations*sizeof(IterationResult*));
}

void ReplicaResult::push(IterationResult* it_res) {
    _iteration_list[_act_iteration] = it_res;
    _act_iteration++;
}

void ReplicaResult::set(IterationResult* it_res, int iteration) {
    _iteration_list[iteration] = it_res;
}

void ReplicaResult::swap(ReplicaResult* r1, ReplicaResult* r2, int iteration) {
    IterationResult* aux = r1->get(iteration);
    r1->set(r2->get(iteration), iteration);
    r2->set(aux, iteration);
}

IterationResult* ReplicaResult::get(int iteration) {
    return _iteration_list[iteration];
}

IterationResult* ReplicaResult::pop() {
    return _iteration_list[_act_iteration-1];
}

ReplicaResult* ReplicaResult::copy() {
    ReplicaResult* res_copy = (ReplicaResult*) malloc(sizeof(ReplicaResult));
    res_copy->_n_iterations = _n_iterations;
    res_copy->_act_iteration = _act_iteration;
    res_copy->_iteration_list = (IterationResult**) malloc(_n_iterations*sizeof(IterationResult*));
    for(int i=0; i<_n_iterations; i++) {
        res_copy->push(_iteration_list[i]->copy());
    }
    return res_copy;
}

// SWAP DEFS.

Swap::Swap(int sw_cand_1, int sw_cand_2) {
    _accepted = false;
    _swap_candidate_1 = sw_cand_1;
    _swap_candidate_2 = sw_cand_2;
}

//

void MCMC_iteration(Replica* model, double temp) {

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

double get_swap_prob(Swap* sw, Replica** models, double* temps) {
    int sw_cand_1 = sw->_swap_candidate_1;
    int sw_cand_2 = sw->_swap_candidate_2;

    // Get the evals
    double* evals = (double*) malloc(2*sizeof(double));
    evals[0] = models[sw_cand_1]->eval();
    evals[1] = models[sw_cand_2]->eval();

    // Calculate the swap probability
    double temp_diff = (1/temps[sw_cand_2])-(1/temps[sw_cand_1]);
    double energy_diff = evals[1]-evals[0];
    double aux = exp(temp_diff*energy_diff);
    double swap_prob = aux/(aux+1);

    free(evals);

    return swap_prob;
}