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