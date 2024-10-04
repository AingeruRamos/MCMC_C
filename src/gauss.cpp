// Copyright Notice ===========================================================
//
// gauss.cpp, Copyright (c) 2023-2024 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#include "../header/gauss.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../header/constants.h"
#include "../header/rand.h"

//===========================================================================//

_HOST_ _DEVICE_ Gauss1DIterationResult::Gauss1DIterationResult() {
    _energy = 0;
}

_HOST_ void print_chain(Stack<Gauss1DIterationResult, N_ITERATIONS>* chain, FILE* fp) {
    Gauss1DIterationResult* g_it;

    for(int i=0; i<N_ITERATIONS; i++) {
        g_it = (Gauss1DIterationResult*) chain->get(i);
        fwrite(&g_it->_energy, sizeof(float), 1, fp);
    }
}

_DEVICE_ void Gauss1D::init() {

    _trial._accepted = 0;

    // Initialize ancillary variables
    _x = 2.0;
    _last_delta = 0.0;

    // Calculate initial iteration
    Gauss1DIterationResult g_it;
    g_it._energy = _x;
    
    _chain->push(g_it);
}

_DEVICE_ void Gauss1D::trial() {
    _trial._accepted = 1;
    _trial._x = (float) _rand_gen.rand_normal(_x, N_ROW);
}

_DEVICE_ double Gauss1D::delta() {
    float d_act = dnorm(_x);
    float d_trial = dnorm(_trial._x);

    _last_delta = d_trial/d_act;
    return _last_delta;
}

_DEVICE_ void Gauss1D::move() {
    _x = _trial._x;
}

_DEVICE_ void Gauss1D::save() {
    Gauss1DIterationResult* g_last_it = (Gauss1DIterationResult*) _chain->top();
    Gauss1DIterationResult g_it;
    g_it._energy = g_last_it->_energy;

    if(_trial._accepted) { //* If trial has been accepted
        g_it._energy = _x;
    }

    _chain->push(g_it);
}

//===========================================================================//

_DEVICE_ float dnorm(float x) {
    float p1 = 0.3989422804; // Inverse of 2*PI square root 
    float p2 = exp((x*x)/-2);
    return p1*p2;
}
