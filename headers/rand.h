// Copyright Notice ===========================================================
//
// rand.h, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#ifndef _RAND_H_
#define _RAND_H_

#include "constants.h"

_CUDA_DECOR_ double rand_uniform();
_CUDA_DECOR_ double rand_uniform(double start, double end);

#endif