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

/**
 * @name rand_uniform
 * @return A random value
 * @brief
 * * Calculates a random in the raqnge of [0, 1]
*/
_CUDA_DECOR_ double rand_uniform();

/**
 * @name rand_uniform
 * @overload
 * @return A random value
 * @brief
 * * Calculates a random in the range of [start, end]
*/
_CUDA_DECOR_ double rand_uniform(double start, double end);

#endif