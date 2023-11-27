// Copyright Notice ===========================================================
//
// main.cu, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#include "./header/constants.h"
#include "./header/ising.h"
#include "./header/mcmc.h"

#define _CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

__global__ void cuda_init_replicas(SpinGlass* device_replicas, int* device_rands) {
    int replica_id = 0; //* <<<<<<
    int initial_rand_state = device_rands[replica_id];

    device_replicas[replica_id]._rand_gen.set_state(initial_rand_state);
    device_replicas[replica_id].init();
}

__global__ void cuda_init_temps(double* device_temps) {
    int replica_id = 0; //* <<<<<<
    device_temps[replica_id] = INIT_TEMP+(replica_id*TEMP_STEP);
}

__global__ void cuda_run(SpinGlass* device_replicas, double* device_temps) {
    int replica_id = 0; //* <<<<<<
    SpinGlass* sp = &device_replicas[replica_id];
    double temp = device_temps[replica_id];

    for(int iteration=1; iteration<N_ITERATIONS; iteration++) {
        MCMC_iteration<SpinGlass>(sp, temp);
    }
}

__global__ void cuda_print(SpinGlass* device_replicas) {
    int replica_id = 0; //* <<<<<<
    SpinGlass* sp = &device_replicas[replica_id];

    SpinGlassIterationResult* sp_it;
    for(int i=0; i<N_ITERATIONS; i++) {
        sp_it = sp->_results.get(i);
        printf("%d ", sp_it->_average_spin);
    }
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

void option_enabeler(int argc, char** argv);

int DEBUG_FLOW = 0;
int DEBUG_RESULTS = 0;
int N_THREADS = 1;

int main(int argc, char** argv) {

    srand(time(NULL));

    option_enabeler(argc, argv);

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|
    int* host_rands = (int*) malloc(1*sizeof(int));
    host_rands[0] = rand();

    int* device_rands;
    _CUDA(cudaMalloc((void**)&device_rands, 1*sizeof(int)));
    _CUDA(cudaMemcpy(device_rands, host_rands, sizeof(int), cudaMemcpyHostToDevice))

    SpinGlass* device_replicas;
    _CUDA(cudaMalloc((void**)&device_replicas, 1*sizeof(SpinGlass)));
    cuda_init_replicas<<<1,1>>>(device_replicas, device_rands);

    if(DEBUG_FLOW) { printf("Device -> Replicas: OK\n"); }

    double* device_temps;
    _CUDA(cudaMalloc((void**)&device_temps, 1*sizeof(double)));
    cuda_init_temps<<<1,1>>>(device_temps);

    if(DEBUG_FLOW) { printf("Device -> Temps: OK\n"); }

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    cuda_run<<<1,1>>>(device_replicas, device_temps);
    if(DEBUG_FLOW) { printf("Device -> Run: OK\n"); }

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|
    // OBTENER LOS RESULTADOS EN EL HOST
    // Y REORDENARLOS

    Stack<MODEL_ITER*, N_ITERATIONS>* results = (Stack<MODEL_ITER*, N_ITERATIONS>*)
                                    malloc(1*sizeof(Stack<MODEL_ITER*, N_ITERATIONS>));

    cuda_print<<<1,1>>>(device_replicas);

    free(results);

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

// LIBERAR TODA LA MEMORIA (LA DEL HOST FALTA (EN OPENMP TAMBIEN))

    _CUDA(cudaFree(device_replicas));
    _CUDA(cudaFree(device_temps));

    return 0;
}

void option_enabeler(int argc, char** argv) {
    for(int i=1; i<argc; i++) {

        if(strcmp(argv[i], "-df") == 0) {
            DEBUG_FLOW = 1;
            continue;
        } else if(strcmp(argv[i], "-dr") == 0) {
            DEBUG_RESULTS = 1;
            continue;
        } 
    }
}
