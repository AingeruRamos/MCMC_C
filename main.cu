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
    int replica_id = threadIdx.x + blockDim.x * threadIdx.y;
    int initial_rand_state = device_rands[replica_id];

    device_replicas[replica_id]._rand_gen.set_state(initial_rand_state);
    device_replicas[replica_id].init();
}

__global__ void cuda_init_temps(double* device_temps) {
    int replica_id = threadIdx.x + blockDim.x * threadIdx.y;
    device_temps[replica_id] = INIT_TEMP+(replica_id*TEMP_STEP);
}

__global__ void cuda_run(SpinGlass* device_replicas, double* device_temps) {
    int replica_id = threadIdx.x + blockDim.x * threadIdx.y;
    SpinGlass* sp = &device_replicas[replica_id];
    double temp = device_temps[replica_id];

    for(int iteration=1; iteration<N_ITERATIONS; iteration++) {
        MCMC_iteration<SpinGlass>(sp, temp);
    }
}

__global__ void cuda_print(SpinGlass* device_replicas, int replica_id) {
    SpinGlass* sp = &device_replicas[replica_id];

    for(int i=0; i<N_ITERATIONS; i++) {
        SpinGlassIterationResult* sp_it = sp->_results.get(i);
        printf("%f,", sp_it->_energy);
    }
    printf("\n");
    for(int i=0; i<N_ITERATIONS; i++) {
        SpinGlassIterationResult* sp_it = sp->_results.get(i);
        printf("%d,", sp_it->_average_spin);
    }
    printf("\n");
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
    _CUDA(cudaMalloc((void**)&device_rands, TOTAL_REPLICAS*sizeof(int)));
    _CUDA(cudaMemcpy(device_rands, host_rands, sizeof(int), cudaMemcpyHostToDevice))

    SpinGlass* device_replicas;
    _CUDA(cudaMalloc((void**)&device_replicas, TOTAL_REPLICAS*sizeof(SpinGlass)));
    cuda_init_replicas<<<1,TOTAL_REPLICAS>>>(device_replicas, device_rands);

    if(DEBUG_FLOW) { printf("Device -> Replicas: OK\n"); }

    double* device_temps;
    _CUDA(cudaMalloc((void**)&device_temps, TOTAL_REPLICAS*sizeof(double)));
    cuda_init_temps<<<1,TOTAL_REPLICAS>>>(device_temps);

    if(DEBUG_FLOW) { printf("Device -> Temps: OK\n"); }

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    cuda_run<<<1,TOTAL_REPLICAS>>>(device_replicas, device_temps);
    if(DEBUG_FLOW) { printf("Device -> Run: OK\n"); }

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    printf("%d*\n", TOTAL_REPLICAS); // SIMULATION CONSTANTS
    printf("%d\n", N_ITERATIONS);
    printf("%d\n", SWAP_ACTIVE);
    printf("%f,%f,%f,%d\n", INIT_TEMP, END_TEMP, TEMP_STEP, TOTAL_REPLICAS);

    printf("#\n"); // MODEL CONSTANTS
    
    printf("%d\n", N_ROW);
    printf("%d\n", N_COL);
    printf("%f\n", SPIN_PLUS_PERCENTAGE);

    printf("#\n");

    // TODO Print Swapping

    printf("#\n"); // TIME

    //printf("%f\n", total_all);
    //printf("%f\n", total_exec);

    printf("#\n"); // RESULTS

    if(SWAP_ACTIVE) {
        // TODO Reorder results
        // TODO Print results
    } else {
        for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
            cuda_print<<<1,1>>>(device_replicas, replica_id);
            cudaDeviceSynchronize();
            printf("#\n");
        }
    }

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    _CUDA(cudaFree(device_replicas));
    _CUDA(cudaFree(device_temps));

    return 0;
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

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
