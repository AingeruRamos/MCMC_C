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
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#include "../header/constants.h"
#include "../header/ising.h"
#include "../header/mcmc.h"

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

__global__ void cuda_init_results(MODEL_RESULTS* device_results) {
    int replica_id = threadIdx.x + blockDim.x * threadIdx.y;
    device_results[replica_id].clean();
}

__global__ void cuda_init_replicas(MODEL_NAME* device_replicas, int* device_rands, MODEL_RESULTS* device_results) {
    int replica_id = threadIdx.x + blockDim.x * threadIdx.y;
    int initial_rand_state = device_rands[replica_id];

    device_replicas[replica_id]._rand_gen.set_state(initial_rand_state);
    device_replicas[replica_id]._results = &device_results[replica_id];
    device_replicas[replica_id].init();
}

__global__ void cuda_init_temps(double* device_temps) {
    int replica_id = threadIdx.x + blockDim.x * threadIdx.y;
    device_temps[replica_id] = INIT_TEMP+(replica_id*TEMP_STEP);
}

__global__ void cuda_init_n_swaps(int* device_n_swaps) {
    for(int i=0; i<N_ITERATIONS; i++) {
        device_n_swaps[i] = (int) (TOTAL_REPLICAS/2);
        if((i%2 != 0) && (TOTAL_REPLICAS%2 == 0)) { //* Number of swaps in odd iterations
            device_n_swaps[i] -= 1;
        }
    }
}

__global__ void cuda_init_swap_planning(Swap*** device_swap_planning, int* device_n_swaps) {
    for(int i=0; i<N_ITERATIONS; i++) {
        device_swap_planning[i] = (Swap**) malloc(device_n_swaps[i]*sizeof(Swap*));

        int sw_cand_1 = 0; //* Defining the starting point
        if(i%2 != 0) { sw_cand_1 = 1; }

        for(int j=0; j<device_n_swaps[i]; j++) {
            device_swap_planning[i][j] = new Swap(sw_cand_1, (sw_cand_1+1));
            sw_cand_1 += 2;
        }
    }
}

__global__ void cuda_run_iteration(MODEL_NAME* device_replicas, double* device_temps) {
    int replica_id = threadIdx.x + blockDim.x * threadIdx.y;
    SpinGlass* sp = &device_replicas[replica_id];
    double temp = device_temps[replica_id];

    MCMC_iteration<MODEL_NAME>(sp, temp);
}

__global__ void cuda_run_swaps(MODEL_NAME* device_replicas, double* device_temps, int* device_n_swaps, Swap*** device_swap_planning, int iteration) {
    int swap_index=threadIdx.x + blockDim.x * threadIdx.y;

    if(swap_index < device_n_swaps[iteration-1]) {
        Swap* swap = device_swap_planning[iteration-1][swap_index];
        double swap_prob = get_swap_prob<MODEL_NAME>(swap, device_replicas, device_temps);

        double r = device_replicas[swap->_swap_candidate_1]._rand_gen.rand_uniform();
        if(r < swap_prob) {
            double aux_temp = device_temps[swap->_swap_candidate_1];
            device_temps[swap->_swap_candidate_1] = device_temps[swap->_swap_candidate_2];
            device_temps[swap->_swap_candidate_2] = aux_temp;

            MODEL_RESULTS* aux_results = device_replicas[swap->_swap_candidate_1]._results;
            device_replicas[swap->_swap_candidate_1]._results = device_replicas[swap->_swap_candidate_2]._results;
            device_replicas[swap->_swap_candidate_2]._results = aux_results;
            swap->_accepted = true;
        }
    }
}

__global__ void cuda_print_swaps(int* device_n_swaps, Swap*** device_swap_planning) {
    for(int i=0; i<N_ITERATIONS; i++) { // SWAP PLANNING (ACCEPTED)
        for(int j=0; j<device_n_swaps[i]; j++) {
            Swap* sw = device_swap_planning[i][j];
            if(sw->_accepted) {
                printf("%d-%d,", sw->_swap_candidate_1, sw->_swap_candidate_2);
            }
        }
        printf("\n");
    }
}

__global__ void cuda_print(MODEL_RESULTS* device_results, int replica_id) {
    MODEL_RESULTS* results = &device_results[replica_id];

    for(int i=0; i<N_ITERATIONS; i++) {
        MODEL_ITER* sp_it = results->get(i);
        printf("%f,", sp_it->_energy);
    }
    printf("\n");
    for(int i=0; i<N_ITERATIONS; i++) {
        MODEL_ITER* sp_it = results->get(i);
        printf("%d,", sp_it->_average_spin);
    }
    printf("\n");
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

void option_enabeler(int argc, char** argv);

int DEBUG_FLOW = 0;
int DEBUG_RESULTS = 0;

int main(int argc, char** argv) {

    srand(time(NULL));

    option_enabeler(argc, argv);

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    time_t begin_all = time(NULL);

    int NUM_BLOCKS = ((TOTAL_REPLICAS)/1024)+1; //* 1024 is the number of thread per block
    int NUM_THREADS = (TOTAL_REPLICAS)%1024;

    int* device_rands;
    _CUDA(cudaMalloc((void**)&device_rands, TOTAL_REPLICAS*sizeof(int)));

    if(DEBUG_FLOW) { printf("Device -> Rands: OK\n"); }

    MODEL_RESULTS* device_results;
    _CUDA(cudaMalloc((void**)&device_results, TOTAL_REPLICAS*sizeof(MODEL_RESULTS)));
    cuda_init_results<<<NUM_BLOCKS, NUM_THREADS>>>(device_results);

    if(DEBUG_FLOW) { printf("Device -> Results: OK\n"); }

    MODEL_NAME* device_replicas;
    _CUDA(cudaMalloc((void**)&device_replicas, TOTAL_REPLICAS*sizeof(MODEL_NAME)));
    cuda_init_replicas<<<NUM_BLOCKS, NUM_THREADS>>>(device_replicas, device_rands, device_results);

    if(DEBUG_FLOW) { printf("Device -> Replicas: OK\n"); }

    double* device_temps;
    _CUDA(cudaMalloc((void**)&device_temps, TOTAL_REPLICAS*sizeof(double)));
    cuda_init_temps<<<NUM_BLOCKS, NUM_THREADS>>>(device_temps);

    if(DEBUG_FLOW) { printf("Device -> Temps: OK\n"); }

    int* device_n_swaps;
    if(SWAP_ACTIVE) {
        _CUDA(cudaMalloc((void**)&device_n_swaps, N_ITERATIONS*sizeof(int)));
        cuda_init_n_swaps<<<1,1>>>(device_n_swaps);
    }

    Swap*** device_swap_planning;
    if(SWAP_ACTIVE) {
        _CUDA(cudaMalloc((void**)&device_swap_planning, N_ITERATIONS*sizeof(Swap**)));
        cuda_init_swap_planning<<<1,1>>>(device_swap_planning, device_n_swaps);
    }

    if(DEBUG_FLOW) { printf("Device -> Swaps: OK\n"); }

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    time_t begin_exec = time(NULL);

    for(int iteration=1; iteration<N_ITERATIONS; iteration++) {
        cuda_run_iteration<<<NUM_BLOCKS, NUM_THREADS>>>(device_replicas, device_temps);
        if(SWAP_ACTIVE) {
            cuda_run_swaps<<<NUM_BLOCKS, NUM_THREADS>>>(device_replicas, device_temps, device_n_swaps, device_swap_planning, iteration);
        }
    }

    time_t end_exec = time(NULL);

    if(DEBUG_FLOW) { printf("Device -> Run: OK\n"); }

    time_t end_all = time(NULL);

    double total_all = (double) (end_all-begin_all);
    double total_exec = (double) (end_exec-begin_exec);

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
    // SWAP PLANNING (ACCEPTED)
    if(SWAP_ACTIVE) {
        cuda_print_swaps<<<1,1>>>(device_n_swaps, device_swap_planning);
        cudaDeviceSynchronize();
    }

    printf("#\n"); // TIME

    printf("%f\n", total_all);
    printf("%f\n", total_exec);

    printf("#\n"); // RESULTS

    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
            cuda_print<<<1,1>>>(device_results, replica_id);
            cudaDeviceSynchronize();
            printf("#\n");
    }

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    _CUDA(cudaFree(device_rands));
    _CUDA(cudaFree(device_results));
    _CUDA(cudaFree(device_replicas));
    _CUDA(cudaFree(device_temps));

    if(SWAP_ACTIVE) {
        _CUDA(cudaFree(device_n_swaps));
        _CUDA(cudaFree(device_swap_planning));
    }

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
