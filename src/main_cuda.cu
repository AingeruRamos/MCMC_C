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
#include <sys/time.h>
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

__global__ void cuda_init_chains(MODEL_CHAIN* device_chains) {
    int replica_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(replica_id < TOTAL_REPLICAS) {
        init_chain<MODEL_CHAIN>(device_chains, replica_id);
    }
}

__global__ void cuda_init_replicas(MODEL_NAME* device_replicas, int* device_rands, MODEL_CHAIN* device_chains) {
    int replica_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(replica_id < TOTAL_REPLICAS) {
        init_replica<MODEL_NAME, MODEL_CHAIN>(device_replicas, device_rands, device_chains, replica_id);
    }
}

__global__ void cuda_init_temps(double* device_temps) {
    int replica_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(replica_id < TOTAL_REPLICAS) {
        init_temp(device_temps, replica_id);
    }
}

__global__ void cuda_run_n_iterations(MODEL_NAME* device_replicas, double* device_temps, int n_iterations) {
    int replica_id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(replica_id < TOTAL_REPLICAS) {
        MODEL_NAME* replica = &device_replicas[replica_id];
        double temp = device_temps[replica_id];

        for(int iteration=0; iteration<n_iterations; iteration++) {
            MCMC_iteration<MODEL_NAME>(replica, temp);
        }
    }
}

__global__ void cuda_run_swaps(MODEL_NAME* device_replicas, double* device_temps, int offset, int n_swaps, Swap* device_swap_planning) {
    int swap_index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(swap_index < n_swaps) {
        Swap* swap = &device_swap_planning[offset+swap_index];
        double swap_prob = get_swap_prob<MODEL_NAME>(swap, device_replicas, device_temps);

        double r = device_replicas[swap->_swap_candidate_1]._rand_gen.rand_uniform();
	if(r < swap_prob) {
            doSwap<MODEL_NAME>(device_temps, device_replicas, swap);
        }
    }
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

void option_enabeler(int argc, char** argv);
int getNextSwapIteration(int* swap_list_offsets, int start_iteration);

int DEBUG_NO_SWAPS = 0;
int DEBUG_NO_CHAINS = 0;
int WARPS_PER_BLOCK = 0;

int main(int argc, char** argv) {

    srand(time(NULL));

    option_enabeler(argc, argv);

    if(SWAP_ACTIVE && (TOTAL_REPLICAS <= 1)) {
        printf("ERROR: This program is not valid.\n");
        return -1;
    }

    FILE* fp = fopen(argv[1], "wb");

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    clock_t begin_init = clock();

    int NUM_BLOCKS = (int) ((TOTAL_REPLICAS/(32*WARPS_PER_BLOCK))+1);

    int* device_rands;
    _CUDA(cudaMalloc((void**)&device_rands, TOTAL_REPLICAS*sizeof(int)));

    MODEL_CHAIN* host_chains;
    //_CUDA(cudaHostAlloc((void**)&host_chains, TOTAL_REPLICAS*sizeof(MODEL_CHAIN), cudaHostAllocMapped));
    host_chains = (MODEL_CHAIN*) malloc(TOTAL_REPLICAS*sizeof(MODEL_CHAIN));

    MODEL_CHAIN* device_chains;
    //cudaHostGetDevicePointer((void**)&device_chains,  (void*)host_chains, 0);
    _CUDA(cudaMalloc((void**)&device_chains, TOTAL_REPLICAS*sizeof(MODEL_CHAIN)));
    cuda_init_chains<<<NUM_BLOCKS, 32*WARPS_PER_BLOCK>>>(device_chains);

    MODEL_NAME* device_replicas;
    _CUDA(cudaMalloc((void**)&device_replicas, TOTAL_REPLICAS*sizeof(MODEL_NAME)));
    cuda_init_replicas<<<NUM_BLOCKS, 32*WARPS_PER_BLOCK>>>(device_replicas, device_rands, device_chains);

    _CUDA(cudaFree(device_rands));

    double* device_temps;
    _CUDA(cudaMalloc((void**)&device_temps, TOTAL_REPLICAS*sizeof(double)));
    cuda_init_temps<<<NUM_BLOCKS, 32*WARPS_PER_BLOCK>>>(device_temps);

    int *host_swap_list_offsets;
    Swap *host_swap_planning, *device_swap_planning;

    if(SWAP_ACTIVE) {
        host_swap_list_offsets = (int*) malloc((N_ITERATIONS+1)*sizeof(int));
        init_swap_list_offsets(host_swap_list_offsets);

        host_swap_planning = (Swap*) malloc(host_swap_list_offsets[N_ITERATIONS]*sizeof(Swap));
        init_swap_planning(host_swap_list_offsets, host_swap_planning);

        _CUDA(cudaMalloc((void**)&device_swap_planning, host_swap_list_offsets[N_ITERATIONS]*sizeof(Swap)));
        _CUDA(cudaMemcpy(device_swap_planning, host_swap_planning, host_swap_list_offsets[N_ITERATIONS]*sizeof(Swap), cudaMemcpyHostToDevice));
    }

    cudaDeviceSynchronize();
    clock_t end_init = clock();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    clock_t begin_exec = clock();
    
    int iteration, run_iterations, offset, new_offset, n_swaps;
    int NUM_BLOCKS_SWAPS;

    iteration = 1;

    if(SWAP_ACTIVE) {
    	offset = host_swap_list_offsets[iteration];
    }

    while(iteration < N_ITERATIONS) {
        
        // Get the next swap iteration. If SWAP_ACTIVE == 0, returns N_ITERATIONS-1
        int next_swap_iteration = getNextSwapIteration(host_swap_list_offsets, iteration);
        
        // Execute all iterations before the swap iteration
        run_iterations = next_swap_iteration-iteration+1;

        cuda_run_n_iterations<<<NUM_BLOCKS, 32*WARPS_PER_BLOCK>>>(device_replicas, device_temps, run_iterations);

        iteration += run_iterations;
        if(iteration == N_ITERATIONS) { break; }  //* We reach the limit of iterations

	    new_offset = host_swap_list_offsets[iteration-1];
	    n_swaps = new_offset-offset;

        NUM_BLOCKS_SWAPS = (int) ((n_swaps/(32*WARPS_PER_BLOCK))+1);

        // Execute the swap iteration
         //* The last for statement inc the iteration. We need to take the before iteration
        cuda_run_swaps<<<NUM_BLOCKS_SWAPS, 32*WARPS_PER_BLOCK>>>(device_replicas, device_temps, 
							offset, n_swaps, device_swap_planning);

	offset = new_offset;
    }
    cudaDeviceSynchronize();
    
    clock_t end_exec = clock();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    clock_t begin_print = clock();

    int i_print;
    float f_print;

    fwrite(&(i_print=1), sizeof(int), 1, fp); // CUDA EXECUTED FLAG

    fwrite(&(i_print=NUM_BLOCKS), sizeof(int), 1, fp); // SIMULATION CONSTANTS
    fwrite(&(i_print=WARPS_PER_BLOCK), sizeof(int), 1, fp);
    fwrite(&(i_print=N_ITERATIONS), sizeof(int), 1, fp);
    fwrite(&(i_print=SWAP_ACTIVE), sizeof(int), 1, fp);
    fwrite(&(i_print=SWAP_INTERVAL), sizeof(int), 1, fp);

    fwrite(&(f_print=INIT_TEMP), sizeof(float), 1, fp);
    fwrite(&(f_print=END_TEMP), sizeof(float), 1, fp);
    fwrite(&(f_print=TEMP_STEP), sizeof(float), 1, fp);
    fwrite(&(i_print=TOTAL_REPLICAS), sizeof(int), 1, fp);

    fwrite(&(i_print=N_ROW), sizeof(int), 1, fp); // MODEL CONSTANTS
    fwrite(&(i_print=N_COL), sizeof(int), 1, fp);
    fwrite(&(f_print=SPIN_PLUS_PERCENTAGE), sizeof(float), 1, fp);

    if(!DEBUG_NO_SWAPS && SWAP_ACTIVE) { // SWAP PLANNING (ACCEPTED)
        fwrite(&(i_print=1), sizeof(int), 1, fp); //* Flag of printed swaps
        _CUDA(cudaMemcpy(host_swap_planning, device_swap_planning, host_swap_list_offsets[N_ITERATIONS]*sizeof(Swap), cudaMemcpyDeviceToHost));

        for(int iteration=0; iteration<N_ITERATIONS; iteration++) {
            int offset = host_swap_list_offsets[iteration];
            int n_swaps = host_swap_list_offsets[iteration+1]-offset;
            print_swap_list(host_swap_planning, offset, n_swaps, fp);
        }
    } else {
        fwrite(&(i_print=0), sizeof(int), 1, fp); //* Flag of NO printed swaps
    }

    if(!DEBUG_NO_CHAINS) { // CHAINS
        fwrite(&(i_print=1), sizeof(int), 1, fp); //* Flag of printed chains

        _CUDA(cudaMemcpy(host_chains, device_chains, TOTAL_REPLICAS*sizeof(MODEL_CHAIN), cudaMemcpyDeviceToHost));

        for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
            print_chain(&host_chains[replica_id], fp);
        }
    } else {
        fwrite(&(i_print=0), sizeof(int), 1, fp); //* Flag of NO printed chains
    }

    clock_t end_print = clock();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    double total_init = (double)(end_init-begin_init)/CLOCKS_PER_SEC;
    double total_exec = (double)(end_exec-begin_exec)/CLOCKS_PER_SEC;
    double total_print = (double)(end_print-begin_print)/CLOCKS_PER_SEC;

    fwrite(&(f_print=total_init), sizeof(float), 1, fp); // TIME
    fwrite(&(f_print=total_exec), sizeof(float), 1, fp);
    fwrite(&(f_print=total_print), sizeof(float), 1, fp);

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    fclose(fp);

    free(host_chains);
    _CUDA(cudaFree(device_chains));
    _CUDA(cudaFree(device_replicas));
    _CUDA(cudaFree(device_temps));

    if(SWAP_ACTIVE) {
        free(host_swap_list_offsets);
        free(host_swap_planning);
        _CUDA(cudaFree(device_swap_planning));
    }
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

void option_enabeler(int argc, char** argv) {
    for(int i=1; i<argc; i++) {

        if(strcmp(argv[i], "-nc") == 0) {
            DEBUG_NO_CHAINS = 1;
            continue;
        } else if(strcmp(argv[i], "-ns") == 0) {
            DEBUG_NO_SWAPS = 1;
            continue;
        } else {
	    WARPS_PER_BLOCK = atoi(argv[i]);
	}
    }
}

int getNextSwapIteration(int* swap_list_offsets, int start_iteration) {
    if(SWAP_ACTIVE) {
        int offset = swap_list_offsets[start_iteration];
        for(int i=start_iteration+1; i<N_ITERATIONS; i++) {
            if(swap_list_offsets[i] != offset) {
                return i;
            }
        }
    } 
    return N_ITERATIONS-1;
}