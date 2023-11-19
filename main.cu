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

#include "./headers/constants.h"
#include "./headers/ising.h"
#include "./headers/mcmc.h"

#define _CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void init_stuff(curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1337, idx, 0, &state[idx]);
}

__device__ float make_rand(curandState* state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    return curand_uniform(&state[idx]);
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

__device__ void* trial_f() {
    int* arr = (int*) malloc(2*sizeof(int));
    arr[0] = (int) (make_rand()*N_ROW);
    arr[1] = (int) (make_rand()*N_COL);
    return arr;
}

__device__ double delta_f(SpinGlass* sp, void* trial) {
    int* trial_int = (int*) trial;
    int index = trial_int[0]*N_ROW+trial_int[1];
    int sum = apply_kernel(sp->_sample, N_ROW, N_COL, index, sp->_kernel_cross, 3); 
    int si = sp->_sample[trial_int[0]*N_ROW+trial_int[1]];
    sp->_last_delta = 2.0*si*sum;
    return sp->_last_delta;
}

__device__ void move_f(SpinGlass* sp, void* trial) {
    int* trial_int = (int*) trial;
    int index = trial_int[0]*N_ROW+trial_int[1];
    sp->_last_spin = sp->_sample[index];
    sp->_sample[index] *= -1;
}

__device__ void save_f(SpinGlass* sp, void* trial) {
    SpinGlassIterationResult* sp_last_it = (SpinGlassIterationResult*) sp->_results.top();
    SpinGlassIterationResult* sp_it = (SpinGlassIterationResult*) sp_last_it->copy();

    if(trial != nullptr) { //* If trial has been accepted
        sp_it->_energy += sp->_last_delta;
    }

    sp_it->_average_spin -= 2*sp->_last_spin;

    sp->_results.push(sp_it);
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

__global__ void cuda_alloc_swaps(Swap*** device_swap_planning, int* device_n_swaps) {
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

__global__ void cuda_MCMC(SpinGlass* device_models, double* device_temp) {

    int replica_id = 0; //* <<<<<<

    SpinGlass* model = &device_models[replica_id];
    double temp = device_temp[replica_id];

    for(int i=1; i<N_ITERATIONS; i++) {

        // Get a trial
        void* trial = trial_f();

        // Calculate the acceptance probability
        double delta_energy = delta_f(model, trial);

        double acc_p = 0;
        if(delta_energy <= 0) { acc_p = 1.0; }
        else { acc_p = exp(-delta_energy/temp); }

        // Change state
        double ranf =  make_rand();
        if(ranf < acc_p) { move_f(model, trial); } //* Trial is accepted
        else {
            free(trial);
            trial = nullptr; //* Used as flag of rejected move
        }

        // Save actual state of the model
        save_f(model, trial);
        free(trial);
    }
}

void option_enabeler(int argc, char** argv);

int DEBUG_FLOW = 0;
int DEBUG_RESULTS = 0;
int N_THREADS = 1;

int main(int argc, char** argv) {

    option_enabeler(argc, argv);

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    if(DEBUG_FLOW) { printf("Host -> Initialazing\n"); }

    SpinGlass* host_models = (SpinGlass*) malloc(TOTAL_REPLICAS*sizeof(SpinGlass));

    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
        host_models[replica_id].init();
    }

    if(DEBUG_FLOW) { printf("Host -> Replicas: OK\n"); }

    double* host_temps = (double*) malloc(TOTAL_REPLICAS*sizeof(double));

    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
        host_temps[replica_id] = INIT_TEMP+(replica_id*TEMP_STEP);
    }

    if(DEBUG_FLOW) { printf("Host -> Temps: OK\n"); }

    int* host_n_swaps = (int*) calloc(N_ITERATIONS, sizeof(int));

    if(TOTAL_REPLICAS > 1) {
        for(int i=0; i<N_ITERATIONS; i++) {
            host_n_swaps[i] = (int) (TOTAL_REPLICAS/2);
            if((i%2 != 0) && (TOTAL_REPLICAS%2 == 0)) { //* Number of swaps in odd iterations
                host_n_swaps[i] -= 1;
            }
        }
    }

    Swap*** host_swap_planning = (Swap***) malloc(N_ITERATIONS*sizeof(Swap**));

    for(int i=0; i<N_ITERATIONS; i++) {
        host_swap_planning[i] = (Swap**) malloc(host_n_swaps[i]*sizeof(Swap*));

        int sw_cand_1 = 0; //* Defining the starting point
        if(i%2 != 0) { sw_cand_1 = 1; }

        for(int j=0; j<host_n_swaps[i]; j++) {
            host_swap_planning[i][j] = new Swap(sw_cand_1, (sw_cand_1+1));
            sw_cand_1 += 2;
        }
    }

    if(DEBUG_FLOW) { printf("Host -> Swaps: OK\n"); }

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    if(DEBUG_FLOW) { printf("Device -> Initialazing\n"); }

    SpinGlass* device_models;
    _CUDA(cudaMalloc((void**)&device_models, TOTAL_REPLICAS*sizeof(SpinGlass)));
    _CUDA(cudaMemcpy(device_models, host_models, TOTAL_REPLICAS*sizeof(SpinGlass), cudaMemcpyHostToDevice));

    if(DEBUG_FLOW) { printf("Device -> Replicas: OK\n"); }

    double* device_temps;
    _CUDA(cudaMalloc((void**)&device_temps, TOTAL_REPLICAS*sizeof(double)));
    _CUDA(cudaMemcpy(device_temps, host_temps, TOTAL_REPLICAS*sizeof(double), cudaMemcpyHostToDevice));

    if(DEBUG_FLOW) { printf("Device -> Temps: OK\n"); }

    int* device_n_swaps;
    _CUDA(cudaMalloc((void**)&device_n_swaps, N_ITERATIONS*sizeof(int)));
    _CUDA(cudaMemcpy(device_n_swaps, host_n_swaps, N_ITERATIONS*sizeof(int), cudaMemcpyHostToDevice));

    Swap*** device_swap_planning;
    _CUDA(cudaMalloc((void**)&device_swap_planning, N_ITERATIONS*sizeof(Swap**)));
    cuda_alloc_swaps<<<1,1>>>(device_swap_planning, device_n_swaps);

    if(DEBUG_FLOW) { printf("Device -> Swaps: OK\n"); }

    curandState* device_state;
    cudaMalloc((void**)&device_state,, TOTAL_REPLICAS);
    init_stuff<<<1, TOTAL_REPLICAS>>>(device_state);

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    //cuda_MCMC<<<1,1>>>(device_models, device_temps);

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

// OBTENER LOS RESULTADOS EN EL HOST
// Y REORDENARLOS

    /*
    // Pasar lo del device al host
    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
        _CUDA(cudaMemcpy(&host_models[replica_id], &device_models[replica_id], sizeof(SpinGlass), cudaMemcpyDeviceToHost));
    }
    */

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

// LIBERAR TODA LA MEMORIA (LA DEL HOST FALTA (EN OPENMP TAMBIEN))

    _CUDA(cudaFree(device_models));
    _CUDA(cudaFree(device_temps));
    _CUDA(cudaFree(device_n_swaps));
    _CUDA(cudaFree(device_swap_planning));
    _CUDA(cudaFree(device_state));

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
