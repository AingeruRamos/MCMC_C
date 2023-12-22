// Copyright Notice ===========================================================
//
// main.cpp, Copyright (c) 2023 Aingeru Ramos
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
#include <string.h>
#include <time.h>
#include <omp.h>

#include "../header/constants.h"
#include "../header/ising.h"
#include "../header/mcmc.h"
#include "../header/rand.h"
#include "../header/stack.h"

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

void omp_init_results(MODEL_RESULTS* results) {
    #pragma omp parallel
    {
        const int N_THREADS = omp_get_num_threads();
        int tid = omp_get_thread_num();

        for(int replica_id=tid; replica_id<TOTAL_REPLICAS; replica_id+=N_THREADS) {
            init_result<MODEL_RESULTS>(results, replica_id);
        }
    }
}

void omp_init_replicas(MODEL_NAME* replicas, int* rands, MODEL_RESULTS* results) {
    #pragma omp parallel
    {
        const int N_THREADS = omp_get_num_threads();
        int tid = omp_get_thread_num();

        for(int replica_id=tid; replica_id<TOTAL_REPLICAS; replica_id+=N_THREADS) {
            init_replica<MODEL_NAME, MODEL_RESULTS>(replicas, rands, results, replica_id);
        }
    }
}

void omp_init_temps(double* temps) {
    #pragma omp parallel
    {
        const int N_THREADS = omp_get_num_threads();
        int tid = omp_get_thread_num();

        for(int replica_id=tid; replica_id<TOTAL_REPLICAS; replica_id+=N_THREADS) {
            init_temp(temps, replica_id);
        }
    }
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

void option_enabeler(int argc, char** argv);

int DEBUG_FLOW = 0;
int DEBUG_NO_SWAPS = 0;
int DEBUG_NO_RESULTS = 0;
int N_THREADS = 0;

int main(int argc, char** argv) {

    srand(time(NULL));

    option_enabeler(argc, argv);

    if(SWAP_ACTIVE && (TOTAL_REPLICAS <= 1)) {
        printf("ERROR: This program is not valid.\n");
        return -1;
    }

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    time_t begin_init = omp_get_wtime();

    RandGen rand_gen;

    int* rands;
    rands = (int*) malloc(TOTAL_REPLICAS*sizeof(int));

    if(DEBUG_FLOW) { printf("Rands: OK\n"); }

    MODEL_RESULTS* results;
    results = (MODEL_RESULTS*) malloc(TOTAL_REPLICAS*sizeof(MODEL_RESULTS));
    omp_init_results(results);

    if(DEBUG_FLOW) { printf("Results: OK\n"); }

    MODEL_NAME* replicas;
    replicas = (MODEL_NAME*) malloc(TOTAL_REPLICAS*sizeof(MODEL_NAME));
    omp_init_replicas(replicas, rands, results);

    free(rands);

    if(DEBUG_FLOW) { printf("Replicas: OK\n"); }

    double* temps;
    temps = (double*) malloc(TOTAL_REPLICAS*sizeof(double));
    omp_init_temps(temps);

    if(DEBUG_FLOW) { printf("Temps: OK\n"); }

    int* n_swaps;
    Swap*** swap_planning;

    if(SWAP_ACTIVE) {
        n_swaps = (int*) calloc(N_ITERATIONS, sizeof(int));
        swap_planning = (Swap***) malloc(N_ITERATIONS*sizeof(Swap**));

        init_swaps(n_swaps, swap_planning);
    }

    if(DEBUG_FLOW) { printf("Swaps: OK\n"); }

    time_t end_init = omp_get_wtime();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    time_t begin_exec = omp_get_wtime();

    if(DEBUG_FLOW) { printf("Executing\n"); }

    #pragma omp parallel
    {
        N_THREADS = omp_get_num_threads();

        int tid = omp_get_thread_num();

        for(int iteration=1; iteration<N_ITERATIONS; iteration++) {

            for(int replica_id=tid; replica_id<TOTAL_REPLICAS; replica_id+=N_THREADS) {
                MCMC_iteration<MODEL_NAME>(&replicas[replica_id], temps[replica_id]);
                if(DEBUG_FLOW) { printf("Replica (%d): OK\n", replica_id); }
            }

            #pragma omp master
            {
                if(DEBUG_FLOW) { printf("Replicas: OK\n"); }
            }

            if(SWAP_ACTIVE) {
                #pragma omp barrier

                for(int swap_index=tid; swap_index<n_swaps[iteration-1]; swap_index+=N_THREADS) {
                    Swap* swap = swap_planning[iteration-1][swap_index];
                    double swap_prob = get_swap_prob<MODEL_NAME>(swap, replicas, temps);

                    if(DEBUG_FLOW) { printf("Swap pre-calculus (%d): OK\n", swap_index); }

                    if(rand_gen.rand_uniform() < swap_prob) {
                        doSwap<MODEL_NAME>(temps, replicas, swap);
                    }
                }

                #pragma omp barrier
            }
        }
    }

    time_t end_exec = omp_get_wtime();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    time_t begin_print = omp_get_wtime();

    int i_print;
    float f_print;

    fwrite(&(i_print=0), sizeof(int), 1, stdout); // OPENMP EXECUTED FLAG
    
    fwrite(&(i_print=N_THREADS), sizeof(int), 1, stdout); // SIMULATION CONSTANTS
    fwrite(&(i_print=N_ITERATIONS), sizeof(int), 1, stdout);
    fwrite(&(i_print=SWAP_ACTIVE), sizeof(int), 1, stdout);

    fwrite(&(f_print=INIT_TEMP), sizeof(float), 1, stdout);
    fwrite(&(f_print=END_TEMP), sizeof(float), 1, stdout);
    fwrite(&(f_print=TEMP_STEP), sizeof(float), 1, stdout);
    fwrite(&(i_print=TOTAL_REPLICAS), sizeof(int), 1, stdout);
    
    fwrite(&(i_print=N_ROW), sizeof(int), 1, stdout); // MODEL CONSTANTS
    fwrite(&(i_print=N_COL), sizeof(int), 1, stdout);
    fwrite(&(f_print=SPIN_PLUS_PERCENTAGE), sizeof(float), 1, stdout);

    if(!DEBUG_NO_SWAPS && SWAP_ACTIVE) { // SWAP PLANNING (ACCEPTED)
        fwrite(&(i_print=1), sizeof(int), 1, stdout); //* Flag of printed swaps
        print_swaps(n_swaps, swap_planning);
    } else {
        fwrite(&(i_print=0), sizeof(int), 1, stdout); //* Flag of NO printed swaps
    }

    if(!DEBUG_NO_RESULTS) { // RESULTS
        fwrite(&(i_print=1), sizeof(int), 1, stdout); //* Flag of printed results
        for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
            print_result(&results[replica_id]);
        }
    } else {
        fwrite(&(i_print=0), sizeof(int), 1, stdout); //* Flag of NO printed results
    }

    time_t end_print = omp_get_wtime();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    double total_init = (double) (end_init-begin_init);
    double total_exec = (double) (end_exec-begin_exec);
    double total_print = (double) (end_print-begin_print);

    fwrite(&(f_print=total_init), sizeof(float), 1, stdout); // TIME
    fwrite(&(f_print=total_exec), sizeof(float), 1, stdout);
    fwrite(&(f_print=total_print), sizeof(float), 1, stdout);

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    free(results);
    free(replicas);
    free(temps);

    if(SWAP_ACTIVE) {
        free(n_swaps);
        free(swap_planning); //TODO Improve "swap_planning" memory free
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
        } else if(strcmp(argv[i], "-nr") == 0) {
            DEBUG_NO_RESULTS = 1;
            continue;
        } else if(strcmp(argv[i], "-ns") == 0) {
            DEBUG_NO_SWAPS = 1;
            continue;
        }
    }
}