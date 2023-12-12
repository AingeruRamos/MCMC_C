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

void option_enabeler(int argc, char** argv);
void print_stack(MODEL_RESULTS* stack);

int DEBUG_FLOW = 0;
int N_THREADS = 1;

int main(int argc, char** argv) {

    srand(time(NULL));

    option_enabeler(argc, argv);

    if(SWAP_ACTIVE && (TOTAL_REPLICAS <= 1)) {
        printf("ERROR: This program is not valid.\n");
        return -1;
    }

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    time_t begin_all = omp_get_wtime();

    RandGen rand_gen;

    int* rands;
    rands = (int*) malloc(TOTAL_REPLICAS*sizeof(int));

    if(DEBUG_FLOW) { printf("Rands: OK\n"); }

    MODEL_RESULTS* results;
    results = (MODEL_RESULTS*) malloc(TOTAL_REPLICAS*sizeof(MODEL_RESULTS));
    initialize_results<MODEL_RESULTS>(results);

    if(DEBUG_FLOW) { printf("Results: OK\n"); }

    MODEL_NAME* replicas;
    replicas = (MODEL_NAME*) malloc(TOTAL_REPLICAS*sizeof(MODEL_NAME));
    initialize_replicas<MODEL_NAME, MODEL_RESULTS>(replicas, rands, results);

    free(rands);

    if(DEBUG_FLOW) { printf("Replicas: OK\n"); }

    double* temps;
    temps = (double*) malloc(TOTAL_REPLICAS*sizeof(double));
    initialize_temps(temps);

    if(DEBUG_FLOW) { printf("Temps: OK\n"); }

    int* n_swaps;
    Swap*** swap_planning;

    if(SWAP_ACTIVE) {
        n_swaps = (int*) calloc(N_ITERATIONS, sizeof(int));
        swap_planning = (Swap***) malloc(N_ITERATIONS*sizeof(Swap**));

        initialize_swaps(n_swaps, swap_planning);
    }

    if(DEBUG_FLOW) { printf("Swaps: OK\n"); }

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
                        double aux_temp = temps[swap->_swap_candidate_1];
                        temps[swap->_swap_candidate_1] = temps[swap->_swap_candidate_2];
                        temps[swap->_swap_candidate_2] = aux_temp;

                        MODEL_RESULTS* aux_results = replicas[swap->_swap_candidate_1]._results;
                        replicas[swap->_swap_candidate_1]._results = replicas[swap->_swap_candidate_2]._results;
                        replicas[swap->_swap_candidate_2]._results = aux_results;
                        swap->_accepted = true;
                    }
                }

                #pragma omp barrier
            }
        }
    }

    time_t end_exec = omp_get_wtime();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    time_t end_all = omp_get_wtime();

    double total_all = (double) (end_all-begin_all);
    double total_exec = (double) (end_exec-begin_exec);

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    printf("%d\n", N_THREADS); // SIMULATION CONSTANTS
    printf("%d\n", N_ITERATIONS);
    printf("%d\n", SWAP_ACTIVE);
    printf("%f,%f,%f,%d\n", INIT_TEMP, END_TEMP, TEMP_STEP, TOTAL_REPLICAS);

    printf("#\n"); // MODEL CONSTANTS
    
    printf("%d\n", N_ROW);
    printf("%d\n", N_COL);
    printf("%f\n", SPIN_PLUS_PERCENTAGE);

    printf("#\n");

    if(SWAP_ACTIVE) { // SWAP PLANNING (ACCEPTED)
        print_swaps(n_swaps, swap_planning);
    }

    printf("#\n"); // TIME

    printf("%f\n", total_all);
    printf("%f\n", total_exec);

    printf("#\n"); // RESULTS

    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
        print_stack(replicas[replica_id]._results);
        printf("#\n");
    }

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

        char* option_string = argv[1];
        if(strcmp(argv[i], "-df") == 0) {
            DEBUG_FLOW = 1;
            continue;
        }
    }
}

void print_stack(MODEL_RESULTS* stack) {
    for(int i=0; i<N_ITERATIONS; i++) {
        MODEL_ITER* sp_it = (MODEL_ITER*) stack->get(i);
        printf("%f,", sp_it->_energy);
    }
    printf("\n");
    for(int i=0; i<N_ITERATIONS; i++) {
        MODEL_ITER* sp_it = (MODEL_ITER*) stack->get(i);
        printf("%d,", sp_it->_average_spin);
    }
    printf("\n");
}