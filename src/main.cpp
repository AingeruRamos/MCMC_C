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
void print_stack(Stack<MODEL_ITER*, N_ITERATIONS>* stack);

int DEBUG_FLOW = 0;
int DEBUG_RESULTS = 0;
int N_THREADS = 1;

int main(int argc, char** argv) {

    time_t begin_all = omp_get_wtime();

    option_enabeler(argc, argv);

    srand(time(NULL));
    RandGen rand_gen;

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    if(DEBUG_FLOW) { printf("Initialazing\n"); }

    MODEL_NAME* models = (MODEL_NAME*) malloc(TOTAL_REPLICAS*sizeof(MODEL_NAME));
    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
        MODEL_NAME* sp = &models[replica_id];
        sp->_rand_gen.set_state(rand());
        sp->init();
    }

    if(DEBUG_FLOW) { printf("Replicas: OK\n"); }

    double* temps = (double*) malloc(TOTAL_REPLICAS*sizeof(double));

    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
        temps[replica_id] = INIT_TEMP+(replica_id*TEMP_STEP);
    }

    if(DEBUG_FLOW) { printf("Temps: OK\n"); }

    int* n_swaps = (int*) calloc(N_ITERATIONS, sizeof(int));

    if(TOTAL_REPLICAS > 1) {
        for(int i=0; i<N_ITERATIONS; i++) {
            n_swaps[i] = (int) (TOTAL_REPLICAS/2);
            if((i%2 != 0) && (TOTAL_REPLICAS%2 == 0)) { //* Number of swaps in odd iterations
                n_swaps[i] -= 1;
            }
        }
    }

    Swap*** swap_planning = (Swap***) malloc(N_ITERATIONS*sizeof(Swap**));

    for(int i=0; i<N_ITERATIONS; i++) {
        swap_planning[i] = (Swap**) malloc(n_swaps[i]*sizeof(Swap*));

        int sw_cand_1 = 0; //* Defining the starting point
        if(i%2 != 0) { sw_cand_1 = 1; }

        for(int j=0; j<n_swaps[i]; j++) {
            swap_planning[i][j] = new Swap(sw_cand_1, (sw_cand_1+1));
            sw_cand_1 += 2;
        }
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
                MCMC_iteration<MODEL_NAME>(&models[replica_id], temps[replica_id]);
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
                    double swap_prob = get_swap_prob<MODEL_NAME>(swap, models, temps);

                    if(DEBUG_FLOW) { printf("Swap pre-calculus (%d): OK\n", swap_index); }

                    if(rand_gen.rand_uniform() < swap_prob) {
                        double aux = temps[swap->_swap_candidate_1];
                        temps[swap->_swap_candidate_1] = temps[swap->_swap_candidate_2];
                        temps[swap->_swap_candidate_2] = aux;
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

    if(DEBUG_FLOW) { printf("Reordering\n"); }

    Stack<MODEL_ITER*, N_ITERATIONS>** results_copy = (Stack<MODEL_ITER*, N_ITERATIONS>**)
                            malloc(TOTAL_REPLICAS*sizeof(Stack<MODEL_ITER*, N_ITERATIONS>*));

    if(SWAP_ACTIVE) {

        for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
            results_copy[replica_id] = models[replica_id]._results.copy();
        }

        int* swap_replica_ids = (int*) malloc(TOTAL_REPLICAS*sizeof(int));
        for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
            swap_replica_ids[replica_id] = replica_id;
        }

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();

            for(int iteration=0; iteration<N_ITERATIONS; iteration++) {

                //? Search a way to do in parallel
                for(int replica_id=tid; replica_id<TOTAL_REPLICAS; replica_id+=N_THREADS) {
                    int swap_replica_id = swap_replica_ids[replica_id];
                    models[replica_id]._results.set(results_copy[swap_replica_id]->get(iteration), iteration);
                }

                #pragma omp barrier

                if(iteration != 0) {
                    for(int swap_index=tid; swap_index<n_swaps[iteration-1]; swap_index+=N_THREADS) {
                        Swap* sw = swap_planning[iteration-1][swap_index];
                        if(sw->_accepted) {
                            int aux = swap_replica_ids[sw->_swap_candidate_1];
                            swap_replica_ids[sw->_swap_candidate_1] = swap_replica_ids[sw->_swap_candidate_2];
                            swap_replica_ids[sw->_swap_candidate_2] = aux;
                        }
                    }
                }

                #pragma omp barrier
            }
        }
    }

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
    /*
    for(int i=0; i<N_ITERATIONS; i++) { // SWAP PLANNING (ACCEPTED)
        for(int j=0; j<n_swaps[i]; j++) {
            Swap* sw = swap_planning[i][j];
            if(sw->_accepted) {
                printf("%d-%d,", sw->_swap_candidate_1, sw->_swap_candidate_2);
            }
        }
        printf("\n");
    }
    */
    printf("#\n"); // TIME

    printf("%f\n", total_all);
    printf("%f\n", total_exec);

    printf("#\n"); // RESULTS
    /*
    if(SWAP_ACTIVE && DEBUG_RESULTS) {
        for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
            print_stack(results_copy[replica_id]);
            printf("#\n");
        }
    }
    */
    //printf("#\n");

    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
        print_stack(&models[replica_id]._results);
        printf("#\n");
    }

    return 0;
}

void option_enabeler(int argc, char** argv) {
    for(int i=1; i<argc; i++) {

        char* option_string = argv[1];
        if(strcmp(argv[i], "-df") == 0) {
            DEBUG_FLOW = 1;
            continue;
        } else if(strcmp(argv[i], "-dr") == 0) {
            DEBUG_RESULTS = 1;
            continue;
        } 
    }
}

void print_stack(Stack<MODEL_ITER*, N_ITERATIONS>* stack) {
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