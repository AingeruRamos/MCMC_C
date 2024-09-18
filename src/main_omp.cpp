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
#include <sys/time.h>
#include <omp.h>

#include "../header/constants.h"
#include "../header/ising.h"
#include "../header/mcmc.h"
#include "../header/rand.h"
#include "../header/stack.h"

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

void omp_init_chains(MODEL_CHAIN* chains) {
    #pragma omp parallel
    {
        const int N_THREADS = omp_get_num_threads();
        int tid = omp_get_thread_num();

        for(int replica_id=tid; replica_id<TOTAL_REPLICAS; replica_id+=N_THREADS) {
            init_chain<MODEL_CHAIN>(chains, replica_id);
        }
    }
}

void omp_init_replicas(MODEL_NAME* replicas, int* rands, MODEL_CHAIN* chains) {
    #pragma omp parallel
    {
        const int N_THREADS = omp_get_num_threads();
        int tid = omp_get_thread_num();

        for(int replica_id=tid; replica_id<TOTAL_REPLICAS; replica_id+=N_THREADS) {
            init_replica<MODEL_NAME, MODEL_CHAIN>(replicas, rands, chains, replica_id);
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
int calcIntervalSize(int iteration);

int DEBUG_NO_SWAPS = 0;
int DEBUG_NO_CHAINS = 0;
int N_THREADS = 0;

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

    double begin_init = omp_get_wtime();

    RandGen rand_gen;

    int* rands;
    rands = (int*) malloc(TOTAL_REPLICAS*sizeof(int));

    MODEL_CHAIN* chains;
    chains = (MODEL_CHAIN*) malloc(TOTAL_REPLICAS*sizeof(MODEL_CHAIN));
    omp_init_chains(chains);

    MODEL_NAME* replicas;
    replicas = (MODEL_NAME*) malloc(TOTAL_REPLICAS*sizeof(MODEL_NAME));
    omp_init_replicas(replicas, rands, chains);

    free(rands);

    double* temps;
    temps = (double*) malloc(TOTAL_REPLICAS*sizeof(double));
    omp_init_temps(temps);

    char* swap_planning;
    if(SWAP_ACTIVE) {
        swap_planning = (char*) calloc(TOTAL_REPLICAS*N_INTERVALS, sizeof(char));
    }

    double end_init = omp_get_wtime();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    double begin_exec = omp_get_wtime();

    #pragma omp parallel
    {
        N_THREADS = omp_get_num_threads();

        MODEL_NAME* replica;
        double temp;
        int tid, iteration, interval_counter, sw_cand_index;
        
        tid = omp_get_thread_num();
        iteration = 1;
        interval_counter = 0;

        while(iteration < N_ITERATIONS) {
            int next_swap_iteration = iteration+calcIntervalSize(iteration);

            // Execute all iterations before the swap iteration
	        for(; (iteration <= next_swap_iteration) && (iteration < N_ITERATIONS); iteration++) {
	    	for(int replica_id=tid; replica_id<TOTAL_REPLICAS; replica_id+=N_THREADS) {
			    replica = &replicas[replica_id];
			    temp = temps[replica_id];
			    MCMC_iteration<MODEL_NAME>(replica, temp);
		    }
	        }

            if(iteration >= N_ITERATIONS) { break; } //* We reach the limit of iterations

            // Execute the swap iteration
            #pragma omp barrier
            sw_cand_index = (interval_counter % 2 == 0);
            sw_cand_index += 2*tid;

            for(; sw_cand_index < TOTAL_REPLICAS-1; sw_cand_index += 2*N_THREADS) {
                double swap_prob = get_swap_prob<MODEL_NAME>(sw_cand_index, sw_cand_index+1, replicas, temps);

                if(rand_gen.rand_uniform() < swap_prob) {
                    doSwap<MODEL_NAME>(sw_cand_index, sw_cand_index+1, interval_counter, swap_planning, replicas, temps);
                }
            }
            #pragma omp barrier

            interval_counter += 1;
        }
    }

    double end_exec = omp_get_wtime();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    double begin_print = omp_get_wtime();

    int i_print;
    float f_print;

    fwrite(&(i_print=0), sizeof(int), 1, fp); // OPENMP EXECUTED FLAG
    
    fwrite(&(i_print=N_THREADS), sizeof(int), 1, fp); // SIMULATION CONSTANTS
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
        for(int aux=0; aux < TOTAL_REPLICAS*N_INTERVALS; aux++) {
            fwrite(&swap_planning[aux], sizeof(char), 1, fp);
        }
    } else {
        fwrite(&(i_print=0), sizeof(int), 1, fp); //* Flag of NO printed swaps
    }

    if(!DEBUG_NO_CHAINS) { // CHAINS
        fwrite(&(i_print=1), sizeof(int), 1, fp); //* Flag of printed chains
        for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
            print_chain(&chains[replica_id], fp);
        }
    } else {
        fwrite(&(i_print=0), sizeof(int), 1, fp); //* Flag of NO printed chains
    }

    double end_print = omp_get_wtime();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    double total_init = end_init-begin_init;
    double total_exec = end_exec-begin_exec;
    double total_print = end_print-begin_print;

    fwrite(&(f_print=total_init), sizeof(float), 1, fp); // TIME
    fwrite(&(f_print=total_exec), sizeof(float), 1, fp);
    fwrite(&(f_print=total_print), sizeof(float), 1, fp);

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    fclose(fp);

    free(chains);
    free(replicas);
    free(temps);

    if(SWAP_ACTIVE) {
        free(swap_planning);
    }

    return 0;
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
        }
    }
}

int calcIntervalSize(int iteration) {
    if(SWAP_ACTIVE) {
        if(iteration > (N_ITERATIONS-SWAP_INTERVAL)) {
            return N_ITERATIONS-iteration;
        } else {
            return SWAP_INTERVAL;
        }
    }
    return N_ITERATIONS-1;
}