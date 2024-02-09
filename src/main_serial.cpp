// Copyright Notice ===========================================================
//
// main_serial.cpp, Copyright (c) 2023 Aingeru Ramos
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

#include "../header/constants.h"
#include "../header/ising.h"
#include "../header/mcmc.h"
#include "../header/rand.h"
#include "../header/stack.h"

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

void init_chains(MODEL_CHAIN* chains) {
    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
        init_chain<MODEL_CHAIN>(chains, replica_id);
    }
}

void init_replicas(MODEL_NAME* replicas, int* rands, MODEL_CHAIN* chains) {
    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
        init_replica<MODEL_NAME, MODEL_CHAIN>(replicas, rands, chains, replica_id);
    }
}

void init_temps(double* temps) {
    for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
        init_temp(temps, replica_id);
    }
}

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

void option_enabeler(int argc, char** argv);
double getElapsedTime(struct timeval* begin_time, struct timeval* end_time);

int DEBUG_NO_SWAPS = 0;
int DEBUG_NO_CHAINS = 0;

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

    RandGen rand_gen;

    int* rands;
    rands = (int*) malloc(TOTAL_REPLICAS*sizeof(int));

    MODEL_CHAIN* chains;
    chains = (MODEL_CHAIN*) malloc(TOTAL_REPLICAS*sizeof(MODEL_CHAIN));
    init_chains(chains);

    MODEL_NAME* replicas;
    replicas = (MODEL_NAME*) malloc(TOTAL_REPLICAS*sizeof(MODEL_NAME));
    init_replicas(replicas, rands, chains);

    free(rands);

    double* temps;
    temps = (double*) malloc(TOTAL_REPLICAS*sizeof(double));
    init_temps(temps);

    int* swap_list_offsets;
    Swap* swap_planning;

    if(SWAP_ACTIVE) {
        swap_list_offsets = (int*) malloc((N_ITERATIONS+1)*sizeof(int));
        init_swap_list_offsets(swap_list_offsets);

        swap_planning = (Swap*) malloc(swap_list_offsets[N_ITERATIONS]*sizeof(Swap));
        init_swap_planning(swap_list_offsets, swap_planning);
    }

    clock_t end_init = clock();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    clock_t begin_exec = clock();

    for(int iteration=1; iteration<N_ITERATIONS; iteration++) {

        for(int replica_id=0; replica_id<TOTAL_REPLICAS; replica_id++) {
            MCMC_iteration<MODEL_NAME>(&replicas[replica_id], temps[replica_id]);
        }

        if(SWAP_ACTIVE) {

            int offset = swap_list_offsets[iteration-1];
            int n_swaps = swap_list_offsets[iteration]-offset;

            for(int swap_index=0; swap_index<n_swaps; swap_index++) {
                Swap* swap = &swap_planning[offset+swap_index];
                double swap_prob = get_swap_prob<MODEL_NAME>(swap, replicas, temps);

                if(rand_gen.rand_uniform() < swap_prob) {
                    doSwap<MODEL_NAME>(temps, replicas, swap);
                }
            }
        }
    }

    clock_t end_exec = clock();

//-----------------------------------------------------------------------------|
//-----------------------------------------------------------------------------|

    clock_t begin_print = clock();

    int i_print;
    float f_print;

    fwrite(&(i_print=2), sizeof(int), 1, fp); // SERIAL EXECUTED FLAG
    
    fwrite(&(i_print=1), sizeof(int), 1, fp); // SIMULATION CONSTANTS
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
        for(int iteration=0; iteration<N_ITERATIONS; iteration++) {
            int offset = swap_list_offsets[iteration];
            int n_swaps = swap_list_offsets[iteration+1]-offset;
            print_swap_list(swap_planning, offset, n_swaps, fp);
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

    free(chains);
    free(replicas);
    free(temps);

    if(SWAP_ACTIVE) {
        free(swap_list_offsets);
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

double getElapsedTime(struct timeval* begin_time, struct timeval* end_time) {
    long secs = end_time->tv_sec - begin_time->tv_sec;
    long u_secs = end_time->tv_usec - begin_time->tv_usec;

    return secs + u_secs*1e-6;
}