// Copyright Notice ===========================================================
//
// ising.h, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#ifndef _ISING_H_
#define _ISING_H_

#include "mcmc.h"

/**
 * @class SpinGlassIterationResult
 * @extends IterationResult
 * @param _energy Energy of the model
 * @param _average_spin Average spin of the model
 * @brief
 * * Instances of this class saves the data generated
 * * by one iteration in a SpinGlass
*/
class SpinGlassIterationResult: public IterationResult {
    public:
        double _energy;
        double _average_spin;

        /**
         * @name SpinGlassIterationResult
         * @remark constructor
         * @param energy Energy of the model
         * @param average_spin Average spin of the model
        */
        SpinGlassIterationResult(double energy, double average_spin);

        /**
         * @memberof IterationResult
        */
        virtual IterationResult* copy();
};

/**
 * @class SpinGlassResult
 * @extends ReplicaResult
 * @brief
 * * Instances of this class saves the data generated
 * * by multiple iterations of the MCMC in a SpinGlass
*/
class SpinGlassResult: public ReplicaResult {
    public:
        /**
         * @remark constructor
         * @note
         * The constructor of this class MUST be the constructor 
         * of the base class. In this case, ReplicaResult
        */
        using ReplicaResult::ReplicaResult;

        /**
         * @memberof ReplicaResult
        */
        virtual ReplicaResult* copy();

        /**
         * @memberof ReplicaResult
        */
        virtual void print();
};

/**
 * @class SpinGlass
 * @extends Replica
 * @param _kernel_cross (Static)
 * @param _kernel_semicross (Static)
 * @param _n_row Number of rows of the model
 * @param _n_col Number of collumns of the model
 * @param _spin_plus_percentage Percentage of spins in +1 state
 * @param _sample Array storing state of the lattice
 * @param _last_delta The last delta calculated
 * @brief
 * * Instances of this class represents a SpinGlass lattice
*/
class SpinGlass: public Replica {
    public:
        static int _kernel_cross[];
        static int _kernel_semicross[];

        int* _sample;

        double _last_delta;
        
        /**
         * @name init
         * @param n_row Number of rows in the model
         * @param n_col Number of collumns in the model
         * @param spin_plus_percentage Percentage of spins in +1 state
         * @brief
         * * Initializes the SpinGlass
        */
        void init(int n_row, int n_col, double spin_plus_percentage);

        /**
         * @memberof Replica
        */
        virtual void* trial();

        /**
         * @memberof Replica
        */
        virtual double eval();

        /**
         * @memberof Replica
        */
        virtual double delta(void* trial);

        /**
         * @memberof Replica
        */
        virtual void move(void* trial);

        /**
         * @memberof Replica
        */
        virtual void save(void* trial);
};

/**
 * @name is_index_in_matrix
 * @param mat Matrix to use
 * @param n_row Number of rows of the matrix
 * @param n_col Number of collumns of the matrix
 * @param row Row of the index to check
 * @param col Collumn of the index to check
 * @brief
 * * Return 1 if index is inside the matrix, else 0
*/
int is_index_in_matrix(int* mat, int n_row, int n_col, int row, int col);

/**
 * @name single_convolve
 * @param mat Matrix to use
 * @param n_row Number of rows of the matrix
 * @param n_col Number of collumns of the matrix
 * @param index Index of the matrix where do the convolution
 * @param kernel Kernel of the convolution
 * @param kernel_size Size of the kernel
 * @return The value of the convolution
 * @brief
 * * Applies the kernel in the specified position of the matrix
 * @note
 * The kernel size is assumed to be (kernel_size x kernel_size). 
 * Those values that, when applying the kernel, are outside the matrix, it will assumed the value 0 
*/
int single_convolve(int* mat, int n_row, int n_col, int index, int* kernel, int kernel_size);

/**
 * @name convolve
 * @param mat Matrix to use
 * @param n_row Number of rows of the matrix
 * @param n_col Number og collumns of the matrix
 * @param kernel Kernel of the convolution
 * @param kernel_size Size of the kernel
 * @return The matrix result of the convolution
 * @brief
 * * Calculates the convolution of the matrix
 * @note
 * The kernel size is assumed to be (kernel_size x kernel_size)
 * Those values that, when applying the kernel, are outside the matrix, it will assumed the value 0
*/
int* convolve(int* mat, int n_row, int n_col, int* kernel, int kernel_size);

/**
 * @name array_sum
 * @param arr Array to use
 * @param length Size of the array
 * @return Sum of all elements of the array
 * @brief
 * * Calculates the sum of the array
*/
int array_sum(int* arr, int length);

/**
 * @name array_sum
 * @overload
*/
double array_sum(double* arr, int length);

#endif