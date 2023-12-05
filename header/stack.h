// Copyright Notice ===========================================================
//
// stack.h, Copyright (c) 2023 Aingeru Ramos
//
// All Rights Reserved ========================================================
//
// This file is part of MCMC_C software project.
//
// MCMC_C is propietary software. The author has all the rights to the work.
// No third party may make use of this work without explicit permission of the author.
//
// ============================================================================

#ifndef _STACK_H_
#define _STACK_H_

#include "constants.h"

/**
 * @class Stack
 * @remark template
 * @param n_elem Number of items inside the stack
 * @param stack Array that constains the elements
 * @brief
 * * Instances of this class represents a stack
 * @note
 * This class is a template. T is the type of items
 * in the stack, and N the maximum number of items
 * in the stack
*/
template <typename T, int N>
class Stack {
    private:
        int n_item;
        T stack[N];

    public:

        /**
         * @name Stack
         * @remark constructor
        */
        _HOST_ _DEVICE_ Stack() {
            n_item = 0;
        }

        /**
         * @name push
         * @param elem Item to push in the stack
         * @brief
         * * Push the item into the stack
        */
        _HOST_ _DEVICE_ void push(T item) {
            stack[n_item] = item;
            n_item += 1;
        }

        /**
         * @name top
         * @return A item
         * @brief
         * * Returns the last item of the stack
        */
        _HOST_ _DEVICE_ T top() {
            return stack[n_item-1];
        }

        /**
         * @name get
         * @param index Index of the item
         * @return A item
         * @brief
         * * Gets the item in the indicated index
        */
        _HOST_ _DEVICE_ T get(int index) {
            return stack[index];
        }

        /**
         * @name set
         * @param item The item to insert
         * @param index Index where insert
         * @brief
         * * Insert the item in the indicated index
        */
        _HOST_ _DEVICE_ void set(T item, int index) {
            stack[index] = item;
        }

        /**
         * @name copy
         * @return A copy of this stack
         * * Copies this stack
        */
        _HOST_ _DEVICE_ Stack<T, N>* copy() {
            Stack<T, N>* stack_pointer = new Stack<T, N>();
            for(int i=0; i<N; i++) {
                stack_pointer->push(stack[i]->copy());
            }
            return stack_pointer;
        }
};

#endif