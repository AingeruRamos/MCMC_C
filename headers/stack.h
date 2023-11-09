#ifndef _STACK_H_
#define _STACK_H_

#include "constants.h"

template <typename T, int N>
class Stack {
    private:
        int n_elem;
        T stack[N];

    public:
        _CUDA_DECOR_ Stack() {
            n_elem = 0;
        }

        _CUDA_DECOR_ void push(T elem) {
            stack[n_elem] = elem;
            n_elem += 1;
        }

        _CUDA_DECOR_ T top() {
            return stack[n_elem-1];
        }

        _CUDA_DECOR_ T get(int index) {
            return stack[index];
        }

        _CUDA_DECOR_ void set(T elem, int index) {
            stack[index] = elem;
        }

        _CUDA_DECOR_ Stack<T, N>* copy() {
            Stack<T, N>* stack_pointer = new Stack<T, N>();
            for(int i=0; i<N; i++) {
                stack_pointer->push(stack[i]->copy());
            }
            return stack_pointer;
        }
};

#endif