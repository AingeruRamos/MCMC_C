# MCMC_C

This project executes the Metropolis-Hastling algorithm.

## Instalation and Execution

### Instalation

Once downloaded the source code, it can be compiled using the Makefile provided in the project folder. To compile type the next in the terminal: ```make main```

### Execution
This code is intended to be executed with OpenMP. To assign the number of threads you want to use, you must introduce in terminal the next: ```export OMP_NUM_THREADS=<num_threads>```

It will appear a executable file in the project folder with name ```main```

## Usage

In order to generalize the implementation of the algorithm to any given model, the user must implement 2 classes that encapsulates the behaviour of the model. Examples of this classes are provided in 'headers/model_template.h'. The program wouldnt compile if this classes are not implemented.

**IMPORTANT NOTE: Not forget to change the ```MODEL_NAME``` and ```MODEL_ITER``` define statements to your class names.**

Once completed this implementations, the program can be compiled and executed.

## Credits

Below, the list of all the participants who have contributed to the project:

- Aingeru Ramos <<aingeru.ramos@ehu.eus>>: (Original Author)

## License

Copyright (c) 2023 Aingeru Ramos - All Rights Reserved

MCMC_C is propietary software. The author has all the rights to the work.

No third party may make use of this work without explicit permission of the author.