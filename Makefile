# Old Used Flags
#	-fpermissive

COMM_FLAGS=-std=c++11 -g

main:
	g++ main.cpp mcmc.cpp ising.cpp tools.cpp -o main -fopenmp $(COMM_FLAGS)

main_x:
	g++ main_x.cpp mcmc.cpp ising.cpp tools.cpp -o main_x -fopenmp $(COMM_FLAGS)

main_mpi:
	mpicc main_mpi.cpp mcmc.cpp ising.cpp tools.cpp -o main_mpi $(COMM_FLAGS)

prueba:
	g++ prueba.cpp mcmc.cpp ising.cpp tools.cpp -o prueba -fopenmp $(COMM_FLAGS)

# NVIDIA COMPILATIONS

main_cu:
	nvcc main.cu -o main_cu

compile_main_cu: # <--
	priscilla exec make main_cu

main_cu_x:
	nvcc main_x.cu -o main_cu_x

compile_main_cu_x: # <--
	priscilla exec make main_cu_x

prueba_cu:
	nvcc prueba.cu -o prueba_cu

compile_prueba_cu: # <--
	priscilla exec make prueba_cu

#######################