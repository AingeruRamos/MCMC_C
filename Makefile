# Old Used Flags
#	-fpermissive

COMM_FLAGS=-std=c++11 -g
CPP_FILES=ising.cpp tools.cpp

main:
	g++ main.cpp $(CPP_FILES) -o main -fopenmp $(COMM_FLAGS)

main_x:
	g++ main_x.cpp $(CPP_FILES) -o main_x -fopenmp $(COMM_FLAGS)

main_mpi:
	mpicc main_mpi.cpp $(CPP_FILES) -o main_mpi $(COMM_FLAGS)

prueba:
	g++ prueba.cpp $(CPP_FILES) -o prueba -fopenmp $(COMM_FLAGS)

# NVIDIA COMPILATIONS

main_cu:
	nvcc main.cu $(CPP_FILES) -o main_cu

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