COMM_FLAGS=-std=c++11
CPP_FILES=src/ising.cpp src/rand.cpp

DEFINES ?=

main:
	g++ src/main_serial.cpp $(CPP_FILES) -o main_serial $(COMM_FLAGS) $(DEFINES)

prueba_serial:
	g++ prueba_serial.cpp $(CPP_FILES) -o prueba_serial $(COMM_FLAGS)

# OPENMP COMPILATIONS

main_omp:
	g++ src/main_omp.cpp $(CPP_FILES) -o main_omp -fopenmp $(COMM_FLAGS) $(DEFINES)

prueba_omp:
	g++ prueba_omp.cpp $(CPP_FILES) -o prueba_omp -fopenmp $(COMM_FLAGS)

# NVIDIA COMPILATIONS

main_cu:
	priscilla exec nvcc -x cu src/main_cuda.cu $(CPP_FILES) -lcurand -o main_cu -rdc=true $(DEFINES)

prueba_cu:
	priscilla exec nvcc -x cu prueba_cuda.cu -o prueba_cu -lcurand -o prueba_cu -rdc=true

#######################
