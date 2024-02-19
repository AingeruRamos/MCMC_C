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
	#bnd -exec nvcc src/main_cuda.cu $(CPP_FILES) -o main_cu -lcurand -rdc=true $(COMM_FLAGS) $(DEFINES)
	apptainer exec -B /pfs --nv /opt/ohpc/pub/containers/NGC-pytorch-23.12-py3.sif nvcc src/main_cuda.cu $(CPP_FILES) -o main_cu -lcurand -rdc=true $(COMM_FLAGS) $(DEFINES)

prueba_cu:
	#bnd -exec nvcc prueba_cuda.cu -lcurand -x cu -rdc=true $(COMM_FLAGS)
	apptainer exec -B /pfs --nv /opt/ohpc/pub/containers/NGC-pytorch-23.12-py3.sif nvcc prueba_cuda.cu $(CPP_FILES) -o prueba_cu -lcurand -rdc=true $(COMM_FLAGS) $(DEFINES)

#######################

