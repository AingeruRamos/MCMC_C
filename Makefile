COMM_FLAGS=-std=c++11
CPP_FILES=src/ising.cpp src/rand.cpp

DEFINES ?=

main:
	g++ src/main.cpp $(CPP_FILES) -o main -fopenmp $(COMM_FLAGS) $(DEFINES)

prueba:
	g++ prueba.cpp $(CPP_FILES) -o prueba -fopenmp $(COMM_FLAGS)

# NVIDIA COMPILATIONS

main_cu:
	priscilla exec nvcc -x cu src/main.cu $(CPP_FILES) -lcurand -o main_cu -rdc=true $(DEFINES)

prueba_cu:
	priscilla exec nvcc -x cu prueba.cu -o prueba_cu -lcurand -o prueba -rdc=true

#######################
