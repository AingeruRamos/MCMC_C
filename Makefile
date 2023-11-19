COMM_FLAGS=-std=c++11 -g
CPP_FILES=ising.cpp rand.cpp

main:
	g++ main.cpp $(CPP_FILES) -o main -fopenmp $(COMM_FLAGS) -DN_ROW=10

prueba:
	g++ prueba.cpp $(CPP_FILES) -o prueba -fopenmp $(COMM_FLAGS)

# NVIDIA COMPILATIONS

main_cu:
	priscilla exec nvcc main.cu $(CPP_FILES) -o main_cu -lcurand

prueba_cu:
	priscilla exec nvcc prueba.cu -o prueba_cu

#######################