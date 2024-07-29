#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define _CUDA(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char** argv) {
	
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);

	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Total global memory:           %u", devProp.totalGlobalMem);
	printf(" bytes\n");
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Total amount of shared memory per block: %u\n",devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %u\n", devProp.memPitch);
	printf("Total amount of constant memory:         %u\n",   devProp.totalConstMem);

	while(1){}

	return 0;
}
