#include "wb.h"

#define NUM_BINS 4096

#define BLOCK_DIM 64

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
                file, line);
        if (abort)
            exit(code);
    }
}

__global__ void histoKernel(unsigned int *input, unsigned int *bins, int inputLength) {

    __shared__ unsigned int privateBins[NUM_BINS];

    int privateI;
    int i = threadIdx.x + (blockIdx.x * blockDim.x);

    if (i < NUM_BINS)
    {
        bins[i] = 0;
    }

    // repartition de l'initialisation du bin prive entre les threads du bloc par partitionnement entrelace
    for (privateI = threadIdx.x; privateI < NUM_BINS; privateI += blockDim.x)
    {
        privateBins[privateI] = 0;
    }

    if (i < inputLength)
    {
        __syncthreads();
        atomicAdd(&(privateBins[input[i]]), 1);
    }

    __syncthreads();

    // repartition de la reduction du bin prive entre les threads du bloc par partitionnement entrelace
    for (privateI = threadIdx.x; privateI < NUM_BINS; privateI += blockDim.x)
    {
        atomicAdd(&(bins[privateI]), privateBins[privateI]);
    }
}

int main(int argc, char *argv[]) {

    wbArg_t args;
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *deviceInput;
    unsigned int *deviceBins;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 0), &inputLength, "Integer");
    hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);
    wbLog(TRACE, "The number of bins is ", NUM_BINS);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    CUDA_CHECK(cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    // Launch kernel
    // ----------------------------------------------------------
    wbLog(TRACE, "Launching kernel");
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Perform kernel computation here
    int gridDim = 1 + ((inputLength - 1) / BLOCK_DIM);
    // On prend des blocs dans lesquels rentrent juste les operations de reduction globale
    histoKernel<<<gridDim,BLOCK_DIM>>>(deviceInput, deviceBins, inputLength);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    CUDA_CHECK(cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost))
    CUDA_CHECK(cudaDeviceSynchronize());
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    CUDA_CHECK(cudaFree(deviceBins));
    CUDA_CHECK(cudaFree(deviceInput));
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostBins, NUM_BINS);

    free(hostBins);
    free(hostInput);
    return 0;
}
