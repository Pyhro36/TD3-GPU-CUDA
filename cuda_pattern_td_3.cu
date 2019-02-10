// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];
#include <driver_types.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>
#include "wb.h"
#define BLOCK_SIZE 512 //@@ You can change this


#define wbCheck(stmt) \
do { \
    cudaError_t err = stmt; \
    if (err != cudaSuccess) { \
        wbLog(ERROR, "Failed to run stmt ", #stmt); \
        wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err)); \
        return -1; \
    } \
} while (0)

__global__ void total(float *input, float *output, int len) {

    //@@ Load a segment of the input vector into shared memory
    int stride;

    __shared__ float partialSum[2 * BLOCK_SIZE];

    int tx = threadIdx.x;
    int bDim = blockDim.x;
    int bx = blockIdx.x;
    int block = bx * bDim;
    int start = 2 * block;
    int ii = start + tx;

    if (ii < len) {
        partialSum[tx] = input[ii];

        if ((ii + bDim) < len) {
            partialSum[bDim + tx] = input[ii + bDim];
        }
    }

    //@@ Traverse the reduction tree
    for (stride = bDim; stride > 0; stride /= 2) {
        __syncthreads();

        if ((tx < stride) && ((ii + stride) < len)) {
            partialSum[tx] += partialSum[tx + stride];
        }
    }

    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index

    __syncthreads();

    if (tx == 0) {
        output[bx] = partialSum[0];
    }
}

int main(int argc, char **argv) {
    int ii;
    wbArg_t args;
    float *hostInput; // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);
    numOutputElements = numInputElements / (BLOCK_SIZE << 1);
    
    if (numInputElements % (BLOCK_SIZE << 1)) {
        numOutputElements++;
    }
    
    hostOutput = (float *)malloc(numOutputElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck(cudaMalloc((void **)&deviceInput, numInputElements * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy((void *)deviceInput, (void *)hostInput, numInputElements * sizeof(float),
            cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    // gridDim = numOutputElements
    // blockDim = BLOCK_SIZE

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    total<<<numOutputElements, BLOCK_SIZE>>>(deviceInput, deviceOutput, numInputElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy((void *)hostOutput, (void *)deviceOutput, numOutputElements * sizeof(float),
                       cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
    * Reduce output vector on the host
    * NOTE: One could also perform the reduction of the output vector
    * recursively and support any size input. For simplicity, we do not
    * require that for this lab.
    ********************************************************************/
    for (ii = 1; ii < numOutputElements; ++ii) {
        hostOutput[0] += hostOutput[ii];
    }

    printf("sum = %.6f\n", hostOutput[0]);

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree((void *)deviceInput));
    wbCheck(cudaFree((void *)deviceOutput));
    wbTime_stop(GPU, "Freeing GPU Memory");

    free(hostInput);
    free(hostOutput);
    return 0;
}

