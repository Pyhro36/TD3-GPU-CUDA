#include "wb.h"
#define BLOCK_DIM 64

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {

    int i = threadIdx.x + (blockDim.x * blockIdx.x);

    if (i < len) {
        out[i] = in1[i] + in2[i];
    }
}

int main(int argc, char **argv) {
    
    wbArg_t args;
    int inputLength;
    size_t inputSize;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;
    float *deviceInput1;
    float *deviceInput2;
    float *deviceOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    inputSize = inputLength * sizeof(float);
    cudaMalloc((void **)(&deviceInput1), inputSize);
    cudaMalloc((void **)(&deviceInput2), inputSize);
    cudaMalloc((void **)(&deviceOutput), inputSize);
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(Copy, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy((void *)deviceInput1, (void *)hostInput1, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)deviceInput2, (void *)hostInput2, inputSize, cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    int gridDim = 1 + ((inputLength - 1) / BLOCK_DIM);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    vecAdd<<<gridDim, BLOCK_DIM>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy((void *)hostOutput, (void *)deviceOutput, inputSize, cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree((void *)deviceInput1);
    cudaFree((void *)deviceInput2);
    cudaFree((void *)deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
