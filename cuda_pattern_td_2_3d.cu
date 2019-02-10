#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>
#include "wb.h"

#define wbCheck(stmt) \
 do { \
 cudaError_t err = stmt; \
 if (err != cudaSuccess) { \
 wbLog(ERROR, "Failed to run stmt ", #stmt); \
 wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err)); \
 return -1; \
 } \
 } while (0)

#define I_TILE_HEIGHT 16
#define I_TILE_WIDTH 4
#define I_TILE_DEPTH 4

#define STENCIL_ADD_BY_AXE 2
#define STENCIL_EDGE ( STENCIL_ADD_BY_AXE / 2 )

#define O_TILE_HEIGHT ( I_TILE_HEIGHT - STENCIL_ADD_BY_AXE )
#define O_TILE_WIDTH ( I_TILE_WIDTH - STENCIL_ADD_BY_AXE )
#define O_TILE_DEPTH ( I_TILE_DEPTH - STENCIL_ADD_BY_AXE )

typedef int i1;
__global__ void stencil(float *output, float *input, int width, int height, int depth) {

    //@@ INSERT CODE HERE
    __shared__ float privateInput[I_TILE_HEIGHT * I_TILE_WIDTH * I_TILE_DEPTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int outRow = tx + (O_TILE_HEIGHT * blockIdx.x);
    int outChannel = tz + (O_TILE_DEPTH * blockIdx.z);
    int inRow = outRow - STENCIL_EDGE;
    int inCol = outCol - STENCIL_EDGE;
    int inChannel = outChannel - STENCIL_EDGE;

    if ((inRow >= 0) && (inRow < height) && (inCol >= 0) && (inCol < width) && (inChannel >= 0) && (inChannel < depth))
    {
        privateInput[((tx  + (ty * I_TILE_HEIGHT)) * I_TILE_DEPTH) + tz] =
                input[(((inCol * height) + inRow) * depth) + inChannel];
    }
    else
    {
        privateInput[((tx + (ty * I_TILE_HEIGHT)) * I_TILE_DEPTH) + tz] = 0.0f;
    }
    // comme on ne se soucie pas des bornes, on utilise pas les valeurs en dehors de height et width de privateInput

    if ((outRow < height) && (outCol < width) && (outChannel < depth)
    && (tx < O_TILE_HEIGHT) && (ty < O_TILE_WIDTH) && (tz < O_TILE_DEPTH))
    {
        output[((outRow + (outCol * height)) * depth) + outChannel] =
                privateInput[((tx + 1 + ((ty + 1) * I_TILE_HEIGHT)) * I_TILE_DEPTH) + tz + 2]           // [i,j,k+1]
              + privateInput[(((tx + 1) + ((ty + 1) * I_TILE_HEIGHT)) * I_TILE_DEPTH) + tz]             // [i,j,k-1]
              + privateInput[((tx + 1 + ((ty + 2) * I_TILE_HEIGHT)) * I_TILE_DEPTH) + tz]               // [i,j+1,k]
              + privateInput[((tx + 1 + (ty * I_TILE_DEPTH)) * I_TILE_HEIGHT) + tz + 1]                 // [i,j-1,k]
              + privateInput[(((tx + 2) + ((ty + 1) * I_TILE_HEIGHT)) * I_TILE_DEPTH) + tz + 1]         // [i+1,j,k]
              + privateInput[((tx + ((ty + 1) * I_TILE_HEIGHT)) * I_TILE_DEPTH) + tz + 1]               // [i-1,j,k]
              - (6 * privateInput[((tx + 1 + ((ty + 1) * I_TILE_HEIGHT)) * I_TILE_DEPTH) + tz + 1]);    // [i,j,k]
    }
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData, int width, int height, int depth) {

    //@@ INSERT CODE HERE
    dim3 gridDim(1 + ((height - 1) / O_TILE_HEIGHT),
            (1 + ((width - 1) / O_TILE_WIDTH)), (1 + ((depth - 1) / O_TILE_DEPTH)));
    dim3 blockDim(I_TILE_HEIGHT, I_TILE_WIDTH, I_TILE_DEPTH);
    stencil<<<gridDim, blockDim>>>(deviceOutputData, deviceInputData, width, height, depth);
    cudaDeviceSynchronize();
}

int main(int argc, char *argv[]) {
    wbArg_t arg;
    int width;
    int height;
    int depth;
    char *inputFile;
    wbImage_t input;
    wbImage_t output;
    float *hostInputData;
    float *hostOutputData;
    float *deviceInputData;
    float *deviceOutputData;
    arg = wbArg_read(argc, argv);
    inputFile = wbArg_getInputFile(arg, 0);
    input = wbImport(inputFile);
    width = wbImage_getWidth(input);
    height = wbImage_getHeight(input);
    depth = wbImage_getChannels(input);
    output = wbImage_new(width, height, depth);
    hostInputData = wbImage_getData(input);
    hostOutputData = wbImage_getData(output);
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&deviceInputData, width * height * depth * sizeof(float));
    cudaMalloc((void **)&deviceOutputData, width * height * depth * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputData, hostInputData, width * height * depth * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");
    wbTime_start(Compute, "Doing the computation on the GPU");
    launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
    wbTime_stop(Compute, "Doing the computation on the GPU");
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputData, deviceOutputData, width * height * depth * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
    cudaFree(deviceInputData);
    cudaFree(deviceOutputData);

    wbSolution(arg, output);

    wbImage_delete(output);
    wbImage_delete(input);
    return 0;
}

