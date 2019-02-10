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

#define I_TILE_WIDTH 16
#define I_TILE_HEIGHT 4
#define I_TILE_DEPTH 4

#define STENCIL_LENGTH 3
#define STENCIL_ADD_BY_AXE 2
#define STENCIL_EDGE 1

#define O_TILE_HEIGHT 14 //( I_TILE_HEIGHT - STENCIL_ADD_BY_AXE )
#define O_TILE_WIDTH 2 // ( I_TILE_WIDTH - STENCIL_ADD_BY_AXE )
#define O_TILE_DEPTH 2 //( I_TILE_DEPTH - STENCIL_ADD_BY_AXE )

__global__ void stencil(float *output, float *input, int width, int height, int depth) {

    //@@ INSERT CODE HERE
    float res;

    __shared__ float privateInput[I_TILE_WIDTH * I_TILE_HEIGHT * I_TILE_DEPTH];

//    __restrict_arr const float mask[STENCIL_LENGTH][STENCIL_LENGTH][STENCIL_LENGTH] {
//        {
//            { 0.0f, 0.0f, 0.0f },
//            { 0.0f, 1.0f, 0.0f },
//            { 0.0f, 0.0f, 0.0f }
//        }, {
//            { 0.0f, 1.0f, 0.0f },
//            { 1.0f, -6.0f, 1.0f },
//            { 0.0f, 1.0f, 0.0f }
//        }, {
//            { 0.0f, 0.0f, 0.0f },
//            { 0.0f, 1.0f, 1.0f },
//            { 0.0f, 0.0f, 0.0f }
//        }
//    };

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int outCol = tx + (O_TILE_WIDTH * blockIdx.x);
    int outRow = ty + (O_TILE_HEIGHT * blockIdx.y);
    int outChannel = tz + (O_TILE_DEPTH * blockIdx.z);
    int inCol = outCol - 1;
    int inRow = outRow - 1;
    int inChannel = outChannel - 1;

    if ((inCol > -1)
    && (inCol < width)
    && (inRow > -1)
    && (inRow < height)
    && (inChannel > -1)
    && (inChannel < depth)) {
        privateInput[(((tx * I_TILE_HEIGHT) + ty) * I_TILE_DEPTH) + tz] =
                input[(((inCol * height) + inRow) * depth) + inChannel];
    } else {
        privateInput[(((tx * I_TILE_HEIGHT) + ty) * I_TILE_DEPTH) + tz] = 0.0f;
    }

//    res = 0.0f;

    __syncthreads();

    if ((tx < O_TILE_WIDTH) && (ty < O_TILE_HEIGHT) && (tz < O_TILE_DEPTH)) {
         res =  privateInput[((((tx + 1) * I_TILE_HEIGHT) + ty + 1) * I_TILE_DEPTH) + tz + 2]           // [i,j,k+1]
              + privateInput[((((tx + 1) * I_TILE_HEIGHT) + ty + 1) * I_TILE_DEPTH) + tz]               // [i,j,k-1]
              + privateInput[((((tx + 1) * I_TILE_HEIGHT) + ty + 2) * I_TILE_DEPTH) + tz + 1]           // [i,j+1,k]
              + privateInput[((((tx + 1) * I_TILE_HEIGHT) + ty) * I_TILE_DEPTH) + tz + 1]               // [i,j-1,k]
              + privateInput[((((tx + 2) * I_TILE_HEIGHT) + ty + 1) * I_TILE_DEPTH) + tz + 1]           // [i+1,j,k]
              + privateInput[(((tx * I_TILE_HEIGHT) + ty + 1) * I_TILE_DEPTH) + tz + 1]                 // [i-1,j,k]
              - (6.0f * privateInput[((((tx + 1) * I_TILE_HEIGHT) + ty + 1) * I_TILE_DEPTH) + tz + 1]); // [i,j,k]
//        for (int i = 0; i < STENCIL_LENGTH; ++i) {
//            for (int j = 0; j < STENCIL_LENGTH; ++j) {
//                for (int k = 0; k < STENCIL_LENGTH; ++k) {
//                    res += mask[i][j][k] *
//                            privateInput[((((tx + i) * I_TILE_HEIGHT) + ty + j) * I_TILE_DEPTH) + tz + k];
//                }
//            }
//        }
    } else {
        res = 0.0f;
    }

    if ((outCol < width) && (outRow < height) && (outChannel < depth)) {

        if (res > 255.0f) {
            output[(((outCol * height) + outRow) * depth) + outChannel] = 1.0f;

        } else if (res < 0.0f) {
            output[(((outCol * height) + outRow) * depth) + outChannel] = 0.0f;

        } else {
            output[(((outCol * height) + outRow) * depth) + outChannel] = res;
        }
    }
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData, int width, int height, int depth) {

    //@@ INSERT CODE HERE
    dim3 gridDim(1 + ((width - 1) / O_TILE_WIDTH),
            1 + ((height - 1) / O_TILE_HEIGHT), (1 + ((depth - 1) / O_TILE_DEPTH)));
    dim3 blockDim(I_TILE_WIDTH, I_TILE_HEIGHT, I_TILE_DEPTH);
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

