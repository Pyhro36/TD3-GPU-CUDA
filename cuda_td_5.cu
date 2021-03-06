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

#define BLOCK_SIDE 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this lab
    __shared__ float privateA[BLOCK_SIDE * BLOCK_SIDE];
    __shared__ float privateB[BLOCK_SIDE * BLOCK_SIDE];

    float cValue;
    int tile, i;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int rowC = tx + (blockDim.x * blockIdx.x);
    int colC = ty + (blockDim.y * blockIdx.y);

    if ((rowC < numCRows) && (colC < numCColumns)) {
        cValue = 0;

        for (tile = 0; tile < (1 + ((numAColumns - 1) / BLOCK_SIDE)); ++tile) {

            if (((tile * BLOCK_SIDE) + tx) < numAColumns) {
                privateA[(tx * BLOCK_SIDE) + ty] = A[(rowC * numAColumns) + (tile * BLOCK_SIDE) + ty];

            } else {
                privateA[(tx * BLOCK_SIDE) + ty] = 0.0f;
            }

            if (((tile * BLOCK_SIDE) + ty) < numBRows) {
                privateB[(tx * BLOCK_SIDE) + ty] = B[(((tile * BLOCK_SIDE) + tx) * numBColumns) + colC];

            } else {
                privateB[(tx * BLOCK_SIDE) + ty] = 0.0f;
            }

            __syncthreads();

            for (i = 0; i < BLOCK_SIDE; ++i) {
                cValue += privateA[(tx * BLOCK_SIDE) + i] * privateB[(i * BLOCK_SIDE) + ty];
            }

            __syncthreads();
        }

        C[(rowC * numCColumns) + colC] = cValue;
    }
}

int main(int argc, char **argv) {

    wbArg_t args;
    float *hostA; // The A matrix
    float *hostB; // The B matrix
    float *hostC; // The output C matrix
    float *deviceA;
    float *deviceB;
    float *deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);
    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    hostC = (float *)malloc((size_t)(sizeof(float) * numCRows * numCColumns));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
    wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck(cudaMalloc(&deviceA, numARows * numAColumns * sizeof(float)));
    wbCheck(cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(float)));
    wbCheck(cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 gridDim(1 + ((numCRows - 1) / BLOCK_SIDE), 1 + ((numCColumns - 1) / BLOCK_SIDE), 1);
    dim3 blockDim(BLOCK_SIDE, BLOCK_SIDE, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns,
            numCRows, numCColumns);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree(deviceA));
    wbCheck(cudaFree(deviceB));
    wbCheck(cudaFree(deviceC));
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);
    return 0;
}
