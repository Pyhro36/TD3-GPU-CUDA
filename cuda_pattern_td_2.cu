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

#define STENCIL_EDGE 1

#define O_TILE_MAX_I ( I_TILE_WIDTH - STENCIL_EDGE )
#define O_TILE_MAX_J ( I_TILE_HEIGHT - STENCIL_EDGE )
#define O_TILE_MAX_K ( I_TILE_DEPTH - STENCIL_EDGE )

__global__ void stencil(float *output, float *input, int width, int height, int depth) {

    //@@ INSERT CODE HERE
    float res;

    __shared__ float privateInput[I_TILE_WIDTH * I_TILE_HEIGHT * I_TILE_DEPTH];

    // on reunit width et depth pour avoir une grille 2D comme demande dans l'enonce
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int privateJ = ty / I_TILE_DEPTH;
    int privateK = ty % I_TILE_DEPTH;
    int xx = tx + (blockDim.x * blockIdx.x); // -> height
    int yy = ty + (blockDim.y * blockIdx.y); // -> width * depth
    int inXx = xx - STENCIL_EDGE;
    int inYy = yy -

    if ((inXx < height) && ((yy / depth) < width) && ((yy % depth) < depth))
    {
        privateInput[(tx * I_TILE_HEIGHT * I_TILE_DEPTH) + ty] = input[(xx * width * depth) + yy];

        // comme on ne se soucie pas des bornes, on utilise pas les valeurs en dehors de height et width de privateInput

        // on prend les threads de STENCIL__ à O_TILE_MAX__ pour calculer les output
        if ((tx > 0)
            && (tx < O_TILE_MAX_I)
            && (privateJ > 0)
            && (privateJ < O_TILE_MAX_J)
            && (privateK > 0)
            && (privateK < O_TILE_MAX_K))
        {
            res = privateInput[(tx * I_TILE_HEIGHT * I_TILE_DEPTH) + ty + 1]      // [i,j,k+1]
                  + privateInput[(tx * I_TILE_HEIGHT * I_TILE_DEPTH) + ty - 1]     // [i,j,k-1]
                  + privateInput[(((tx * I_TILE_HEIGHT) + 1) * I_TILE_DEPTH) + ty] // [i,j+1,k]
                  + privateInput[(((tx * I_TILE_HEIGHT) - 1) * I_TILE_DEPTH) + ty] // [i,j-1,k]
                  + privateInput[((tx + 1) * I_TILE_HEIGHT * I_TILE_DEPTH) + ty]   // [i+1,j,k]
                  + privateInput[((tx - 1) * I_TILE_HEIGHT * I_TILE_DEPTH) + ty]   // [i-1,j,k]
                  - (6 * privateInput[(tx * I_TILE_HEIGHT * I_TILE_DEPTH) + ty]);  // [i,j,k]

            if (res < 0.0f)
            {
                output[(xx * width * depth) + yy] = 0.0f;
            }
            else if (res > 255.0f)
            {
                output[(xx * width * depth) + yy] = 255.0f;
            }
            else
            {
                output[(xx * width * depth) + yy] = res;
            }
        }
        else
        {
            // aux bornes
            output[(xx * width * depth) + yy] = input[(xx * width * depth) + yy];
        }
    }
}

static void launch_stencil(float *deviceOutputData, float *deviceInputData, int width, int height, int depth) {

    //@@ INSERT CODE HERE
    // on reunit width et depth pour avoir une grille 2D comme demande dans l'enonce
    dim3 gridDim(1 + ((height - 1) / I_TILE_WIDTH),
            (1 + ((width - 1) / I_TILE_HEIGHT)) * (1 + ((depth - 1) / I_TILE_DEPTH)), 1);
    dim3 blockDim(I_TILE_WIDTH, I_TILE_HEIGHT * I_TILE_DEPTH, 1);
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

