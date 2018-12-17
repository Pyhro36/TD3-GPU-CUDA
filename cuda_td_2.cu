#include "wb.h"

#define BLOCK_SIDE 16

#define wbCheck(stmt)                                                       \
    do {                                                                    \
        cudaError_t err = stmt;                                             \
        if (err != cudaSuccess) {                                           \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                     \
            wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err));   \
            return -1;                                                      \
        }                                                                   \
    } while (0)
    
//@@ INSERT CODE HERE
__global__ void colorToGrayShadesKernel(float *in, float *out, int height, int width, int channels) {

    int ii = threadIdx.x + (blockDim.x * blockIdx.x);
    int jj = threadIdx.y + (blockDim.y * blockIdx.y);

    if ((ii < height) && (jj < width)) {
        int idx = (height * ii) + jj;
        float r = in[channels * idx];
        float g = in[(channels * idx) + 1];
        float b = in[(channels * idx) + 2];
        out[idx] = (0.21 * r) + (0.71 * g) + (0.07 * b);
    }
}

int main(int argc, char *argv[]) {

    wbArg_t args;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char *inputImageFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float *hostInputImageData;
    float *hostOutputImageData;
    float *deviceInputImageData;
    float *deviceOutputImageData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    // For this lab the value is always 3
    imageChannels = wbImage_getChannels(inputImage);
    // Since the image is monochromatic, it only contains one channel
    outputImage = wbImage_new(imageWidth, imageHeight, 1);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    wbCheck(cudaMalloc((void **)&deviceInputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceOutputImageData,
            imageWidth * imageHeight * sizeof(float)));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    wbCheck(cudaMemcpy((void *)deviceInputImageData, (void *)hostInputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float),
            cudaMemcpyHostToDevice));
    wbTime_stop(Copy, "Copying data to the GPU");

    ///////////////////////////////////////////////////////
    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    dim3 gridDim(1 + (((imageHeight) - 1) / BLOCK_SIDE), 1 + ((imageWidth - 1) / BLOCK_SIDE), 1);
    dim3 blockDim(BLOCK_SIDE, BLOCK_SIDE, 1);

    colorToGrayShadesKernel<<<gridDim, blockDim>>>(deviceInputImageData,
            deviceOutputImageData, imageHeight, imageWidth, imageChannels);
    wbTime_stop(Compute, "Doing the computation on the GPU");
    ///////////////////////////////////////////////////////

    wbTime_start(Copy, "Copying data from the GPU");
    wbCheck(cudaMemcpy((void *)hostOutputImageData, (void *)deviceOutputImageData,
            imageWidth * imageHeight * sizeof(float),
            cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    int i, j;
    FILE *fp = fopen("first.ppm", "wb"); /* b - binary mode */
    (void) fprintf(fp, "P5\n%d %d\n255\n", imageWidth, imageHeight);
    for (j = 0; j < imageWidth; ++j)
    {
        for (i = 0; i < imageHeight; ++i)
        {
            static unsigned char color[3];
            color[0] = hostOutputImageData[ (i * imageHeight) + j];
            (void) fwrite(color, 1, 1, fp);
        }
    }
    (void) fclose(fp);

    wbCheck(cudaFree(deviceInputImageData));
    wbCheck(cudaFree(deviceOutputImageData));
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    return 0;
}
