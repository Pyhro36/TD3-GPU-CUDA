#include "wb.h"

#define wbCheck(stmt) \
    do { \
    cudaError_t err = stmt; \
    if (err != cudaSuccess) { \
        wbLog(ERROR, "Failed to run stmt ", #stmt); \
        wbLog(ERROR, "Got CUDA error ... ", cudaGetErrorString(err)); \
        return -1; \
    } \
} while (0) \

#define BLUR_SIZE 5

//@@ INSERT CODE HERE

int main(int argc, char *argv[]) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
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
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, 3);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **)&deviceInputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **)&deviceOutputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData, hostInputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float),
    cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");
    ///////////////////////////////////////////////////////
    wbTime_start(Compute, "Doing the computation on the GPU");
    wbTime_stop(Compute, "Doing the computation on the GPU");
    ///////////////////////////////////////////////////////

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData, deviceOutputImageData,
            imageWidth * imageHeight * sizeof(float),
    cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
    
    int i, j;
    FILE *fp = fopen("first.ppm", "wb"); /* b - binary mode */
    (void) fprintf(fp, "P6\n%d %d\n255\n", imageWidth, imageHeight);
    for (j = 0; j < imageWidth; ++j)
    {
        for (i = 0; i < imageHeight; ++i)
        {
            static unsigned char color[3];
            color[0] = hostOutputImageData[i*imageHeight+j];  /* red */
            color[1] = hostOutputImageData[i*imageHeight+j+1];  /* green */
            color[2] = hostOutputImageData[i*imageHeight+j+2];  /* blue */
      
            (void) fwrite(color, 1, 3, fp);
        }
    }
    (void) fclose(fp);
    
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    return 0;
}
