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

#define BLUR_SIZE 1

//@@ INSERT CODE HERE
#define BLOCK_SIDE 16

__global__ void blurringKernel(float *in, float *out, int height, int width, int channels) {

    int row = threadIdx.x + (blockDim.x * blockIdx.x);
    int col = threadIdx.y + (blockDim.y * blockIdx.y);
    int channel = threadIdx.z + (blockDim.z * blockIdx.z);

    if ((row < height) && (col < width)) {

        float pixVal = 0.0;
        int blurPixelCount = 0;

        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow)
        {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol)
            {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if ((curRow >= 0) && (curRow < height) && (curCol >= 0) && (curCol < width))
                {
                    pixVal += in[(((curCol * height) + curRow) * channels) + channel];
                    blurPixelCount++;
                }
            }
        }

        out[(((col * height) + row) * channels) + channel] = pixVal / ((float)blurPixelCount);
    }
}

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
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
    wbTime_start(GPU, "Doing GPU memory allocation");
    wbCheck(cudaMalloc((void **)&deviceInputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceOutputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbTime_stop(GPU, "Doing GPU memory allocation");
    wbTime_start(Copy, "Copying data to the GPU");
    wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float),
    cudaMemcpyHostToDevice));
    wbTime_stop(Copy, "Copying data to the GPU");
    ///////////////////////////////////////////////////////
    wbTime_start(Compute, "Doing the computation on the GPU");
    dim3 gridDim(1 + ((imageHeight - 1) / BLOCK_SIDE), 1 + ((imageWidth - 1) / BLOCK_SIDE), imageChannels);
    dim3 blockDim(BLOCK_SIDE, BLOCK_SIDE, 1); // le but est d'avoir toujours des blocks de 16 * 16
    blurringKernel<<<gridDim, blockDim>>>(deviceInputImageData,
            deviceOutputImageData, imageHeight, imageWidth, imageChannels);
    wbTime_stop(Compute, "Doing the computation on the GPU");
    ///////////////////////////////////////////////////////

    wbTime_start(Copy, "Copying data from the GPU");
    wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float),
            cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying data from the GPU");
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
    
    int row, col, channel;
    FILE *fp = fopen("first.ppm", "wb"); /* b - binary mode */
    (void) fprintf(fp, "P6\n%d %d\n255\n", imageWidth, imageHeight);
    for (col = 0; col < imageWidth; ++col)
    {
        for (row = 0; row < imageHeight; ++row)
        {
            static unsigned char color[3];
            for (channel = 0; channel < imageChannels; ++channel)
            {
                color[channel] = (unsigned char)(hostOutputImageData[(((col * imageHeight) + row) * imageChannels) + channel]
                        * 255.0f);
            }

            (void) fwrite(color, 1, imageChannels, fp);
        }
    }
    (void) fclose(fp);

    wbTime_start(GPU, "Doing GPU memory freeing");
    wbCheck(cudaFree(deviceInputImageData));
    wbCheck(cudaFree(deviceOutputImageData));
    wbTime_stop(GPU, "Doing GPU memory freeing");

    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
