#include <math.h>
#include <stdio.h>
#include <ctime>
#include <string>
#include <iostream>
#include <iomanip>
using namespace std;

#include "Matrix.h" //Matrix class
#include "utils.h" //Utils used in both Gauss and Sobel
#include "Gauss.h" //Serial Gauss
#include "GaussFilter.cuh" //Parallel Gauss
#include "Sobel.cu"

void parallelEdgeDetector(Matrix grayImage, Matrix gaussianKernel, string pathName, int numThreads, int imageWidth, int imageHeight, float* result);

int main(int argc, char* argv[]){
    string imagePathName_jpg;
    string imagePathName_ppm;
    string jpgstr = ".jpg";
    string ppmstr = ".ppm";

    if (argc < 2)
    {
        printf("Please provide image path.");
    } else
    {
        imagePathName_jpg = argv[1];
        imagePathName_ppm = argv[1];
    }
    imagePathName_ppm.replace(imagePathName_ppm.find(jpgstr),jpgstr.length(), ppmstr);
    string command = "convert " + imagePathName_jpg + " " + imagePathName_ppm;

    system(command.c_str());
    
    int gaussKernelSize = 7;
    int numThreads = 32; //Threads per block -- 32x32 or 16x16 or 8x8
    double sigma = 1.5;

    //////////////Read Image///////////////
    PPMImage *image = readPPM(imagePathName_ppm.c_str());
    string outputFileName_ppm = getFileName(imagePathName_ppm,false);

    //////////////Pre-Processing///////////////////
    Matrix grayImage = rgb2gray(image);
    Matrix gaussianKernel = createGaussianKernel(gaussKernelSize, sigma);
    int imageWidth = grayImage.getCols(), imageHeight = grayImage.getRows();
    float* result = allocateArray(imageWidth*imageHeight);

    //////////////Parallel Processing///////////////
    parallelEdgeDetector(grayImage, gaussianKernel, outputFileName_ppm, numThreads, imageWidth, imageHeight, result);

    ///////////////Save Image Result/////////////////    
    PPMImage* imageResult = createImage(imageHeight, imageWidth);
    Matrix dataResult = arrayToMatrix(result, imageHeight, imageWidth);
    Matrix normalized = normalize(dataResult, 0, 255); //Normalize values
    matrixToImage(normalized, imageResult);
    outputFileName_ppm = "./outputs/"+outputFileName_ppm+"_gpu.ppm";
    string outputFileName_jpg = outputFileName_ppm;
    outputFileName_jpg.replace(outputFileName_jpg.find(ppmstr),ppmstr.length(), jpgstr);
    writePPM(outputFileName_ppm.c_str(), imageResult);
    freeImage(imageResult);

    int* finalresult = new int[imageWidth*imageHeight];
    for (int i = 0; i < imageWidth*imageHeight; i++)
    {
        finalresult[i] = (int)result[i];
    }
    command = "convert " + outputFileName_ppm + " " + outputFileName_jpg;
    system(command.c_str());

    //Free memory
    freeArray(result);
    freeImage(image);

    return 0;
}


/*
Parallel Edge Detection using CUDA
*/
void parallelEdgeDetector(Matrix grayImage, Matrix gaussianKernel, string pathName, int numThreads, int imageWidth, int imageHeight, float* result){

    cudaError_t err;

    //Arrays of CPU
    float* grayImageArray = grayImage.toArray();
    float* gaussKernelArray = gaussianKernel.toArray();
    
    //Arrays used in GPU
    float* kernelDevice; //Used in Gauss
    float* imageDevice, *resultDevice; //Used in both Gauss and Sobel
    int imageSize = imageWidth * imageHeight;
    int kernelSize = gaussianKernel.getRows();

    //Allocate Device Memory
    cudaMalloc((void**)&imageDevice, imageSize * sizeof(float));
    cudaMalloc((void**)&resultDevice, imageSize* sizeof(float));
    cudaMalloc((void**)&kernelDevice, kernelSize * kernelSize * sizeof(float));

    //Copy values from CPU to GPU
    cudaMemcpy(imageDevice, grayImageArray, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernelDevice, gaussKernelArray, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    
    //Gaussian Blur
    int tileWidth = numThreads - kernelSize + 1;
    int blockWidth = tileWidth + kernelSize - 1;
    dim3 dimGaussBlock(blockWidth, blockWidth);
    dim3 dimGaussGrid((imageWidth - 1) / tileWidth + 1, (imageHeight - 1) / tileWidth + 1);
    

    DynamicTiledConvolution <<< dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >>> (imageDevice, resultDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
    err = cudaGetLastError();
    if (err != cudaSuccess)	printf("Gauss Error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
    
    //Copy Gaussian result from device to device
    cudaMemcpy(imageDevice, resultDevice, imageSize * sizeof(float), cudaMemcpyDeviceToDevice);
    
    //Tiled Sobel Filter
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imageWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (imageHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    sobelInCuda <<<blocksPerGrid, threadsPerBlock >>>(imageDevice, resultDevice, imageHeight, imageWidth);
    
    //Copy Result from GPU to CPU
    cudaMemcpy(result, resultDevice, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    ///////////////Save Image Result/////////////////
    //PPMImage* imageResult = createImage(imageHeight, imageWidth);
    //Matrix dataResult = arrayToMatrix(result, imageHeight, imageWidth);
    //Matrix normalized = normalize(dataResult, 0, 255); //Normalize values
    //matrixToImage(normalized, imageResult);
    //pathName = "./output/"+pathName+"_gpu.ppm";
    //writePPM(pathName.c_str(), imageResult);
    //freeImage(imageResult);
    /////////////////////////////////////////////////

    ////////////////Free Memory used in GPU and CPU//////////////////
    cudaFree(imageDevice);
    cudaFree(resultDevice); cudaFree(kernelDevice);
    freeArray(grayImageArray); freeArray(gaussKernelArray);
    /////////////////////////////////////////////////////////////////
}
