
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

void parallelEdgeDetector(Matrix grayImage, Matrix gaussianKernel, string pathName, int numThreads);

int main(int argc, char* argv[]){
    string imagePathName("./input/image1.ppm");
    
    int gaussKernelSize = 7;
    int numThreads = 32; //Threads per block -- 32x32 or 16x16 or 8x8
    double sigma = 1.5;

    //////////////Read Image///////////////
    PPMImage *image = readPPM(imagePathName.c_str());
    string outputFileName = getFileName(imagePathName,false);

    //////////////Pre-Processing///////////////////
    Matrix grayImage = rgb2gray(image);
    Matrix gaussianKernel = createGaussianKernel(gaussKernelSize, sigma);

    //////////////Parallel Processing///////////////
    double parallelTime = parallelEdgeDetector(grayImage, gaussianKernel, outputFileName, numThreads);

    //Free memory
    freeImage(image);

    return 0;
}


/*
Parallel Edge Detection using CUDA
*/
void parallelEdgeDetector(Matrix grayImage, Matrix gaussianKernel, string pathName, int numThreads){

    int imageWidth = grayImage.getCols(), imageHeight = grayImage.getRows();
    int imageSize = imageWidth * imageHeight;
    int kernelSize = gaussianKernel.getRows();
    cudaError_t err;

    //Arrays of CPU
    float* result = allocateArray(imageSize);
    float* grayImageArray = grayImage.toArray();
    float* gaussKernelArray = gaussianKernel.toArray();
    
    //Arrays used in GPU
    float* kernelDevice; //Used in Gauss
    float* imageDevice, *resultDevice; //Used in both Gauss and Sobel

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
    

    DynamicTiledConvolution << < dimGaussGrid, dimGaussBlock, blockWidth * blockWidth * sizeof(float) >> > (imageDevice, resultDevice, imageWidth, imageHeight, kernelDevice, kernelSize, tileWidth, blockWidth);
    err = cudaGetLastError();
    if (err != cudaSuccess)	printf("Gauss Error: %s\n", cudaGetErrorString(err));

    cudaDeviceSynchronize();
    
    //Copy Gaussian result from device to device
    cudaMemcpy(imageDevice, resultDevice, imageSize * sizeof(float), cudaMemcpyDeviceToDevice);
    
    //Tiled Sobel Filter
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    sobelInCuda <<<blocksPerGrid, threadsPerBlock >>>(imageDevice, resultDevice, imageHeight, imageWidth);
    
    //Copy Result from GPU to CPU
    cudaMemcpy(result, resultDevice, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    ///////////////Save Image Result/////////////////
    PPMImage* imageResult = createImage(imageHeight, imageWidth);
    Matrix dataResult = arrayToMatrix(result, imageHeight, imageWidth);
    Matrix normalized = normalize(dataResult, 0, 255); //Normalize values
    matrixToImage(normalized, imageResult);
    #ifdef _WIN32
        pathName = "..\\output\\"+pathName+"_gpu.ppm";
    #else
        pathName = "../output/"+pathName+"_gpu.ppm";
    #endif
    writePPM(pathName.c_str(), imageResult);
    freeImage(imageResult);
    /////////////////////////////////////////////////

    cout << "GPU Gauss Time: " << setprecision(3) << fixed << gaussTime << "s." << endl;
    cout << "GPU Sobel Time: " << setprecision(3) << fixed << sobelTime << "s." << endl;
    cout << "GPU Time: " << setprecision(3) << fixed << gaussTime + sobelTime << "s." << endl;

    ////////////////Free Memory used in GPU and CPU//////////////////
    cudaFree(xGradDevice);	cudaFree(yGradDevice); cudaFree(imageDevice);
    cudaFree(resultDevice); cudaFree(kernelDevice);
    freeArray(xGradient);	freeArray(yGradient);  freeArray(result);
    freeArray(grayImageArray); freeArray(gaussKernelArray);
    /////////////////////////////////////////////////////////////////
}