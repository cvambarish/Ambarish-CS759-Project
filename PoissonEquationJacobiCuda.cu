//============================================================================
// Name        : PoissonEquationJacobiCuda.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using namespace std;

const float PI = 4*atan(1);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void jacobiMethod(float* grid,float* potential, int sizeX,int sizeY,float scale,int noIters,float tolerance){

	extern __shared__ float sharedMem[];
	/*
		Shared memory
			1st part is grid
			2nd part is initial guess
			3rd part is current Solution
	
	*/


	// Copying from global to shared memory
	int threadIdX = threadIdx.x;
	int threadIdY = threadIdx.y;

	if (threadIdX == 0 && threadIdY == 0) {
		//printf("At Beginning\n");
	}


	int bOx = blockIdx.x * blockDim.x;
	int bOy = blockIdx.y * blockDim.y;

	//int totalBlockThreadId = threadIdY*blockDim.x + threadIdX;

	//int blockThreadIdx = threadIdX-noIters;
	//int blockThreadIdy = threadIdY-noIters;	
	
	int effBlockSizeX = blockDim.x + 2 * noIters;
	int effBlockSizeY = blockDim.y + 2 * noIters;
	
	int totalSize = sizeX*sizeY;

	int sharedMemSize = effBlockSizeX*effBlockSizeY;

	for(int i= threadIdX;i<effBlockSizeX;i+= blockDim.x)
		for (int j = threadIdY; j < effBlockSizeY; j += blockDim.y) {
			int currElemSM = i*effBlockSizeX + j;
			int currElemMain = (i - noIters + bOy)*sizeX + (j - noIters + bOx);
			if (currElemMain >= 0 && currElemMain < totalSize) {
				sharedMem[currElemSM] = grid[currElemMain];
				sharedMem[currElemSM + sharedMemSize] = potential[currElemMain];
			}
			else {
				sharedMem[currElemSM] = 0;
				sharedMem[currElemSM + sharedMemSize] = 0;
			}
			sharedMem[currElemSM + 2 * sharedMemSize] = 0;
		}
	__syncthreads();
	if (threadIdX == 0 && threadIdY == 0) {
		//printf("Copied to shared memory\n");
	}

	for(int k=0;k<noIters;k++){
		for(int i= threadIdX;i<effBlockSizeX;i+= blockDim.x)
			for(int j= threadIdY;j<effBlockSizeY;j+= blockDim.y){
				int currPos = i*effBlockSizeX +j+ sharedMemSize*2;
				sharedMem[currPos]=0;
				if(i>1){
					sharedMem[currPos]+=(sharedMem[currPos- effBlockSizeY- sharedMemSize]/4);
				}
				if(i<effBlockSizeX -1){
					sharedMem[currPos]+=(sharedMem[currPos+ effBlockSizeY - sharedMemSize]/4);
				}
				if(j>1){
					sharedMem[currPos]+=(sharedMem[currPos-1- sharedMemSize]/4);
				}
				if(j<effBlockSizeY-1){
					sharedMem[currPos]+=(sharedMem[currPos+1- sharedMemSize]/4);
				}
				if(i== effBlockSizeX-1||j== effBlockSizeY-1){
					//currSolution[currPos]=0;
				}else{
					sharedMem[currPos]+=(scale*scale/4* sharedMem[currPos-2* sharedMemSize]);
				}
			}
		__syncthreads();

		for (int i = threadIdX; i<effBlockSizeX; i += blockDim.x)
			for (int j = threadIdY; j<effBlockSizeY; j += blockDim.y) {
				int currPos = i*effBlockSizeX + j + sharedMemSize * 2;
				sharedMem[currPos- sharedMemSize]= sharedMem[currPos];
			}
		__syncthreads();
	}
	if (threadIdX == 0 && threadIdY == 0) {
		//printf("Done computation\n");
	}

	for (int i = threadIdX; i<effBlockSizeX; i += blockDim.x)
		for (int j = threadIdY; j < effBlockSizeY; j += blockDim.y) {
			if (i >= noIters && j >= noIters && i < effBlockSizeX - noIters && j < effBlockSizeX - noIters) {
				int currElemSM = i*effBlockSizeX + j;
				int currElemMain = (i - noIters + bOy)*sizeX + (j - noIters + bOx);
				potential[currElemMain] = sharedMem[currElemSM + 2* sharedMemSize];
			}
		}
	if (threadIdX == 0 && threadIdY == 0) {
		//printf("Copied to memory\n");
	}

}

void createDiskInitialCharge(float* input, float totalCharge, int sizeX,int sizeY, int xCen, int yCen, int radius){
	float chargePerPoint = totalCharge/(PI*radius*radius);
	int countPoints=0;
	for(int i=0;i<sizeX;i++)
		for(int j=0;j<sizeY;j++){
			float currentDist = (i-xCen)*(i-xCen) + (j-yCen)*(j-yCen);
			//printf("Check at %d,%d\n",i,j);
			int currPos = i*sizeY+j;
			if(currentDist <= radius*radius ){
				input[currPos]=chargePerPoint;
				countPoints++;
			}
			else{
				input[currPos]=0;
			}
		}

}

void initializeProblem(float* grid,float* potential,float totalCharge,int sizeX,int sizeY,int sizeXCen,int sizeYCen,int discRadius){

	for(int i=0;i<sizeX*sizeY;i++)
		potential[i]=0.0;

	createDiskInitialCharge(grid, totalCharge, sizeX, sizeY, sizeXCen,sizeYCen, discRadius);
}

void calculateElectricField(float* potential,float* field,int sizeX,int sizeY,float scale){

	for(int i=0;i<sizeX;i++)
		for(int j=0;j<sizeY;j++){
			int currPos = (i*sizeY+j);

			float currValPlusY=0;
			float currValPlusX=0;
			float currVal=potential[currPos];
			if(i==sizeY-1){
				currValPlusY=0;
			}
			else{
				currValPlusY = potential[currPos+sizeX];
			}

			if(j==sizeX-1){
				currValPlusX=0;
			}
			else{
				currValPlusX = potential[currPos+1];
			}
			field[2*currPos] = -(currValPlusX-currVal)/scale;
			field[2*currPos+1] = -(currValPlusY-currVal)/scale;
		}
}



void writeResultToFile(float* field,char* filename,int totalSize, int multiplicity){
	FILE* file = fopen(filename,"w");
	for(int i=0;i<totalSize;i++){
		for(int j=0;j<multiplicity;j++){
			fprintf(file,"%f ",field[i*multiplicity+j]);
		}
		fprintf(file,"\n");
	}
	fclose(file);
}


int main(int argc, char** argv) {

	int sizeX=100;
	int sizeY=100;
	float actualRadius = 0.05; // in m
	float scale = 0.01; // in m
	int discRadius = actualRadius/scale;
	int noIters = 10000;
	int noExtra=1;
	if(argc == 5){
		sizeX = atoi(argv[1]);
		sizeY=sizeX;
		discRadius=atoi(argv[2]);
		noIters=atoi(argv[3]);
		noExtra=atoi(argv[4]);
	}
	else{
		printf("Incorrect usage, proper usage is jacobi $sizeX $discRadius $noThreads");
		return;
	}
	printf("Radius : %d\n",discRadius);
	printf("Grid Size : %d\n",sizeX);
	printf("No Iterations : %d\n",noIters);
	printf("No Extra : %d\n",noExtra);

	int sizeXCen=sizeX/2;
	int sizeYCen=sizeY/2;
	int totalSize=sizeX*sizeY;

    float *grid = (float *)malloc(totalSize * sizeof(float));

    float* potential = (float *)malloc(totalSize * sizeof(float));

    printf("Begin..\n");

	float totalCharge = 250000;
	float timeTaken = 1.0f;

	initializeProblem(grid,potential, totalCharge, sizeX, sizeY, sizeXCen,sizeYCen, discRadius);

    printf("Problem Initialized..\n");

    float tolerance = 0.001;
	float charge = 0.0;
	for (int i = 0; i<sizeX; i++)
		for (int j = 0; j<sizeY; j++) {
			int currPos = i*sizeY + j;
			charge += grid[currPos];
			//printf("%f\n",potential[currPos]);
		}
	printf("Total charge is %f\n", charge);

    float *d_grid, *d_potential;
	cudaMalloc(&d_grid, totalSize*sizeof(float));
	cudaMalloc(&d_potential, totalSize * sizeof(float));

	cudaEvent_t startEventOrig_inc, stopEventOrig_inc;
	cudaEventCreate(&startEventOrig_inc);
	cudaEventCreate(&stopEventOrig_inc);
	cudaEventRecord(startEventOrig_inc, 0);
	
	gpuErrchk(cudaMemcpy(d_grid, grid, totalSize * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_potential, potential, totalSize * sizeof(float), cudaMemcpyHostToDevice));
    
	int threadPerBlock=32;
	dim3 blockSize(threadPerBlock,threadPerBlock,1);
	int noBlocks = sizeX/threadPerBlock +1;
	dim3 gridSize(noBlocks,noBlocks,1);

	int sharedMemSize = 3*(threadPerBlock+2*noExtra)*(threadPerBlock+2*noExtra)*sizeof(float);
	int totalIters = noIters/noExtra+1;
	
	printf("Total Kernel Calls : %d\n", totalIters);
	gpuErrchk(cudaPeekAtLastError());

	for (int k = 0; k < totalIters; k++) {

		jacobiMethod << <gridSize, blockSize, sharedMemSize >> >(d_grid, d_potential, sizeX, sizeY, scale, noExtra, tolerance);
		gpuErrchk(cudaPeekAtLastError());
	}
	//gpuErrchk(cudaDeviceSynchronize());
	cudaMemcpy(potential, d_potential, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
	//gpuErrchk(cudaPeekAtLastError());
	printf("Jacobi Method Ended\n");

	//Calculating Field
	float* field = (float*)malloc(sizeX*sizeY*2*sizeof(float));

	calculateElectricField(potential,field,sizeX,sizeY,scale);
	
	float time = 0.0;
	cudaEventRecord(stopEventOrig_inc, 0);  //ending timing for inclusive
	cudaEventSynchronize(stopEventOrig_inc);
	cudaEventElapsedTime(&time, startEventOrig_inc, stopEventOrig_inc);
	
	printf("Time taken %f\n", time);

	cudaDeviceReset();
	
	
	
	
	
	
	
	
	char* outputFilenameField = "ElectricField.out";
	char* outputFilenamePotential = "Potential.out";
	char* outputFilenameGrid = "Grid.out";
	printf("Writing to file..\n");
    fflush(stdout);
	writeResultToFile(field,outputFilenameField,totalSize,2);
	writeResultToFile(potential,outputFilenamePotential,totalSize,1);
	writeResultToFile(grid,outputFilenameGrid,totalSize,1);
	cudaFree(d_grid);
	cudaFree(d_potential);
	free(grid);
	free(field);
	free(potential);
	printf("All Done..\n");
	return 0;
}
