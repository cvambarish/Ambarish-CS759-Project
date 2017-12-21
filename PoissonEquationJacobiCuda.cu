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

const double PI = 4*atan(1);

__global__ void jacobiMethod(double* grid,double* potential, int sizeX,int sizeY,double scale,int noIters,double tolerance){

	extern __shared__ double sharedMem[];
	/*
		Shared memory
			1st part is grid
			2nd part is initial guess
			3rd part is current Solution
	
	*/


	// Copying from global to shared memory
	
	int bOx = blockIdx.x * blockDim.x;
	int bOy = blockIdx.y * blockDim.y;
	
	int threadIdX = threadIdx.x;
	int threadIdY = threadIdx.y;
	
	int totalBlockThreadId = threadIdY*blockDim.x + threadIdX;

	int blockThreadIdx = threadIdX-noIters;
	int blockThreadIdy = threadIdY-noIters;	
	
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




	for(int k=0;k<noIters;k++){
		for(int i= threadIdX;i<effBlockSizeX;i+= blockDim.x)
			for(int j= threadIdY;j<effBlockSizeY;j+= blockDim.y){
				int currPos = i*effBlockSizeX +j+ sharedMemSize*2;
				sharedMem[currPos]=0;
				if(i>1){
					sharedMem[currPos]+=(sharedMem[currPos- effBlockSizeY- sharedMemSize]/4);
				}
				if(i<sizeX-1){
					sharedMem[currPos]+=(sharedMem[currPos+ effBlockSizeY - sharedMemSize]/4);
				}
				if(j>1){
					sharedMem[currPos]+=(sharedMem[currPos-1- sharedMemSize]/4);
				}
				if(j<sizeY-1){
					sharedMem[currPos]+=(sharedMem[currPos+1- sharedMemSize]/4);
				}
				if(i==sizeX-1||j==sizeY-1){
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
	
	for (int i = threadIdX; i<effBlockSizeX; i += blockDim.x)
		for (int j = threadIdY; j < effBlockSizeY; j += blockDim.y) {
			if (i >= noIters && j >= noIters && i < effBlockSizeX - noIters && j < effBlockSizeX - noIters) {
				int currElemSM = i*effBlockSizeX + j;
				int currElemMain = (i - noIters + bOy)*sizeX + (j - noIters + bOx);
				potential[currElemMain] = sharedMem[currElemSM + 2* sharedMemSize];
			}
		}

}



void createDiskInitialCharge(double* input, float totalCharge, int sizeX,int sizeY, int xCen, int yCen, int radius){
	double chargePerPoint = totalCharge/(PI*radius*radius);
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

void initializeProblem(double* grid,double* potential,double totalCharge,int sizeX,int sizeY,int sizeXCen,int sizeYCen,int discRadius){

	for(int i=0;i<sizeX*sizeY;i++)
		potential[i]=0.0;

	createDiskInitialCharge(grid, totalCharge, sizeX, sizeY, sizeXCen,sizeYCen, discRadius);
}

void calculateElectricField(double* potential,double* field,int sizeX,int sizeY,double scale){

	for(int i=0;i<sizeX;i++)
		for(int j=0;j<sizeY;j++){
			int currPos = (i*sizeY+j);

			double currValPlusY=0;
			double currValPlusX=0;
			double currVal=potential[currPos];
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



void writeResultToFile(double* field,char* filename,int totalSize, int multiplicity){
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
	double scale = 0.01; // in m
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
	printf("Size : %d\n",sizeX);
	printf("No Iterations : %d\n",noIters);
	printf("No Threads : %d\n",noExtra);

	int sizeXCen=sizeX/2;
	int sizeYCen=sizeY/2;
	int totalSize=sizeX*sizeY;

    double *grid = (double *)malloc(totalSize * sizeof(double));

    double* potential = (double *)malloc(totalSize * sizeof(double));

    printf("Begin..\n");

	float totalCharge = 250000;
	float timeTaken = 0.0;

	initializeProblem(grid,potential, totalCharge, sizeX, sizeY, sizeXCen,sizeYCen, discRadius);

    printf("Problem Initialized..\n");

    double tolerance = 0.001;

    double *d_grid, *d_potential;
	cudaMalloc(&d_grid, totalSize*sizeof(double));
	cudaMalloc(&d_potential, totalSize * sizeof(double));

	cudaEvent_t startEventOrig_inc, stopEventOrig_inc;
	cudaEventCreate(&startEventOrig_inc);
	cudaEventCreate(&stopEventOrig_inc);
	cudaEventRecord(startEventOrig_inc, 0);
	
	cudaMemcpy(d_grid, grid, totalSize * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_potential, potential, totalSize * sizeof(double), cudaMemcpyHostToDevice);
    
	int threadPerBlock=32;
	
	dim3 blockSize(threadPerBlock,threadPerBlock);
	
	int noBlocks = ceil(sizeX/32);
	
	dim3 gridSize(noBlocks,noBlocks);
	printf("No. of Blocks %d\n", noBlocks);
	int sharedMemSize = 3*(threadPerBlock+2*noExtra)*(threadPerBlock+2*noExtra)*sizeof(double);
	
	int totalIters = noIters/noExtra+1;
	
	printf("Total iters : %d\n", totalIters);

	for (int k = 0; k < totalIters; k++) {
		jacobiMethod << <gridSize, blockSize, sharedMemSize >> >(d_grid, d_potential, sizeX, sizeY, scale, noExtra, tolerance);
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess)
			printf(cudaGetErrorString(err));
	}
    	
    
    cudaMemcpy(potential, d_potential, totalSize * sizeof(double), cudaMemcpyDeviceToHost);    
        
	printf("Jacobi Method Ended\n");
	float charge=0.0;
	for(int i=0;i<sizeX;i++)
		for(int j=0;j<sizeY;j++){
			int currPos = i*sizeY+j;
			charge+=grid[currPos];
			//printf("%f\n",potential[currPos]);
		}

	//Calculating Field
	double* field = (double*)malloc(sizeX*sizeY*2*sizeof(double));
	printf("Total charge is %f\n",charge);
//    fflush(stdout);

	calculateElectricField(potential,field,sizeX,sizeY,scale);
	
	cudaEventRecord(stopEventOrig_inc, 0);  //ending timing for inclusive
	cudaEventSynchronize(stopEventOrig_inc);
	cudaEventElapsedTime(&timeTaken, startEventOrig_inc, stopEventOrig_inc);

	printf("Time taken %f\n",timeTaken);
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
