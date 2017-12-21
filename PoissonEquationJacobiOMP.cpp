//============================================================================
// Name        : PoissonEquationJacobiOMP.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
using namespace std;

const double PI = 4*atan(1);

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

#pragma omp parallel for
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
			// x - derivative
//			if(i<sizeX-1){
//				field[2*currPos] = (potential[currPos+sizeY]-potential[currPos])/(scale);
//				printf("%f\n",potential[currPos+sizeY]);
//				printf("%f\n",potential[currPos]);
//			}
//			else{
//				field[2*currPos]=0;
//				field[2*currPos+1]=0;
//				continue;
//			}
//			// y - derivative
//			if(j<sizeY-1){
//				field[2*currPos+1] = (potential[currPos+1]-potential[currPos])/(scale);
//			}
//			else{
//				field[2*currPos]=0;
//				field[2*currPos+1]=0;
//			}
		}
}


void jacobiMethod(double* grid,double* potential, int sizeX,int sizeY,double scale,int noIters,double tolerance){

	double* currSolution = (double*)malloc(sizeX*sizeY*sizeof(double));


	int totalSize=sizeX*sizeY;

	for(int k=0;k<noIters;k++){
		for(int i=0;i<sizeX;i++)
			for(int j=0;j<sizeY;j++){
				int currPos = i*sizeY+j;
				currSolution[currPos]=0;

				if(i>1){
					currSolution[currPos]+=(potential[currPos-sizeY]/4);
				}
				if(i<sizeX-1){
					currSolution[currPos]+=(potential[currPos+sizeY]/4);
				}
				if(j>1){
					currSolution[currPos]+=(potential[currPos-1]/4);
				}
				if(j<sizeY-1){
					currSolution[currPos]+=(potential[currPos+1]/4);
				}

				if(i==sizeX-1||j==sizeY-1){
					//currSolution[currPos]=0;
				}else{
					currSolution[currPos]+=(scale*scale/4*grid[currPos]);
				}


			}


		int converged=1;
		for(int i=0;i<sizeX;i++)
			for(int j=0;j<sizeY;j++){
				int currPos = i*sizeY+j;
				if(fabs(currSolution[currPos]-potential[currPos]) > tolerance && converged==1){
					converged=0;
				}
				potential[currPos]=currSolution[currPos];
			}
		if(converged==1 && k > 5){
			printf("Converged after %d iterations\n",k);
			return;
		}

	}

	free(currSolution);
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


int main() {
	int sizeX=100;
	int sizeY=100;
	int sizeXCen=sizeX/2;
	int sizeYCen=sizeY/2;
	int totalSize=sizeX*sizeY;

    double *grid = (double *)malloc(totalSize * sizeof(double));

    double* potential = (double *)malloc(totalSize * sizeof(double));

    printf("Begin..\n");

	float totalCharge = 250000;

	float actualRadius = 0.05; // in m
    double scale = 0.01; // in m

	int discRadius = actualRadius/scale;
    initializeProblem(grid,potential, totalCharge, sizeX, sizeY, sizeXCen,sizeYCen, discRadius);

    printf("Problem Initialized..\n");
    fflush(stdout);
    int noIters = 10000;

    double tolerance = 0.001;

	jacobiMethod(grid,potential,sizeX,sizeY,scale,noIters,tolerance);
	printf("Jacobi Method Ended\n");
	float charge=0.0;
    fflush(stdout);
	for(int i=0;i<sizeX;i++)
		for(int j=0;j<sizeY;j++){
			int currPos = i*sizeY+j;
			charge+=grid[currPos];
			//printf("%f\n",potential[currPos]);
		}

	//Calculating Field
	double* field = (double*)malloc(sizeX*sizeY*2*sizeof(double));
	printf("Total charge is %f\n",charge);
    fflush(stdout);

	calculateElectricField(potential,field,sizeX,sizeY,scale);

	char* outputFilenameField = "ElectricField.out";
	char* outputFilenamePotential = "Potential.out";
	char* outputFilenameGrid = "Grid.out";
	printf("Writing to file..\n");
    fflush(stdout);
	writeResultToFile(field,outputFilenameField,totalSize,2);
	writeResultToFile(potential,outputFilenamePotential,totalSize,1);
	writeResultToFile(grid,outputFilenameGrid,totalSize,1);
	printf("All Done..\n");
	return 0;
}
