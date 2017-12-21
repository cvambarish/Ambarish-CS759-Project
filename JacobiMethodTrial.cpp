//============================================================================
// Name        : JacobiMethodTrial.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdlib.h>
#include <math.h>
using namespace std;


/*\
 *
 *
 * x1 + x2 = 4
 * x1 - x2 = 2
 *
 *
 * x1_K+1 = 4 - x2_K
 * x2_K+1 = -2 + x1_K
 *
 */

float* jacobiMethod(float* initialGuess,int size,int noIters,float tolerance){
	float* currSolution = (float*)malloc(size * sizeof(float));
	//int size = sizeof(initialGuess)/sizeof(float);
	float equation[4][4] = {{3,1,2,1},{1,-3,1,1},{2,3,8,1},{1,4,1,8}};
	float equate[] = {4,2,1,1};

	//printf("Size is %d\n",size);
	for(int k=0;k<noIters;k++){
		for(int i=0;i<size;i++){
			float currSum=equate[i];
			for(int j=0;j<size;j++){
				if(j!=i){
					currSum-=(equation[i][j]*initialGuess[j]);
				}
			}
			currSolution[i]=currSum/equation[i][i];
			//printf("%f\n",currSolution[i]);

		}
		int converged=1;
		//printf("Tolerance is %f\n",tolerance);
		for(int i=0;i<size;i++){
			if(fabs(initialGuess[i]-currSolution[i]) > tolerance ){
				converged=0;
			}
			//printf("Difference for %f variable\n",fabs(initialGuess[i]-currSolution[i]));
			initialGuess[i]=currSolution[i];
		}
		//printf("In %d iteration, Converged is %d\n",k,converged);
		if(converged==1 && k > 5){
			printf("Converged after %d iterations\n",k);
			return currSolution;
		}

	}

//	for(int k=0;k<noIters;k++){
//		currSolution[0] = 4 - initialGuess[1];
//		currSolution[1] =-2 + initialGuess[0];
//		initialGuess[0] = currSolution[0];
//		initialGuess[1] = currSolution[1];
//
//	}

	return currSolution;
}


float* gaussSeidel(float* initialGuess,int size,int noIters,float tolerance){

	float* currSolution = (float*)malloc(size*sizeof(float));
	//int size = sizeof(initialGuess)/sizeof(int);


	float equation[4][4] = {{3,1,2,1},{1,-3,1,1},{2,3,8,1},{1,4,1,8}};
	float equate[] = {4,2,1,1};
	for(int k=0;k<noIters;k++){
		for(int i=0;i<size;i++){
			float currElem=equation[i][i];
			float currSum=equate[i];
			for(int j=0;j<size;j++){
				if(j<i){
					currSum-=(equation[i][j]*currSolution[j]);
				}
				else if(j>i){
					currSum-=(equation[i][j]*initialGuess[j]);
				}
			}
//			printf("%f\n",currSum);
			currSum/=currElem;
			currSolution[i]=currSum;
		}
		int converged=1;
		for(int i=0;i<size;i++){
			if(fabs(initialGuess[i]-currSolution[i]) > tolerance && converged==1){
				converged=0;
			}
			initialGuess[i]=currSolution[i];
		}
		if(converged==1 && k > 5){
			printf("Converged after %d iterations\n",k);
			return currSolution;
		}
	}




	return currSolution;
}



int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!

	float initialSolution[] = {1,0,10,1};
	int size=4;


	float* solution = jacobiMethod(initialSolution,4,100,0.001);
	float* solution2 = gaussSeidel(initialSolution,4,100,0.001);

	for(int i=0;i<size;i++){
		printf("%f\n",solution[i]);
	}

	for(int i=0;i<size;i++){
		printf("%f\n",solution2[i]);
	}

	return 0;
}
