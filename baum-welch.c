#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include "tsc_x86.h"

//printing the data of an double array
//R is number of rows
//C is number of columns
void print_matrix(const double * const  a, const int R, const int C){
	for(int i =0; i < R; i++){
		for(int j = 0; j < C-1; j++){
			printf("%lf, ",a[i*C + j]);
		}
		printf("%lf \n", a[i*C + C-1]);
	}
	printf("\n");
}

//printing the data of a double vector
//L is lenght of vector
void print_vector(const double * const a, const int L){
	print_matrix(a,L,1);
}

//printing the data of a integer vector
//L is lenght of vector
void print_vector_int(const int * const a, const int L){
	for(int j = 0; j < L-1; j++){
		printf("%i, ",a[j]);
	}
	printf("%i \n", a[L-1]);
	
	printf("\n");
}

//generate a random number and return the index...
//where the sum of the probabilities up to this index of the vector...
//is bigger than the random number
//choices is the lenght of the vector
int chooseOf(const int choices, const double* const probArray){
	//decide at which proba to stop
	double decider= (double)rand()/(double)RAND_MAX;
	//printf("%lf \n",decider);
	double probSum=0;
	for(int i=0; i<choices;i++){
		//if decider in range(probSum[t-1],probSum[t])->return t
		probSum+=probArray[i];
		if (decider<=probSum)
		{
			return i;
		}
	}
	//some rounding error
	printf("%f",probSum);
	printf("The probabilites were not enough...");
	exit(-1);
}

//Generate random observations 
void makeObservations(const int hiddenStates, const int differentObservables, const int groundInitialState, const double* const groundTransitionMatrix, const double* const groundObservationMatrix, const  const int T, int* const observations){

	int currentState=groundInitialState;
	for(int i=0; i<T;i++){
		//this ordering of observations and current state, because first state is not necessarily affected by transitionMatrix
		//write down observation, based on occurenceMatrix of currentState
		observations[i]=chooseOf(differentObservables,groundObservationMatrix+currentState*differentObservables);
		//choose next State, given transitionMatrix of currentState
	currentState=chooseOf(hiddenStates,groundTransitionMatrix+currentState*hiddenStates);

	}
}

//make a vector with random probabilities such that all probabilities sum up to 1
//options is the lenght of the vector
void makeProbabilities(double* const probabilities, const int options){
	
	//ratio between smallest and highest probability
	const double ratio = 100;

	double totalProbabilites=0;
	for (int i=0; i<options;i++){

		double currentValue= (double)rand()/(double)(RAND_MAX) * ratio;
		probabilities[i]=currentValue;
		totalProbabilites+=currentValue;
	}

	for (int i=0; i<options;i++){
		probabilities[i]=probabilities[i]/totalProbabilites;
	}

}

//make a Matrix with random entries such that each row sums up to 1
//dim1 is number of rows
//dim2 is number of columns
void makeMatrix(const int dim1,const int dim2, double* const matrix){

	for (int row=0;row<dim1;row++){
		//make probabilites for one row
		makeProbabilities(matrix + row*dim2,dim2);
	}
}


//Luca
void forward(const double* const a, const double* const p, const double* const b, double* const alpha,  const int * const y, const int N, const int K, const int T){


	/* OPTIMIZATION AND SIMD
		SIMPLE APPROACH		
		precompute a matrix for  b[s*K + y[t]] => B_expl
		1. loop: 
			can use _mm256_mul_pd
		2. loop: 
			changing the order of the outer loops should be beneficial,
			because then alpha is accessed in row major order (except in inner most loop)
			for SIMD we can increase the stepsize of loop iterator t to 4
			replace alpha[s*T+t]=0 with __m256d a = _mms256_setzero_pd()
			inner most loop:
				perfect FMA
				a = _mm256_fma_pd(alpha_col,a_col)
				Here the access pattern is a problem because both arguments
				in the fma are accessed column wise order.
			replace alpha[s*T + t] *= b[s*K + y[t]]; with a = _mm256_mul_pd(a,b)
			where _m256d b = _mm256_load_pd(B_expl + s*K + t)

		IMPROVED APPROACH
		Currently I'm not sure if this is safe but I think we can
		change the loop order to j,s,t => check with Jan similar function
		Then in the current inner most loop:
			matrix a would be accesed in row wise order instead of colum wise order
			both alphas would be accessed in row wise order
		At no point in the loop hierarchy we would access in column wise order
		The SIMD instructions remain as before.
	
	*/

	for(int s = 0; s < N; s++){
		//printf("%lf %lf %lf \n", alpha[s*N], p[s], b[s*K]);
		alpha[s*T] = p[s] * b[s*K + y[0]];
	}

	//print_matrix(alpha,N,T);

	for(int t = 1; t < T; t++){
		for(int s = 0; s<N; s++){
			alpha[s*T + t] = 0;
			for(int j = 0; j < N; j++){

				alpha[s*T + t] += alpha[j*T + t-1] * a[j*N + s]; 
				//printf("%lf %lf %lf %lf \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t]]);
			}

			alpha[s*T + t] *= b[s*K + y[t]];
			//print_matrix(alpha,N,T);
		}
	}

	//print_matrix(alpha,N,T);

	return;
}

//Ang
void backward(const double* const a, const double* const e, double* const beta, const int N, const int K, const int T ){
	return;
}

void update(const double* const a, const double* const e, const double* const alpha, const double * const beta, double * const xi, const int N, const int K, const int T){
	return;
}

//Jan
int finished(){
	return 1;
}



//Jan
int similar(const double * const a, const double * const b ){
	return 1;
}


void wikipedia_example(){

	int hiddenStates = 2;
	int differentObservables = 2;
	int T = 10;


	//the observations we made
	int* observations = (int*) malloc ( T * sizeof(int));
	

	//the matrices which should approximate the ground truth
	double* transitionMatrix = (double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* emissionMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));


	transitionMatrix[0] = 0.5;
	transitionMatrix[1] = 0.5;
	transitionMatrix[2] = 0.3;
	transitionMatrix[3] = 0.7;

	emissionMatrix[0] = 0.3;
	emissionMatrix[1] = 0.7;
	emissionMatrix[2] = 0.8;
	emissionMatrix[3] = 0.2;

	observations[0] = 0;
	observations[1] = 0;
	observations[2] = 0;
	observations[3] = 0;
	observations[4] = 0;
	observations[5] = 1;
	observations[6] = 1;
	observations[7] = 0;
	observations[8] = 0;
	observations[9] = 0;

	//init state distribution
	double* piMatrix  = (double*) malloc(hiddenStates * sizeof(double));

	piMatrix[0] = 0.2;
	piMatrix[1] = 0.8;
	
	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * T * sizeof(double));

	forward(transitionMatrix, piMatrix, emissionMatrix, alpha, observations, hiddenStates, differentObservables, T);	
	backward(transitionMatrix, emissionMatrix, beta, hiddenStates, differentObservables, T);	//Ang
	update(transitionMatrix, emissionMatrix, alpha,beta, xi, hiddenStates, differentObservables, T);//???
	
	/*
	printf("new transition matrix from wikipedia example: \n \n");
	print_matrix(transitionMatrix,hiddenStates,hiddenStates);
	*/

}

int main(int argc, char *argv[]){

	//wikipedia_example();

	if(argc != 5){
		printf("USAGE: ./run <seed> <hiddenStates> <observables> <T> \n");
		return -1;
	}

	const int maxRuns=10;
	const int seed = atoi(argv[1]);  
	const int hiddenStates = atoi(argv[2]); 
	const int differentObservables = atoi(argv[3]); 
	const int T = atoi(argv[4]); 

	printf("Parameters: \n");
	printf("seed = %i \n", seed);
	printf("hidden States = %i \n", hiddenStates);
	printf("different Observables = %i \n", differentObservables);
	printf("number of observations= %i \n", T);
	printf("\n");

	myInt64 cycles;
    	myInt64 start;

	//set random according to seed
	srand(seed);

	//the ground TRUTH we want to approximate:
	double* groundTransitionMatrix = (double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* groundObservationMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));

	//set ground truth to some random values
	makeMatrix(hiddenStates, hiddenStates, groundTransitionMatrix);
	makeMatrix(hiddenStates, differentObservables, groundObservationMatrix);
	
	//the observations we made
	int* observations = (int*) malloc ( T * sizeof(int));

	//the matrices which should approximate the ground truth
	double* transitionMatrix = (double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* emissionMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));

	//init state distribution
	double* piVector  = (double*) malloc(hiddenStates * sizeof(double));

	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * (T-1) * sizeof(double));

	for (int run=0; run<maxRuns; run++){

		start = start_tsc();

		//init transition Matrix, emission Matrix and initial state distribution random
		makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
		makeMatrix(hiddenStates, differentObservables, emissionMatrix);	
		makeProbabilities(piVector,hiddenStates);
		
		/* for debugging
		printf("transition Matrix \n");
		print_matrix(transitionMatrix, hiddenStates, hiddenStates);

		printf("emission Matrix \n");
		print_matrix(emissionMatrix, hiddenStates, differentObservables);
		
		printf("PI Matrix \n");
		print_vector(piVector, hiddenStates);
		*/

		//make some random observations
		int groundInitialState = rand()%hiddenStates;
		makeObservations(hiddenStates, differentObservables, groundInitialState, groundTransitionMatrix,groundObservationMatrix,T, observations);
		
		/*for debugging
		printf("observations \n");
		print_vector_int(observations,T);
		*/
	

		// CHANGE TO WHILE LOOP. do while only for testing
		do{
			//observations=forward(observations, transitionMatrix, observationMatrix); //Luca
			forward(transitionMatrix, piVector, emissionMatrix, alpha, observations, hiddenStates, differentObservables, T);	//Luca
			backward(transitionMatrix, emissionMatrix, beta, hiddenStates, differentObservables, T);	//Ang
			update(transitionMatrix, emissionMatrix, alpha,beta, xi, hiddenStates, differentObservables, T);//???

		}while (!finished());//Jan

		cycles = stop_tsc(start);

		//Jan
		if (similar(groundTransitionMatrix,transitionMatrix) && similar(groundObservationMatrix,emissionMatrix)){
			printf("run %i: \t %llu cycles \n",run, cycles);
		}else{
			free(groundTransitionMatrix);
			free(groundObservationMatrix);
			free(observations);
			free(transitionMatrix);
			free(emissionMatrix);
			free(piVector);
			free(alpha);
			free(beta);
			free(gamma);
			free(xi);
			printf("Something went wrong! \n");
			return -1;//error Jan
		}


	}

	free(groundTransitionMatrix);
	free(groundObservationMatrix);
	free(observations);
	free(transitionMatrix);
	free(emissionMatrix);
	free(piVector);
	free(alpha);
	free(beta);
	free(gamma);
	free(xi);

	return 0; 
} 
