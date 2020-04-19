#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include "tsc_x86.h"

#include "io.h"

#define EPSILON 0.001
#define DELTA 0.001


void set_zero(double* const a, const int rows, const int cols){
	for(int row = 0 ; row < rows; row++){
		for(int col = 0; col < cols; col++){
			a[row * cols + col] = 0.0;
		}
	}
}

int compare_doubles (const void *a, const void *b) //for sorting at the end
{
  const double *da = (const double *) a;
  const double *db = (const double *) b;

  return (*da > *db) - (*da < *db);
}
//generate a random number [0,1] and return the index...
//where the sum of the probabilities up to this index of the vector...
//is bigger than the random number
//argument choices is the lenght of the vector
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
void makeObservations(const int hiddenStates, const int differentObservables, const int groundInitialState, const double* const groundTransitionMatrix, const double* const groundEmissionMatrix, const int T, int* const observations){

	int currentState=groundInitialState;
	for(int i=0; i<T;i++){
		//this ordering of observations and current state, because first state is not necessarily affected by transitionMatrix
		//write down observation, based on occurenceMatrix of currentState
		observations[i]=chooseOf(differentObservables,groundEmissionMatrix+currentState*differentObservables);
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
		alpha[s*T] = p[s] * b[s*K + y[0]];
		//printf("%lf %lf %lf \n", alpha[s*T], p[s], b[s*K+y[0]]);
	}

	//print_matrix(alpha,N,T);

	for(int t = 1; t < T; t++){
		for(int s = 0; s<N; s++){// s=new_state
			alpha[s*T + t] = 0;
			for(int j = 0; j < N; j++){//j=old_states

				alpha[s*T + t] += alpha[j*T + t-1] * a[j*N + s];
				//printf("%lf %lf %lf %lf %i \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t+1]],y[t]);
			}

			alpha[s*T + t] *= b[s*K + y[t]];
			//print_matrix(alpha,N,T);
		}
	}

	//print_matrix(alpha,N,T);

	return;
}

//Ang
void backward(const double* const a, const double* const b, double* const beta, const int * const y, const int N, const int K, const int T ){
	for(int s = 1; s < N+1; s++){
		beta[s*T-1] = 1.;
	}

	for(int t = T-1; t > 0; t--){
		for(int s = 0; s < N; s++){//s=older state
       			beta[s*T + t-1] = 0.;
			for(int j = 0; j < N; j++){//j=newer state
				beta[s*T + t-1] += beta[j*T + t ] * a[s*N + j] * b[j*K + y[t]];
				//printf("%lf %lf %lf %lf %i \n", beta[s*T + t-1], beta[j*T+t], a[s*N+j], b[j*K+y[t]],y[t]);
			}
		}
	}
	//print_matrix(beta,N,T);
	return;
}

void update(double* const a, double* const p, double* const b, const double* const alpha, const double* const beta, double* const gamma, double* const xi, const int* const y, const int N, const int K, const int T){
	double evidence, xi_sum, gamma_sum_numerator, gamma_sum_denominator;
	for(int t = 1; t < T; t++){
		evidence = 0.; // P(Y|theta)
		// denominator at time t
		for(int s = 0; s < N; s++){
			evidence += alpha[s*T + t-1] * beta[s*T + t-1];
		}

		for(int s = 0; s < N; s++){ // s old state
			gamma[s*T + t-1] = (alpha[s*T + t-1] * beta[s*T + t-1]) / evidence;
			for(int j = 0; j < N; j++){ // j new state
				xi[((t-1) * N + s) * N + j] = (alpha[s*T + t-1] * a[s*N + j] * beta[j*T + t] * b[j*K + y[t]]) / evidence;
			}
		}
	}

	for(int s = 0; s < N; s++){
		// new pi
    		p[s] = gamma[s*T];
    
		for(int j = 0; j < N; j++){
			xi_sum = 0.;
			gamma_sum_denominator = 0.;
			for(int t = 1; t < T; t++){
				xi_sum += xi[((t-1) * N + s) * N + j];
				gamma_sum_denominator += gamma[s*T + t-1];
			}
			// new transition matrix
			a[s*N + j] = xi_sum / gamma_sum_denominator;
		}

		gamma_sum_denominator += gamma[s*T + T-1];
		for(int v = 0; v < K; v++){
			gamma_sum_numerator = 0.;
			for(int t = 1; t <= T; t++){
				if(y[t] == v){// XXX rather AllPossibleValues[v] ???
					gamma_sum_numerator += gamma[s*T + t-1];
				}
			}
			// new emmision matrix
			b[s*K + v] = gamma_sum_numerator / gamma_sum_denominator;
		}
	}

	return;
}

//Jan
int finished(const double* const alpha,const double* const beta, double* const logLikelihood,const int N,const int T){
	double oldLogLikelihood=*logLikelihood;
	double newLogLikelihood= 0.0;
	for(int t = 1; t < T; t++){
        	for(int i = 0; i < N; i++){
            		newLogLikelihood += alpha[i*T + t-1] * beta[i*T + t-1];
        	}
	}

	newLogLikelihood=log(newLogLikelihood);
	*logLikelihood=newLogLikelihood;
	return (newLogLikelihood-oldLogLikelihood)<EPSILON;
}



//Jan
int similar(const double * const a, const double * const b , const int N, const int M){
	//Frobenius norm
	double sum=0.0;
	double abs=0.0;
	for(int i=0;i<N;i++){
		for(int j=0;j<M;j++){
			abs=a[i*M+j]-b[i*M+j];
			sum+=abs*abs;
		}
	}
	return sqrt(sum)<DELTA; 
}

void heatup(const double* transitionMatrix,const double* piVector,const double* emissionMatrix,const int* const observations,const int hiddenStates,const int differentObservables,const int T){
	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * T * sizeof(double));
	for(int j=0;j<10;j++){
		forward(transitionMatrix, piVector, emissionMatrix, alpha, observations, hiddenStates, differentObservables, T);	
		backward(transitionMatrix, emissionMatrix, beta, observations, hiddenStates, differentObservables, T);	//Ang
		update(transitionMatrix, piVector, emissionMatrix, alpha, beta, gamma, xi, observations, hiddenStates, differentObservables, T);//Ang
	};	
	
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
	double* piMatrix  = (double*) malloc(hiddenStates * sizeof(double));

	char tname[100]="wikipedia_matrices/transitionMatrix.csv";
	read_matrix_file(transitionMatrix,2,2,tname);	

	char ename[100]="wikipedia_matrices/emissionMatrix.csv";
	read_matrix_file(emissionMatrix,2,2,ename);	

	char oname[100]="wikipedia_matrices/observations.csv";
	read_vector_file_int(observations,T,oname);	

	char pname[100]="wikipedia_matrices/piMatrix.csv";
	read_vector_file(piMatrix,2,pname);	

	/*
	print_matrix(transitionMatrix,2,2);
	print_matrix(emissionMatrix,2,2);
	print_vector_int(observations,T);
	print_vector(piMatrix,2);
	*/

	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * T * sizeof(double));

	forward(transitionMatrix, piMatrix, emissionMatrix, alpha, observations, hiddenStates, differentObservables, T);	
	backward(transitionMatrix, emissionMatrix, beta, observations, hiddenStates, differentObservables, T);	//Ang
	update(transitionMatrix, piMatrix, emissionMatrix, alpha, beta, gamma, xi, observations, hiddenStates, differentObservables, T);//Ang
	
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
	double runs[maxRuns]; //for medianTime
	//set random according to seed
	srand(seed);

	//the ground TRUTH we want to approximate:
	double* groundTransitionMatrix = (double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* groundEmissionMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));

	//set ground truth to some random values
	makeMatrix(hiddenStates, hiddenStates, groundTransitionMatrix);
	makeMatrix(hiddenStates, differentObservables, groundEmissionMatrix);
	int groundInitialState = rand()%hiddenStates;
	
	//the observations we made
	int* observations = (int*) malloc ( T * sizeof(int));
	makeObservations(hiddenStates, differentObservables, groundInitialState, groundTransitionMatrix,groundEmissionMatrix,T, observations);//??? added
	
	//the matrices which should approximate the ground truth
	double* transitionMatrix = (double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* emissionMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));

	//init state distribution
	double* stateProb  = (double*) malloc(hiddenStates * sizeof(double));

	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * (T-1) * sizeof(double)); //??? Wieso T-1 ?
	
	double logLikelihood=0.0;

	//heatup needs some data.
	makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
	makeMatrix(hiddenStates, differentObservables, emissionMatrix);
	makeProbabilities(stateProb,hiddenStates);
	//heatup(transitionMatrix,stateProb,emissionMatrix,observations,hiddenStates,differentObservables,T);
	
	for (int run=0; run<maxRuns; run++){

		//XXX start after makeMatrix
		start = start_tsc();

		//init transition Matrix, emission Matrix and initial state distribution random
		makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
		makeMatrix(hiddenStates, differentObservables, emissionMatrix);	
		makeProbabilities(stateProb,hiddenStates);
		
		set_zero(alpha,hiddenStates,T);
		set_zero(beta,hiddenStates,T);
		set_zero(gamma,hiddenStates,T);
		set_zero(xi,hiddenStates*hiddenStates,T-1);



		//make some random observations
		int groundInitialState = rand()%hiddenStates;
		makeObservations(hiddenStates, differentObservables, groundInitialState, groundTransitionMatrix,groundEmissionMatrix,T, observations); //??? ground___ zu ___ wechseln?
		
			//can be use for debugging and for testing with other libraries
		//write all matrices to csv files
		/*		
		write_all(groundTransitionMatrix,
			groundEmissionMatrix,
			transitionMatrix,
			emissionMatrix,
			observations,
			stateProb,
			alpha,
			beta,
			gamma,
			xi,
			hiddenStates,
			differentObservables,
			T);		
		*/
		while (!finished(alpha, beta, &logLikelihood, hiddenStates, T)){
			forward(transitionMatrix, stateProb, emissionMatrix, alpha, observations, hiddenStates, differentObservables, T);	//Luca
			backward(transitionMatrix, emissionMatrix, beta,observations, hiddenStates, differentObservables, T);	//Ang
			update(transitionMatrix, stateProb, emissionMatrix, alpha, beta, gamma, xi, observations, hiddenStates, differentObservables, T);  //Ang

		}

		//print_matrix(alpha,hiddenStates,T);

		cycles = stop_tsc(start);

		//Jan
		if (similar(groundTransitionMatrix,transitionMatrix,hiddenStates,hiddenStates) && similar(groundEmissionMatrix,emissionMatrix,differentObservables,hiddenStates)){
			runs[run]=cycles;
			printf("run %i: \t %llu cycles \n",run, cycles);
		}else{	
		
			write_all(groundTransitionMatrix,
				groundEmissionMatrix,
				transitionMatrix,
				emissionMatrix,
				observations,
				stateProb,
				alpha,
				beta,
				gamma,
				xi,
				hiddenStates,
				differentObservables,
				T);		
		
			free(groundTransitionMatrix);
			free(groundEmissionMatrix);
			free(observations);
			free(transitionMatrix);
			free(emissionMatrix);
			free(stateProb);
			free(alpha);
			free(beta);
			free(gamma);
			free(xi);
			printf("Something went wrong! \n");
			return -1;//error Jan
		}


	}
	qsort (runs, maxRuns, sizeof (double), compare_doubles);
  	double medianTime = runs[maxRuns/2];
	printf("Median Time: \t %lf cycles \n", medianTime); 


	write_all(groundTransitionMatrix,
		groundEmissionMatrix,
		transitionMatrix,
		emissionMatrix,
		observations,
		stateProb,
		alpha,
		beta,
		gamma,
		xi,
		hiddenStates,
		differentObservables,
		T);		
		

	free(groundTransitionMatrix);
	free(groundEmissionMatrix);
	free(observations);
	free(transitionMatrix);
	free(emissionMatrix);
	free(stateProb);
	free(alpha);
	free(beta);
	free(gamma);
	free(xi);

	return 0; 
} 
