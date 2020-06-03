#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <float.h> //for DOUBL_MAX
#include "tsc_x86.h"
#include "util.h"

#include "io.h"
#include "tested.h"


double EPSILON = 1e-4;
#define DELTA 1e-2
#define BUFSIZE 1<<26   // ~60 MB



void forward(const double* const a, const double* const p, const double* const b, double* const alpha,  const int * const y, double* const ct, const int N, const int K, const int T){

	ct[0]=0.0;
	//compute alpha(0) and scaling factor for t = 0
	for(int s = 0; s < N; s++){
		alpha[s*T] = p[s] * b[s*K + y[0]];
		ct[0] += alpha[s*T];
		//printf("%lf %lf %lf \n", alpha[s*T], p[s], b[s*K+y[0]]);
	}
	
	//scaling factor for t = 0
	ct[0] = 1.0 / ct[0];

	//scale alpha(0)
	for(int s = 0; s < N; s++){
		alpha[s*T] *= ct[0];
		//printf("%lf %lf %lf \n", alpha[s*T], p[s], b[s*K+y[0]]);
	}
	//print_matrix(alpha,N,T);

	for(int t = 1; t < T; t++){
		ct[t]=0.0;
		for(int s = 0; s<N; s++){// s=new_state
			alpha[s*T + t] = 0;
			for(int j = 0; j < N; j++){//j=old_states

				alpha[s*T + t] += alpha[j*T + t-1] * a[j*N + s];
				//printf("%lf %lf %lf %lf %i \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t+1]],y[t]);
			}

			alpha[s*T + t] *= b[s*K + y[t]];
			//print_matrix(alpha,N,T);
			ct[t] += alpha[s*T + t];
		}
		//scaling factor for t 
		ct[t] = 1.0 / ct[t];
		
		//scale alpha(t)
		for(int s = 0; s<N; s++){// s=new_state
			alpha[s*T + t] *= ct[t];
		}
		
	}
	//print_matrix(alpha,N,T);

	return;
}


void backward(const double* const a, const double* const b, double* const beta, const int * const y, const double * const ct, const int N, const int K, const int T ){
	for(int s = 1; s < N+1; s++){
		beta[s*T-1] = /* 1* */ct[T-1];
	}

	for(int t = T-1; t > 0; t--){
		for(int s = 0; s < N; s++){//s=older state
       			beta[s*T + t-1] = 0.;
			for(int j = 0; j < N; j++){//j=newer state
				beta[s*T + t-1] += beta[j*T + t ] * a[s*N + j] * b[j*K + y[t]];
				//printf("%lf %lf %lf %lf %i \n", beta[s*T + t-1], beta[j*T+t], a[s*N+j], b[j*K+y[t]],y[t]);
			}
			beta[s*T + t-1] *= ct[t-1];
		}
	}
	//print_matrix(beta,N,T);
	return;
}

void update(double* const a, double* const p, double* const b, const double* const alpha, const double* const beta, double* const gamma, double* const xi, const int* const y, const double* const ct,const int N, const int K, const int T){


	double xi_sum, gamma_sum_numerator, gamma_sum_denominator;

	for(int t = 0; t < T; t++){
		for(int s = 0; s < N; s++){ // s old state
			gamma[s*T + t] = alpha[s*T + t] * beta[s*T + t];
		}
	}

	for(int t = 1; t < T; t++){
		for(int s = 0; s < N; s++){
			for(int j = 0; j < N; j++){ // j new state
				xi[((t-1) * N + s) * N + j] = alpha[s*T + t-1] * a[s*N + j] * beta[j*T + t] * b[j*K + y[t]]; 
			}
		}
	}


	/*
	//Only here to show that the evidence is the same as the result of alternative computations
	//evidence for xi
	double cT = 1.0;
	for(int time = 0; time < T; time++){
		cT *=ct[time];
	}

	for(int t = 1; t < T; t++){
		double evidence=0;	

		for(int s = 0; s < N; s++){
			for (int nextState=0; nextState < N; nextState++){
				evidence+=alpha[s*T+t-1]*a[s*N+nextState]*beta[nextState*T+t]*b[nextState*K+y[t]];
			}
		}

		printf("evidence for XI at time %i: %.10lf \n", t,evidence/cT);
	}
		
	*/

	for(int s = 0; s < N; s++){
		// new pi
		//XXX the next line is not in the r library hmm.
    		p[s] = gamma[s*T]/ct[0];
    
		for(int j = 0; j < N; j++){
			xi_sum = 0.;
			gamma_sum_denominator = 0.;
			for(int t = 1; t < T; t++){
				xi_sum += xi[((t-1) * N + s) * N + j];
				gamma_sum_denominator += gamma[s*T + t-1]/ct[t-1];
			}
			// new transition matrix
			a[s*N + j] = xi_sum / gamma_sum_denominator;
		}

		gamma_sum_denominator += gamma[s*T + T-1]/ct[T-1];

		for(int v = 0; v < K; v++){
			gamma_sum_numerator = 0.;
			for(int t = 0; t < T; t++){
				if(y[t] == v){
					gamma_sum_numerator += gamma[s*T + t]/ct[t];
				}
			}
			// new emmision matrix
			b[s*K + v] = gamma_sum_numerator / gamma_sum_denominator;
		}
	}

	return;
}



void heatup(double* const transitionMatrix,double* const piVector,double* const emissionMatrix,const int* const observations,const int hiddenStates,const int differentObservables,const int T){

	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * T * sizeof(double));
	double* ct = (double*) malloc(T * sizeof(double));
	
	for(int j=0;j<10;j++){
		forward(transitionMatrix, piVector, emissionMatrix, alpha, observations, ct, hiddenStates, differentObservables, T);	
		backward(transitionMatrix, emissionMatrix, beta, observations, ct, hiddenStates, differentObservables, T);
		update(transitionMatrix, piVector, emissionMatrix, alpha, beta, gamma, xi, observations, ct, hiddenStates, differentObservables, T);
	}	

	free(alpha);
	free(beta);
	free(gamma);
	free(xi);
   	free(ct);
	
}


int main(int argc, char *argv[]){


	if(argc < 5){
		printf("USAGE: ./run <seed> <hiddenStates> <observables> <T> \n");
		return -1;
	}

	const int seed = atoi(argv[1]);  
	const int hiddenStates = atoi(argv[2]); 
	const int differentObservables = atoi(argv[3]); 
	const int T = atoi(argv[4]);
	
	if(argc ==6){
		int exp = atoi(argv[5]);
		EPSILON  = pow(10,-exp);
	}
   	
   	/*
	printf("Parameters: \n");
	printf("seed = %i \n", seed);
	printf("hidden States = %i \n", hiddenStates);
	printf("different Observables = %i \n", differentObservables);
	printf("number of observations= %i \n", T);
	printf("\n");
    	*/
	
	myInt64 cycles;
   	myInt64 start;
    	int minima=10;
	int variableSteps=100-cbrt(hiddenStates*differentObservables*T)/3;
	int maxSteps=minima < variableSteps ? variableSteps : minima;
	minima=1;    
	variableSteps=10-log10(hiddenStates*differentObservables*T);
	int maxRuns=minima < variableSteps ? variableSteps : minima;
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
	double* transitionMatrixSafe = (double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* transitionMatrixTesting=(double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* emissionMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));
	double* emissionMatrixSafe = (double*) malloc(hiddenStates*differentObservables*sizeof(double));
	double* emissionMatrixTesting=(double*) malloc(hiddenStates*differentObservables*sizeof(double));

	//init state distribution
	double* stateProb  = (double*) malloc(hiddenStates * sizeof(double));
	double* stateProbSafe  = (double*) malloc(hiddenStates * sizeof(double));
	double* stateProbTesting  = (double*) malloc(hiddenStates * sizeof(double));

	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * (T-1) * sizeof(double)); 

	double* ct = (double*) malloc(T*sizeof(double));
	
	//Generate matrices
	makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
	makeMatrix(hiddenStates, differentObservables, emissionMatrix);
	makeProbabilities(stateProb,hiddenStates);

	//make a copy of matrices to be able to reset matrices after each run to initial state and to be able to test implementation.
	memcpy(transitionMatrixSafe, transitionMatrix, hiddenStates*hiddenStates*sizeof(double));
   	memcpy(emissionMatrixSafe, emissionMatrix, hiddenStates*differentObservables*sizeof(double));
      	memcpy(stateProbSafe, stateProb, hiddenStates * sizeof(double));


	//heatup(transitionMatrix,stateProb,emissionMatrix,observations,hiddenStates,differentObservables,T);
	
	volatile unsigned char* buf = malloc(BUFSIZE*sizeof(char));
        int steps = 0;
	
	for (int run=0; run<maxRuns; run++){

		//init transition Matrix, emission Matrix and initial state distribution random
       	memcpy(transitionMatrix, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
	   	memcpy(emissionMatrix, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
        	memcpy(stateProb, stateProbSafe, hiddenStates * sizeof(double));	
	
        	double logLikelihood=-DBL_MAX; //Took down here.

		//only needed for testing with R
		//write_init(transitionMatrix, emissionMatrix, observations, stateProb, hiddenStates, differentObservables, T);
        
        	steps=0;
	        _flush_cache(buf,BUFSIZE); // ensure the cache is cold
		start = start_tsc();

		do{
			forward(transitionMatrix, stateProb, emissionMatrix, alpha, observations, ct, hiddenStates, differentObservables, T);	
			backward(transitionMatrix, emissionMatrix, beta,observations, ct, hiddenStates, differentObservables, T);
			update(transitionMatrix, stateProb, emissionMatrix, alpha, beta, gamma, xi, observations, ct, hiddenStates, differentObservables, T);
            		steps+=1;
            		
		}while (!finished(ct, &logLikelihood, hiddenStates, T,EPSILON) && steps<maxSteps);

		cycles = stop_tsc(start);
        	cycles = cycles/steps;

		runs[run]=cycles;



	}

	qsort (runs, maxRuns, sizeof (double), compare_doubles);
  	double medianTime = runs[maxRuns/2];
	printf("Median Time: \t %lf cycles \n", medianTime); 

	//write_result(transitionMatrix, emissionMatrix, observations, stateProb, steps, hiddenStates, differentObservables, T);

	//used for testing
	memcpy(transitionMatrixTesting, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
	memcpy(emissionMatrixTesting, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
	memcpy(stateProbTesting, stateProbSafe, hiddenStates * sizeof(double));

	tested_implementation(hiddenStates, differentObservables, T, transitionMatrixTesting, emissionMatrixTesting, stateProbTesting, observations,EPSILON,DELTA);
	
	/*
	//Show results
	print_matrix(transitionMatrix,hiddenStates,hiddenStates);
	print_matrix(emissionMatrix, hiddenStates,differentObservables);
	print_vector(stateProb, hiddenStates);
	
	//Show tested results
	printf("tested \n");
	print_matrix(transitionMatrixTesting,hiddenStates,hiddenStates);
	print_matrix(emissionMatrixTesting, hiddenStates,differentObservables);
	print_vector(stateProbTesting, hiddenStates);
	*/
 
	if (!similar(transitionMatrixTesting,transitionMatrix,hiddenStates,hiddenStates,DELTA) && similar(emissionMatrixTesting,emissionMatrix,differentObservables,hiddenStates,DELTA)){
		printf("Something went wrong !");	
		
	}
	
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
   	free(ct);
    	free(transitionMatrixSafe);
	free(emissionMatrixSafe);
   	free(stateProbSafe);
	free(transitionMatrixTesting);
	free(emissionMatrixTesting);
	free(stateProbTesting);
	free((void*)buf);

	return 0; 
} 
