#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <float.h> //for DOUBL_MAX
#include "tsc_x86.h"

#include "io.h"
#include "tested.h"

#define EPSILON 1e-12
#define DELTA 0.2
#define maxSteps 100


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

//Ang
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
				//Unlike evidence, Xi has a and b under the line in Wikipedia. The notation "P(Y|theta)" on Wikipedia is misleading.
				//Discussion from 22.4.20 showed that this should be the same. Notation on wikipedia is consistent.
			}
		}
	}

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
			for(int t = 0; t < T; t++){//why 1 indented => better?
				if(y[t] == v){// XXX rather AllPossibleValues[v] ??? => don't understand the question. What is AllPossibleValues[v]?
					gamma_sum_numerator += gamma[s*T + t]/ct[t];//why different t here than in y[t] => I think this was a typo. Indeed it should be the same t for gamma and y.
				}
			}
			// new emmision matrix
			b[s*K + v] = gamma_sum_numerator / gamma_sum_denominator;
		}
	}

	return;
}



//Jan
int finished( const double* const ct, double* const l,const int N,const int T){
	
	//log likelihood
	double oldLogLikelihood=*l;

	double newLogLikelihood = 0.0;
	//evidence with alpha only:

	for(int time = 0; time < T; time++){
		newLogLikelihood -= log10(ct[time]);
	}
	
	*l=newLogLikelihood;

	//printf("log likelihood %.10lf , Epsilon %.10lf result %.10lf \n", newLogLikelihood, EPSILON,newLogLikelihood-oldLogLikelihood);
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
    //DEBUG off
	//printf("Frobenius norm = %.10lf delta = %.10lf\n", sqrt(sum), DELTA);
	return sqrt(sum)<DELTA; 
}

void heatup(double* const transitionMatrix,double* const piVector,double* const emissionMatrix,const int* const observations,const int hiddenStates,const int differentObservables,const int T){

	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * T * sizeof(double));
	double* ct = (double*) malloc(T * sizeof(double));
	
	for(int j=0;j<10;j++){
		forward(transitionMatrix, piVector, emissionMatrix, alpha, observations, ct, hiddenStates, differentObservables, T);	
		backward(transitionMatrix, emissionMatrix, beta, observations, ct, hiddenStates, differentObservables, T);	//Ang
		update(transitionMatrix, piVector, emissionMatrix, alpha, beta, gamma, xi, observations, ct, hiddenStates, differentObservables, T);//Ang
	}	
	
}
	

void baum_welch(const double* const a, const double* const b, const double* const p, const int* const y, double * const gamma_total, double* const gamma_T, double* const ct, const int N, const int K, const int T){


	//FORWARD

	double* alpha = (double*) malloc(N * T * sizeof(double));

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



	//FUSED BACKWARD and UPDATE STEP


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
				//Unlike evidence, Xi has a and b under the line in Wikipedia. The notation "P(Y|theta)" on Wikipedia is misleading.
				//Discussion from 22.4.20 showed that this should be the same. Notation on wikipedia is consistent.
			}
		}
	}

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
			for(int t = 0; t < T; t++){//why 1 indented => better?
				if(y[t] == v){// XXX rather AllPossibleValues[v] ??? => don't understand the question. What is AllPossibleValues[v]?
					gamma_sum_numerator += gamma[s*T + t]/ct[t];//why different t here than in y[t] => I think this was a typo. Indeed it should be the same t for gamma and y.
				}
			}
			// new emmision matrix
			b[s*K + v] = gamma_sum_numerator / gamma_sum_denominator;
		}
	}
	
	return;

}


int main(int argc, char *argv[]){


	if(argc != 5){
		printf("USAGE: ./run <seed> <hiddenStates> <observables> <T> \n");
		return -1;
	}

	const int maxRuns=10;
	const int seed = atoi(argv[1]);  
	const int hiddenStates = atoi(argv[2]); 
	const int differentObservables = atoi(argv[3]); 
	const int T = atoi(argv[4]); 
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
	double* emissionMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));
	double* emissionMatrixSafe = (double*) malloc(hiddenStates*differentObservables*sizeof(double));

	//init state distribution
	double* stateProb  = (double*) malloc(hiddenStates * sizeof(double));
	double* stateProbSafe  = (double*) malloc(hiddenStates * sizeof(double));

	//double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	//double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	//double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	//double* xi = (double*) malloc(hiddenStates * hiddenStates * (T-1) * sizeof(double)); 

	//sum of gamma over t for all states
	double* gamma_sum = (double*) malloc(hiddenStates * sizeof(double));
	//gamma for all states at time T
	double* gamma_T = (double*) malloc(hiddenStates * sizeof(double));

	double* ct = (double*) malloc(T*sizeof(double));

	//make some random observations
	makeObservations(hiddenStates, differentObservables, groundInitialState, groundTransitionMatrix,groundEmissionMatrix,T, observations); 

	
	//random init transition matrix, emission matrix and state probabilities.
	makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
	makeMatrix(hiddenStates, differentObservables, emissionMatrix);
	makeProbabilities(stateProb,hiddenStates);

	//make a copy of matrices to be able to reset matrices after each run to initial state and to be able to test implementation.
	memcpy(transitionMatrixSafe, transitionMatrix, hiddenStates*hiddenStates*sizeof(double));
   	memcpy(emissionMatrixSafe, emissionMatrix, hiddenStates*differentObservables*sizeof(double));
      	memcpy(stateProbSafe, stateProb, hiddenStates * sizeof(double));	

	//heat up cache
	heatup(transitionMatrix,stateProb,emissionMatrix,observations,hiddenStates,differentObservables,T);
	
        int steps=0;
	for (int run=0; run<1; run++){

		//init transition Matrix, emission Matrix and initial state distribution random
		memcpy(transitionMatrix, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
   		memcpy(emissionMatrix, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
      		memcpy(stateProb, stateProbSafe, hiddenStates * sizeof(double));

        	double logLikelihood=-DBL_MAX; //Took down here.

		//only needed for testing with R
		write_init(transitionMatrix, emissionMatrix, observations, stateProb, hiddenStates, differentObservables, T);
        
        	steps=0;
		start = start_tsc();
	
		do{
			//forward(transitionMatrix, stateProb, emissionMatrix, alpha, observations, ct, hiddenStates, differentObservables, T);	//Luca
			//backward(transitionMatrix, emissionMatrix, beta,observations, ct, hiddenStates, differentObservables, T);	//Ang
			//update(transitionMatrix, stateProb, emissionMatrix, alpha, beta, gamma, xi, observations, ct, hiddenStates, differentObservables, T);  //Ang
            		
			baum_welch(transitionMatrix,emissionMatrix, stateProb, observations, gamma_sum, gamma_T, ct, hiddenStates, differentObservables, T);	

			steps+=1;
		}while (!finished(ct, &logLikelihood, hiddenStates, T) && steps<maxSteps);

		cycles = stop_tsc(start);
        	cycles = cycles/steps;
		//Jan

		//DEBUGGING
		print_matrix(transitionMatrix,hiddenStates,hiddenStates);
		print_matrix(emissionMatrix, hiddenStates,differentObservables);


        	tested_implementation(hiddenStates, differentObservables, T, transitionMatrixSafe, emissionMatrixSafe, stateProbSafe, observations);
		if (similar(transitionMatrixSafe,transitionMatrix,hiddenStates,hiddenStates) && similar(emissionMatrixSafe,emissionMatrix,differentObservables,hiddenStates)){
			runs[run]=cycles;
            //DEBUG OFF
			//printf("run %i: \t %llu cycles \n",run, cycles);
		}else{	
		
			free(groundTransitionMatrix);
			free(groundEmissionMatrix);
			free(observations);
			free(transitionMatrix);
			free(emissionMatrix);
			free(stateProb);
			//free(alpha);
			//free(beta);
			//free(gamma);
			//free(xi);
			free(gamma_sum);
			free(gamma_T);
			free(ct);
			printf("Something went wrong! \n");
			return -1;//error Jan
		}


	}

	qsort (runs, maxRuns, sizeof (double), compare_doubles);
  	double medianTime = runs[maxRuns/2];
	printf("Median Time: \t %lf cycles \n", medianTime); 

	write_result(transitionMatrix, emissionMatrix, observations, stateProb, steps, hiddenStates, differentObservables, T);
        
	free(groundTransitionMatrix);
	free(groundEmissionMatrix);
	free(observations);
	free(transitionMatrix);
	free(emissionMatrix);
	free(stateProb);
	//free(alpha);
	//free(beta);
	//free(gamma);
	//free(xi);

	free(gamma_sum);
	free(gamma_T);
	free(ct);

	return 0; 
} 
