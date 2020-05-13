#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <float.h> //for DOUBL_MAX
#include "tsc_x86.h"

#include "io.h"
#include "tested.h"

#define EPSILON 1e-12
#define DELTA 1e-2
#define maxSteps 100

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
void forward(const double* const a, const double* const p, const double* const b, double* const alpha,  const int * const y, double* const ct, const int N, const int K, const int T){


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
	double ct0 = 0.0;
	//ct[0]=0.0;
	//compute alpha(0) and scaling factor for t = 0
	int y0 = y[0];
	for(int s = 0; s < N; s++){
		double alphasT = p[s] * b[s*K + y0];
		ct0 += alphasT;
		alpha[s*T] = alphasT;
		//printf("%lf %lf %lf \n", alpha[s*T], p[s], b[s*K+y[0]]);
	}
	
	//scaling factor for t = 0
	ct0 = 1.0 / ct0;

	//scale alpha(0)
	for(int s = 0; s < N; s++){
		alpha[s*T] *= ct0;
		//printf("%lf %lf %lf \n", alpha[s*T], p[s], b[s*K+y[0]]);
	}
	//print_matrix(alpha,N,T);
	ct[0] = ct0;

	for(int t = 1; t < T; t++){
		double ctt = 0.0;	
		const int yt = y[t];	
		//ct[t]=0.0;
		for(int s = 0; s<N; s++){// s=new_state
			double alphasTt = 0;
			//alpha[s*T + t] = 0;
			for(int j = 0; j < N; j++){//j=old_states
				alphasTt += alpha[j*T + t-1] * a[j*N + s];
				//alpha[s*T + t] += alpha[j*T + t-1] * a[j*N + s];
				//printf("%lf %lf %lf %lf %i \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t+1]],y[t]);
			}

			alphasTt *= b[s*K + yt];
			//print_matrix(alpha,N,T);
			ctt += alphasTt;
			alpha[s*T + t] = alphasTt;
		}
		//scaling factor for t 
		ctt = 1.0 / ctt;
		
		//scale alpha(t)
		for(int s = 0; s<N; s++){// s=new_state
			alpha[s*T + t] *= ctt;
		}
		ct[t] = ctt;
	}

	//print_matrix(alpha,N,T);

	return;
}

//Ang
void backward(const double* const a, const double* const b, double* const beta, const int * const y, const double * const ct, const int N, const int K, const int T ){
	
	double ctT1 = ct[T-1];	
	for(int s = 1; s < N+1; s++){
		beta[s*T-1] = /* 1* */ctT1;
	}

	for(int t = T-1; t > 0; t--){
		const int yt =y[t];
		const double ctt1 = ct[t-1];
		for(int s = 0; s < N; s++){//s=older state
			double betasTt1 = 0;
       			//beta[s*T + t-1] = 0.;
			for(int j = 0; j < N; j++){//j=newer state
				betasTt1 += beta[j*T + t ] * a[s*N + j] * b[j*K + yt];
				//printf("%lf %lf %lf %lf %i \n", beta[s*T + t-1], beta[j*T+t], a[s*N+j], b[j*K+y[t]],y[t]);
			}
			beta[s*T + t-1] = ctt1*betasTt1;
		}
	}
	//print_matrix(beta,N,T);
	return;
}

void update(double* const a, double* const p, double* const b, const double* const alpha, const double* const beta, double* const gamma, double* const xi, const int* const y, const double* const ct,const int N, const int K, const int T){


	double xi_sum, gamma_sum_numerator, gamma_sum_denominator;

	for(int s = 0; s < N; s++){ // s old state
		for(int t = 0; t < T; t++){
			gamma[s*T + t] = alpha[s*T + t] * beta[s*T + t];
		}
	}

	for(int t = 1; t < T; t++){
		const int yt = y[t];
		for(int s = 0; s < N; s++){
			const double alphasTt1 = alpha[s*T + t-1] ;
			for(int j = 0; j < N; j++){ // j new state
				xi[((t-1) * N + s) * N + j] = alphasTt1 * a[s*N + j] * beta[j*T + t] * b[j*K + yt]; 
				//Unlike evidence, Xi has a and b under the line in Wikipedia. The notation "P(Y|theta)" on Wikipedia is misleading.
				//Discussion from 22.4.20 showed that this should be the same. Notation on wikipedia is consistent.
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

	const double ct0div = 1/ct[0];
	for(int s = 0; s < N; s++){
		// new pi
		//XXX the next line is not in the r library hmm.
    		p[s] = gamma[s*T]*ct0div;
    
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
		const double gamma_sum_denominator_div = 1/gamma_sum_denominator ;
		for(int v = 0; v < K; v++){
			gamma_sum_numerator = 0.;
			for(int t = 0; t < T; t++){//why 1 indented => better?
				if(y[t] == v){// XXX rather AllPossibleValues[v] ??? => don't understand the question. What is AllPossibleValues[v]?
					gamma_sum_numerator += gamma[s*T + t]/ct[t];//why different t here than in y[t] => I think this was a typo. Indeed it should be the same t for gamma and y.
				}
			}
			// new emmision matrix
			b[s*K + v] = gamma_sum_numerator * gamma_sum_denominator_div;
		}
	}

	return;
}



void evidence_testing(const double* const alpha, const double* const beta,const double* const a,const double* const b,const int* const y, const double* const ct, const int N, const int T,int K){
	
	double evidence = 0;
	//evidence with alpha only:

	double cT = 1.0;
	for(int time = 0; time < T; time++){
		cT *=ct[time];
	}

	for(int state = 1; state < N+1; state++){
		evidence += alpha[state*T -1]; 
	}	
	evidence/=cT;

	printf("evidence with sum over alpha(T) : %.10lf \n", evidence);

	//evidence with alpha * beta for every time t:
	for(int time = 0 ; time < T; time++){
		evidence = 0;
		for(int state = 0; state < N; state++){
			evidence += alpha[state*T + time]*beta[state*T + time]; 
		}
		evidence/=cT*ct[time];
		printf("evidence at time %i with sum over alpha(t)*beta(t) : %.10lf \n",time, evidence);
	}

	//CONCLUSION
	//Evidence P(Y|M) = sum alpha(T) = sum alpha(t)*beta(t)	= sum sum alpha(t) * a_kw * beta(t+1)b_w(y[t+1])

}

//Jan
int finished(const double* const alpha,const double* const beta, const double* const ct, double* const l,const int N,const int T){

	/*
	//true evidence
	double oldLikelihood=*l;

	double newLikelihood = 0.0;
	//evidence with alpha only:

	double cT = 1.0;
	for(int time = 0; time < T; time++){
		cT *=ct[time];
	}

	for(int state = 1; state < N+1; state++){
		newLikelihood += alpha[state*T -1]; 
	}	
	newLikelihood/=cT;
	
	*l=newLikelihood;

	print_matrix(alpha,N,T);

	printf("evidence %.100lf , Epsilon %.100lf result %.100lf \n", newLikelihood, EPSILON,newLikelihood-oldLikelihood);
	return (newLikelihood-oldLikelihood)<EPSILON;
	*/

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
	double* ct = (double*) malloc(T * sizeof(double));

	for(int t = 0; t < 1000; t++){
		forward(transitionMatrix, piMatrix, emissionMatrix, alpha, observations, ct, hiddenStates, differentObservables, T);	
		backward(transitionMatrix, emissionMatrix, beta, observations, ct, hiddenStates, differentObservables, T);	//Ang
		update(transitionMatrix, piMatrix, emissionMatrix, alpha, beta, gamma, xi, observations, ct,  hiddenStates, differentObservables, T);//Ang
	}

	printf("new transition matrix from wikipedia example: \n \n");
	print_matrix(transitionMatrix,hiddenStates,hiddenStates);
	print_matrix(emissionMatrix, hiddenStates,differentObservables);
	print_vector(piMatrix,2);
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

	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * (T-1) * sizeof(double)); 


	double* ct = (double*) malloc(T*sizeof(double));
	
	//heatup needs some data.
	makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
	makeMatrix(hiddenStates, differentObservables, emissionMatrix);
	makeProbabilities(stateProb,hiddenStates);
	heatup(transitionMatrix,stateProb,emissionMatrix,observations,hiddenStates,differentObservables,T);
	
        int steps=0;
	for (int run=0; run<maxRuns; run++){

		//init transition Matrix, emission Matrix and initial state distribution random
		makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
       		memcpy(transitionMatrixSafe, transitionMatrix, hiddenStates*hiddenStates*sizeof(double));
		makeMatrix(hiddenStates, differentObservables, emissionMatrix);
	   	memcpy(emissionMatrixSafe, emissionMatrix, hiddenStates*differentObservables*sizeof(double));
		makeProbabilities(stateProb,hiddenStates);
        	memcpy(stateProbSafe, stateProb, hiddenStates * sizeof(double));	
		//XXX brauchen wir nach jedem run neue transitionMatrix, emissionMatrix und stateProb?

        	double logLikelihood=-DBL_MAX; //Took down here.
		double Likelihood = 0;
		//make some random observations
		int groundInitialState = rand()%hiddenStates;
		makeObservations(hiddenStates, differentObservables, groundInitialState, groundTransitionMatrix,groundEmissionMatrix,T, observations); //??? ground___ zu ___ wechseln? => Verstehe deine Frage nicht...
		//XXX Ist es notwendig nach jedem run neue observations zu machen?

		//only needed for testing
		//write_init(transitionMatrix, emissionMatrix, observations, stateProb, hiddenStates, differentObservables, T);
        
		//XXX start after makeMatrix
        	steps=0;
		start = start_tsc();

	
		do{
			forward(transitionMatrix, stateProb, emissionMatrix, alpha, observations, ct, hiddenStates, differentObservables, T);	//Luca
			backward(transitionMatrix, emissionMatrix, beta,observations, ct, hiddenStates, differentObservables, T);	//Ang
			update(transitionMatrix, stateProb, emissionMatrix, alpha, beta, gamma, xi, observations, ct, hiddenStates, differentObservables, T);  //Ang
            		steps+=1;
		}while (!finished(alpha, beta, ct, &logLikelihood, hiddenStates, T) && steps<maxSteps);

		cycles = stop_tsc(start);
        	cycles = cycles/steps;
		//Jan

		//print_matrix(xi,hiddenStates*hiddenStates,T);
		//print_matrix(transitionMatrix,hiddenStates,hiddenStates);
		//print_matrix(emissionMatrix, hiddenStates,differentObservables);


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
            	free(alpha);
            	free(beta);
            	free(gamma);
            	free(xi);
                free(ct);
                free(inv_ct);
                free(transitionMatrixSafe);
            	free(emissionMatrixSafe);
                free(stateProbSafe);
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
	free(alpha);
	free(beta);
	free(gamma);
	free(xi);
    free(ct);
    free(transitionMatrixSafe);
	free(emissionMatrixSafe);
    free(stateProbSafe);

	return 0; 
} 
