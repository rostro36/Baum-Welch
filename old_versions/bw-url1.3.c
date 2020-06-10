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

void transpose(double* a, const int rows, const int cols){
	double* transpose = (double*)calloc(cols*rows, sizeof(double));
	memcpy(transpose, a, rows*cols*sizeof(double));
	for(int row = 0 ; row < rows; row++){
		for(int col =0; col < cols; col++){
			a[col * rows + row]  = transpose[row * cols + col];
		}
	}
	free(transpose);
}

//for sorting at the end
int compare_doubles (const void *a, const void *b){
	const double *da = (const double *) a;
	const double *db = (const double *) b;

	return (*da > *db) - (*da < *db);
}

int chooseOf(const int choices, const double* const probArray){
	//decide at which proba to stop
	double decider= (double)rand()/(double)RAND_MAX;
	double probSum=0;
	for(int i=0; i<choices;i++){
		//if decider in range(probSum[t-1],probSum[t])->return t
		probSum+=probArray[i];
		if (decider<=probSum)
		{
			return i;
		}
	}
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
void makeMatrix(const int dim1,const int dim2, double* const matrix){

	for (int row=0;row<dim1;row++){
		//make probabilites for one row
		makeProbabilities(matrix + row*dim2,dim2);
	}
}
void baum_welch(double* const a, double* const b, double* const p, const int* const y, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int N, const int K, const int T){

	//XXX It is not optimal to create four arrays in each iteration.
	//This is only used to demonstarte the use of the pointer swap at the end
	//When inlined the  arrays should be generated at the beginning of the main function like the other arrays we use
	//Then the next four lines can be deleted 
	//Rest needs no modification
	double* beta = (double*) malloc(T  * sizeof(double));
	double* beta_new = (double*) malloc(T * sizeof(double));
	double* alpha = (double*) malloc(N * T * sizeof(double));

	int yt = y[T-1];
	//add remaining parts of the sum of gamma 
	for(int s = 0; s < N; s++){
		double gamma_Ts = gamma_T[s];
		//if you use real gamma you have to divide by ct[t-1]
		gamma_T[s] += gamma_sum[s]; /* /ct[T-1] */;
        b_new[yt*N+s]+=gamma_Ts;
	}

	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] / gamma_T[s];
			b_new[v*N + s] = 0.0;
		}
	}

	//FORWARD

	double ctt = 0.0;
	//compute alpha(0) and scaling factor for t = 0
	int y0 = y[0];
	for(int s = 0; s < N; s++){
		double alphas = p[s] * b[y0*N + s];
		ctt += alphas;
		alpha[s] = alphas;
	}
	
	ctt = 1.0 / ctt;

	//scale alpha(0)
	for(int s = 0; s < N; s++){
		alpha[s] *= ctt;
	}
	//print_matrix(alpha,N,T);
	ct[0] = ctt;

	//Compute alpha(1) and scale transitionMatrix
	ctt = 0.0;	
	yt = y[1];	
	for(int s = 0; s<N; s++){// s=new_state
		double alphatNs = 0;
		for(int j = 0; j < N; j++){//j=old_states
			double ajNs =  a_new[j*N + s] / gamma_sum[j];
			a[j*N + s] = ajNs;
			alphatNs += alpha[0*N + j] * ajNs;
			a_new[j*N+s] = 0.0;
		}
		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[1*N + s] = alphatNs;
	}
	//scaling factor for t 
	ctt = 1.0 / ctt;
	
	//scale alpha(t)
	for(int s = 0; s<N; s++){// s=new_state
		alpha[1*N+s] *= ctt;
	}
	ct[1] = ctt;

	for(int t = 2; t < T-1; t++){
		ctt = 0.0;	
		yt = y[t];	
		for(int s = 0; s<N; s++){// s=new_state
			double alphatNs = 0;
			//alpha[s*T + t] = 0;
			for(int j = 0; j < N; j++){//j=old_states
				alphatNs += alpha[(t-1)*N + j] * a[j*N + s];
			}
			alphatNs *= b[yt*N + s];
			ctt += alphatNs;
			alpha[t*N + s] = alphatNs;
		}
		//scaling factor for t 
		ctt = 1.0 / ctt;
		
		//scale alpha(t)
		for(int s = 0; s<N; s++){// s=new_state
			alpha[t*N+s] *= ctt;
		}
		ct[t] = ctt;
	}
		
	//compute alpha(T-1)
	ctt = 0.0;	
	yt = y[T-1];	
	for(int s = 0; s<N; s++){// s=new_state
		double alphatNs = 0;
		//alpha[s*T + t] = 0;
		for(int j = 0; j < N; j++){//j=old_states
			alphatNs += alpha[(T-2)*N + j] * a[j*N + s];
		}

		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[(T-1)*N + s] = alphatNs;
	}
	//scaling factor for T-1
	ctt = 1.0 / ctt;
		
	//scale alpha(t)
	for(int s = 0; s<N; s++){// s=new_state
		double alphaT1Ns = alpha[(T-1) * N + s]*ctt;
		alpha[(T-1)*N+s] = alphaT1Ns;
		gamma_T[s] = alphaT1Ns /* *ct[T-1]*/;
	}
	ct[T-1] = ctt;

	//FUSED BACKWARD and UPDATE STEP

	for(int s = 0; s < N; s++){
		beta[s] = /* 1* */ctt;
		gamma_sum[s] = 0.0;
	}

	//compute sum of xi and gamma from t= 0...T-2
    yt = y[T-1];
	for(int t = T-1; t > 0; t--){
		const int yt1 = y[t-1];
		ctt = ct[t-1];
		for(int s = 0; s < N ; s++){
			double beta_news = 0.0;
			double alphat1Ns = alpha[(t-1)*N + s];
			for(int j = 0; j < N; j++){
				double temp = a[s*N +j] * beta[j] * b[yt*N + j];
				
				double xi_sjt = alphat1Ns * temp;
				a_new[s*N+j] +=xi_sjt;
				beta_news += temp;
				
			}
			double ps =alphat1Ns*beta_news/* *ct[t-1]*/;  
			p[s] = ps;
			beta_new[s] = beta_news*ctt;
			gamma_sum[s]+= ps /* /ct[t-1] */ ;
            b_new[yt1*N+s]+=ps;
		}
		double * temp = beta_new;
		beta_new = beta;
		beta = temp;
		yt=yt1;
	}

	free(alpha);
	free(beta);
	free(beta_new);
	
	return;
}

void final_scaling(double* const a, double* const b, double* const p, const int* const y, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int N, const int K, const int T){
	//compute new transition matrix
	for(int s = 0; s < N; s++){
		double gamma_sums_inv = 1./gamma_sum[s];
		for(int j = 0; j < N; j++){
			a[s*N+j] = a_new[s*N+j]*gamma_sums_inv;
		}
	}

	int yt =y[T-1];
	//add remaining parts of the sum of gamma 
	for(int s = 0; s < N; s++){	
		double gamma_Ts = gamma_T[s];
		//if you use real gamma you have to divide by ct[t-1]
		gamma_sum[s] += gamma_Ts /* /ct[T-1] */;
        b_new[yt*N+s]+=gamma_Ts;
	}

	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] / gamma_sum[s];
		}
	}
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

void heatup(double* const transitionMatrix,double* const stateProb,double* const emissionMatrix,const int* const observations,const int hiddenStates,const int differentObservables,const int T){

	double* ct = (double*) malloc( T * sizeof(double));
	double* gamma_T = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma_sum = (double*) malloc(hiddenStates * T * sizeof(double));
	double* a_new = (double*) malloc(hiddenStates * hiddenStates * sizeof(double));
	double* b_new = (double*) malloc(differentObservables*hiddenStates * sizeof(double));
	
	for(int j=0;j<10;j++){
		baum_welch(transitionMatrix, emissionMatrix, stateProb, observations, gamma_sum, gamma_T,a_new,b_new,ct, hiddenStates, differentObservables, T);
	}

	free(ct);
	free(gamma_T);
	free(gamma_sum);
	free(a_new);
	free(b_new);		
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
	double* transitionMatrixTesting=(double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* emissionMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));
	double* emissionMatrixSafe = (double*) malloc(hiddenStates*differentObservables*sizeof(double));
	double* emissionMatrixTesting=(double*) malloc(hiddenStates*differentObservables*sizeof(double));

	//init state distribution
	double* stateProb  = (double*) malloc(hiddenStates * sizeof(double));
	double* stateProbSafe  = (double*) malloc(hiddenStates * sizeof(double));
	double* stateProbTesting  = (double*) malloc(hiddenStates * sizeof(double));

	double* gamma_T = (double*) malloc( hiddenStates * sizeof(double));
	double* gamma_sum = (double*) malloc( hiddenStates * sizeof(double));
	
	double* a_new = (double*) malloc(hiddenStates * hiddenStates * sizeof(double));
	double* b_new = (double*) malloc(differentObservables*hiddenStates * sizeof(double));
	
	double* ct = (double*) malloc(T*sizeof(double));


	double* beta = (double*) malloc(T  * sizeof(double));
	double* beta_new = (double*) malloc(T * sizeof(double));
	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* ab= (double*) malloc(hiddenStates*hiddenStates*differentObservables*sizeof(double));
	
	//random init transition matrix, emission matrix and state probabilities.
	makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
	makeMatrix(hiddenStates, differentObservables, emissionMatrix);
	makeProbabilities(stateProb,hiddenStates);

	transpose(emissionMatrix, hiddenStates, differentObservables);

	//make a copy of matrices to be able to reset matrices after each run to initial state and to be able to test implementation.
	memcpy(transitionMatrixSafe, transitionMatrix, hiddenStates*hiddenStates*sizeof(double));
   	memcpy(emissionMatrixSafe, emissionMatrix, hiddenStates*differentObservables*sizeof(double));
    memcpy(stateProbSafe, stateProb, hiddenStates * sizeof(double));

	//heat up cache
	heatup(transitionMatrix,stateProb,emissionMatrix,observations,hiddenStates,differentObservables,T);
	
	int steps=0;
	double disparance;
	for (int run=0; run<maxRuns; run++){

		//init transition Matrix, emission Matrix and initial state distribution random
		memcpy(transitionMatrix, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
   		memcpy(emissionMatrix, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
      		memcpy(stateProb, stateProbSafe, hiddenStates * sizeof(double));
		
		memcpy(transitionMatrixTesting, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
   		memcpy(emissionMatrixTesting, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
      		memcpy(stateProbTesting, stateProbSafe, hiddenStates * sizeof(double));

        	double logLikelihood=-DBL_MAX; //Took down here.

		//only needed for testing with R
		//write_init(transitionMatrix, emissionMatrix, observations, stateProb, hiddenStates, differentObservables, T);
        
        	steps=1;
		start = start_tsc();
		
//Assumptions for unrolling: hiddenStates%4=0, differentObservables%4=0
		
//Initial step

	//FORWARD

		double ct0 = 0.0;
		int y0 = observations[0];
		for(int s = 0; s < hiddenStates; s+=4){
			//s
			double alphas = stateProb[s] * emissionMatrix[y0*hiddenStates + s];
			ct0 += alphas;
			alpha[s] = alphas;
			//s+1
			alphas = stateProb[s+1] * emissionMatrix[y0*hiddenStates + s+1];
			ct0 += alphas;
			alpha[s+1] = alphas;
			//s+2
			alphas = stateProb[s+2] * emissionMatrix[y0*hiddenStates + s+2];
			ct0 += alphas;
			alpha[s+2] = alphas;
			//s+3
			alphas = stateProb[s+3] * emissionMatrix[y0*hiddenStates + s+3];
			ct0 += alphas;
			alpha[s+3] = alphas;
		}
	
		ct0 = 1.0 / ct0;
	
		//scale alpha(0)
		for(int s = 0; s < hiddenStates; s+=4){
			//s
			alpha[s] *= ct0;
			//s+1
			alpha[s+1] *= ct0;
			//s+2
			alpha[s+2] *= ct0;
			//s+3
			alpha[s+3] *= ct0;
		}
		//print_matrix(alpha,N,T);
		ct[0] = ct0;
	
		for(int t = 1; t < T-1; t++){
			double ctt = 0.0;	
			const int yt = observations[t];	
			for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				double alphatNs = 0;
				double alphatNs1 = 0;
				double alphatNs2 = 0;
				double alphatNs3 = 0;
				for(int j = 0; j < hiddenStates; j+=4){//j=old_states
					double alphaFactor=alpha[(t-1)*hiddenStates + j];
					double alphaFactor1=alpha[(t-1)*hiddenStates + j+1];
					double alphaFactor2=alpha[(t-1)*hiddenStates + j+2];
					double alphaFactor3=alpha[(t-1)*hiddenStates + j+3];
				
					//s,j
					alphatNs += alphaFactor * transitionMatrix[j*hiddenStates + s];
					//s,j+1
					alphatNs += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s];
					//s,j+2
					alphatNs += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s];
					//s,j+3
					alphatNs += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s];
					//s+1,j
					alphatNs1 += alphaFactor * transitionMatrix[j*hiddenStates + s+1];
					//s+1,j+1
					alphatNs1 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+1];
					//s+1,j+2
					alphatNs1 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+1];
					//s+1,j+3
					alphatNs1 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+1];
					//s+2,j
					alphatNs2 += alphaFactor * transitionMatrix[j*hiddenStates + s+2];
					//s+2,j+1
					alphatNs2 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+2];
					//s+2,j+2
					alphatNs2 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+2];
					//s+2,j+3
					alphatNs2 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+2];
					//s+3,j
					alphatNs3 += alphaFactor * transitionMatrix[j*hiddenStates + s+3];
					//s+3,j+1
					alphatNs3 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+3];
					//s+3,j+2
					alphatNs3 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+3];
					//s+3,j+3
					alphatNs3 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+3];
				}
				alphatNs *= emissionMatrix[yt*hiddenStates + s];
				ctt += alphatNs;
				alpha[t*hiddenStates + s] = alphatNs;
				alphatNs1 *= emissionMatrix[yt*hiddenStates + s+1];
				ctt += alphatNs1;
				alpha[t*hiddenStates + s+1] = alphatNs1;
				alphatNs2 *= emissionMatrix[yt*hiddenStates + s+2];
				ctt += alphatNs2;
				alpha[t*hiddenStates + s+2] = alphatNs2;
				alphatNs3 *= emissionMatrix[yt*hiddenStates + s+3];
				ctt += alphatNs3;
				alpha[t*hiddenStates + s+3] = alphatNs3;
				
				
			
			}
			//scaling factor for t 
			ctt = 1.0 / ctt;
			
			//scale alpha(t)
			for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				alpha[t*hiddenStates+s] *= ctt;
				alpha[t*hiddenStates+s+1] *= ctt;
				alpha[t*hiddenStates+s+2] *= ctt;
				alpha[t*hiddenStates+s+3] *= ctt;
			}
			ct[t] = ctt;
			
		}
			
		double ctt = 0.0;	
		int yt = observations[T-1];	
		for(int s = 0; s<hiddenStates; s+=4){// s=new_state
			double alphatNs = 0; //alpha[s*T + t] = 0;
			double alphatNs1 = 0;
			double alphatNs2 = 0;
			double alphatNs3 = 0;
			for(int j = 0; j < hiddenStates; j+=4){//j=old_states
				double alphaFactor=alpha[(T-2)*hiddenStates + j];
				double alphaFactor1=alpha[(T-2)*hiddenStates + j+1];
				double alphaFactor2=alpha[(T-2)*hiddenStates + j+2];
				double alphaFactor3=alpha[(T-2)*hiddenStates + j+3];
				//s
				//j
				alphatNs += alphaFactor * transitionMatrix[j*hiddenStates + s];
				//j+1
				alphatNs += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s];
				//j+2
				alphatNs += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s];
				//j+3
				alphatNs += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s];
				//s+1
				//j
				alphatNs1 += alphaFactor * transitionMatrix[j*hiddenStates + s+1];
				//j+1
				alphatNs1 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+1];
				//j+2
				alphatNs1 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+1];
				//j+3
				alphatNs1 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+1];
				//s+2
				//j
				alphatNs2 += alphaFactor * transitionMatrix[j*hiddenStates + s+2];
				//j+1
				alphatNs2 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+2];
				//j+2
				alphatNs2 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+2];
				//j+3
				alphatNs2 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+2];
				//s+3
				//j
				alphatNs3 += alphaFactor * transitionMatrix[j*hiddenStates + s+3];
				//j+1
				alphatNs3 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+3];
				//j+2
				alphatNs3 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+3];
				//j+3
				alphatNs3 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+3];
			}
			alphatNs *= emissionMatrix[yt*hiddenStates + s];
			ctt += alphatNs;
			alpha[(T-1)*hiddenStates + s] = alphatNs;
			
			alphatNs1 *= emissionMatrix[yt*hiddenStates + s+1];
			ctt += alphatNs1;
			alpha[(T-1)*hiddenStates + s+1] = alphatNs1;
			
			alphatNs2 *= emissionMatrix[yt*hiddenStates + s+2];
			ctt += alphatNs2;
			alpha[(T-1)*hiddenStates + s+2] = alphatNs2;
			
			alphatNs3 *= emissionMatrix[yt*hiddenStates + s+3];
			ctt += alphatNs3;
			alpha[(T-1)*hiddenStates + s+3] = alphatNs3;
			
		}
		//scaling factor for T-1
		ctt = 1.0 / ctt;
			
		//scale alpha(t)
		for(int s = 0; s<hiddenStates; s+=4){// s=new_state
			//s
			double alphaT1Ns = alpha[(T-1) * hiddenStates + s]*ctt;
			alpha[(T-1)*hiddenStates + s] = alphaT1Ns;
			gamma_T[s] = alphaT1Ns /* *ct[T-1]*/;
			//s+1
			double alphaT1Ns1 = alpha[(T-1) * hiddenStates + s+1]*ctt;
			alpha[(T-1)*hiddenStates + s+1] = alphaT1Ns1;
			gamma_T[s+1] = alphaT1Ns1 /* *ct[T-1]*/;
			//s+2
			double alphaT1Ns2 = alpha[(T-1) * hiddenStates + s+2]*ctt;
			alpha[(T-1)*hiddenStates + s+2] = alphaT1Ns2;
			gamma_T[s+2] = alphaT1Ns2 /* *ct[T-1]*/;
			//s+3
			double alphaT1Ns3 = alpha[(T-1) * hiddenStates + s+3]*ctt;
			alpha[(T-1)*hiddenStates + s+3] = alphaT1Ns3;
			gamma_T[s+3] = alphaT1Ns3 /* *ct[T-1]*/;
		}
		ct[T-1] = ctt;
		
		
		
		
		//FUSED BACKWARD and UPDATE STEP
	
		for(int s = 0; s < hiddenStates; s+=4){
			for(int j = 0; j < hiddenStates; j+=4){
				//s
				//j
				a_new[s*hiddenStates + j] =0.0;
				//j+1
				a_new[s*hiddenStates + j+1] =0.0;
				//j+2
				a_new[s*hiddenStates + j+2] =0.0;
				//j+3
				a_new[s*hiddenStates + j+3] =0.0;
				//s+1
				//j
				a_new[(s+1)*hiddenStates + j] =0.0;
				//j+1
				a_new[(s+1)*hiddenStates + j+1] =0.0;
				//j+2
				a_new[(s+1)*hiddenStates + j+2] =0.0;
				//j+3
				a_new[(s+1)*hiddenStates + j+3] =0.0;
				//s+2
				//j
				a_new[(s+2)*hiddenStates + j] =0.0;
				//j+1
				a_new[(s+2)*hiddenStates + j+1] =0.0;
				//j+2
				a_new[(s+2)*hiddenStates + j+2] =0.0;
				//j+3
				a_new[(s+2)*hiddenStates + j+3] =0.0;
				//s+3
				//j
				a_new[(s+3)*hiddenStates + j] =0.0;
				//j+1
				a_new[(s+3)*hiddenStates + j+1] =0.0;
				//j+2
				a_new[(s+3)*hiddenStates + j+2] =0.0;
				//j+3
				a_new[(s+3)*hiddenStates + j+3] =0.0;
			}
			
			//s
			beta[s] = /* 1* */ctt;
			gamma_sum[s] = 0.0;
			//s+1
			beta[s+1] = /* 1* */ctt;
			gamma_sum[s+1] = 0.0;
			//s+2
			beta[s+2] = /* 1* */ctt;
			gamma_sum[s+2] = 0.0;
			//s+3
			beta[s+3] = /* 1* */ctt;
			gamma_sum[s+3] = 0.0;
		}
	
		for(int v = 0;  v < differentObservables; v+=4){
			for(int s = 0; s < hiddenStates; s+=4){
				//v,s
				b_new[v*hiddenStates + s] = 0.0;
				//v,s+1
				b_new[v*hiddenStates + s+1] = 0.0;
				//v,s+2
				b_new[v*hiddenStates + s+2] = 0.0;
				//v,s+3
				b_new[v*hiddenStates + s+3] = 0.0;
				//v+1,s
				b_new[(v+1)*hiddenStates + s] = 0.0;
				//v+1,s+1
				b_new[(v+1)*hiddenStates + s+1] = 0.0;
				//v+1,s+2
				b_new[(v+1)*hiddenStates + s+2] = 0.0;
				//v+1,s+3
				b_new[(v+1)*hiddenStates + s+3] = 0.0;
				//v+2,s
				b_new[(v+2)*hiddenStates + s] = 0.0;
				//v+2,s+1
				b_new[(v+2)*hiddenStates + s+1] = 0.0;
				//v+2,s+2
				b_new[(v+2)*hiddenStates + s+2] = 0.0;
				//v+2,s+3
				b_new[(v+2)*hiddenStates + s+3] = 0.0;
				//v+3,s
				b_new[(v+3)*hiddenStates + s] = 0.0;
				//v+3,s+1
				b_new[(v+3)*hiddenStates + s+1] = 0.0;
				//v+3,s+2
				b_new[(v+3)*hiddenStates + s+2] = 0.0;
				//v+3,s+3
				b_new[(v+3)*hiddenStates + s+3] = 0.0;
			}
		}
		
		
		for(int v = 0; v < differentObservables; v++){
			for(int s = 0; s < hiddenStates; s+=4){
				for(int j = 0; j < hiddenStates; j+=4){
					ab[(v*hiddenStates + s) * hiddenStates + j] = transitionMatrix[s*hiddenStates + j] * emissionMatrix[v*hiddenStates +j];
					ab[(v*hiddenStates + s) * hiddenStates + j+1] = transitionMatrix[s*hiddenStates + j+1] * emissionMatrix[v*hiddenStates +j+1];
					ab[(v*hiddenStates + s) * hiddenStates + j+2] = transitionMatrix[s*hiddenStates + j+2] * emissionMatrix[v*hiddenStates +j+2];
					ab[(v*hiddenStates + s) * hiddenStates + j+3] = transitionMatrix[s*hiddenStates + j+3] * emissionMatrix[v*hiddenStates +j+3];
					
					ab[(v*hiddenStates + s+1) * hiddenStates + j] = transitionMatrix[(s+1)*hiddenStates + j] * emissionMatrix[v*hiddenStates +j];
					ab[(v*hiddenStates + s+1) * hiddenStates + j+1] = transitionMatrix[(s+1)*hiddenStates + j+1] * emissionMatrix[v*hiddenStates +j+1];
					ab[(v*hiddenStates + s+1) * hiddenStates + j+2] = transitionMatrix[(s+1)*hiddenStates + j+2] * emissionMatrix[v*hiddenStates +j+2];
					ab[(v*hiddenStates + s+1) * hiddenStates + j+3] = transitionMatrix[(s+1)*hiddenStates + j+3] * emissionMatrix[v*hiddenStates +j+3];
					
					ab[(v*hiddenStates + s+2) * hiddenStates + j] = transitionMatrix[(s+2)*hiddenStates + j] * emissionMatrix[v*hiddenStates +j];
					ab[(v*hiddenStates + s+2) * hiddenStates + j+1] = transitionMatrix[(s+2)*hiddenStates + j+1] * emissionMatrix[v*hiddenStates +j+1];
					ab[(v*hiddenStates + s+2) * hiddenStates + j+2] = transitionMatrix[(s+2)*hiddenStates + j+2] * emissionMatrix[v*hiddenStates +j+2];
					ab[(v*hiddenStates + s+2) * hiddenStates + j+3] = transitionMatrix[(s+2)*hiddenStates + j+3] * emissionMatrix[v*hiddenStates +j+3];
					
					ab[(v*hiddenStates + s+3) * hiddenStates + j] = transitionMatrix[(s+3)*hiddenStates + j] * emissionMatrix[v*hiddenStates +j];
					ab[(v*hiddenStates + s+3) * hiddenStates + j+1] = transitionMatrix[(s+3)*hiddenStates + j+1] * emissionMatrix[v*hiddenStates +j+1];
					ab[(v*hiddenStates + s+3) * hiddenStates + j+2] = transitionMatrix[(s+3)*hiddenStates + j+2] * emissionMatrix[v*hiddenStates +j+2];
					ab[(v*hiddenStates + s+3) * hiddenStates + j+3] = transitionMatrix[(s+3)*hiddenStates + j+3] * emissionMatrix[v*hiddenStates +j+3];
					
				}
			}
		}
		

		//compute sum of xi and gamma from t= 0...T-2
    		yt = observations[T-1];
		for(int t = T-1; t > 0; t--){
			const int yt1 = observations[t-1];
			const double ctt = ct[t-1];
			for(int s = 0; s < hiddenStates ; s+=4){
				double beta_news0 = 0.0;
				double alphat1Ns0 = alpha[(t-1)*hiddenStates + s];
				double beta_news1 = 0.0;
				double alphat1Ns1 = alpha[(t-1)*hiddenStates + s+1];
				double beta_news2 = 0.0;
				double alphat1Ns2 = alpha[(t-1)*hiddenStates + s+2];
				double beta_news3 = 0.0;
				double alphat1Ns3 = alpha[(t-1)*hiddenStates + s+3];
				for(int j = 0; j < hiddenStates; j+=4){
					
            								double beta0 = beta[j];
					double beta1 = beta[j+1];
					double beta2 = beta[j+2];
					double beta3 = beta[j+3];
				
					//s
					//j
					double temp00 = ab[(yt*hiddenStates + s)*hiddenStates + j] * beta0;
					a_new[s*hiddenStates+j] += alphat1Ns0 * temp00;
					beta_news0 += temp00;
					
					//j+1
					double temp01 = ab[(yt*hiddenStates + s)*hiddenStates + j+1] * beta1;
					a_new[s*hiddenStates+j+1] += alphat1Ns0 * temp01;
					beta_news0 += temp01;
					
									
					//j+2
					double temp02 = ab[(yt*hiddenStates + s)*hiddenStates + j+2] * beta2;
					a_new[s*hiddenStates+j+2] += alphat1Ns0 * temp02;
					beta_news0 += temp02;
					
					//j+3
					double temp03 = ab[(yt*hiddenStates + s)*hiddenStates + j+3] * beta3;
					a_new[s*hiddenStates+j+3] += alphat1Ns0 * temp03;
					beta_news0 += temp03;
					
					
					//s+1
					
					//j
					double temp10 = ab[(yt*hiddenStates + s+1)*hiddenStates + j] * beta0;
					a_new[(s+1)*hiddenStates+j] += alphat1Ns1 * temp10;
					beta_news1 += temp10;
					
					//j+1
					double temp11 = ab[(yt*hiddenStates + s+1)*hiddenStates + j+1] * beta1;
					a_new[(s+1)*hiddenStates+j+1] += alphat1Ns1 * temp11;
					beta_news1 += temp11;
					
									
					//j+2
					double temp12 = ab[(yt*hiddenStates + s+1)*hiddenStates + j+2] * beta2;
					a_new[(s+1)*hiddenStates+j+2] += alphat1Ns1 * temp12;
					beta_news1 += temp12;
					
					//j+3
					double temp13 = ab[(yt*hiddenStates + s+1)*hiddenStates + j+3] * beta3;
					a_new[(s+1)*hiddenStates+j+3] += alphat1Ns1 * temp13;
					beta_news1 += temp13;
					
					//s+2
				
					//j
					double temp20 = ab[(yt*hiddenStates + s+2)*hiddenStates + j] * beta0;
					a_new[(s+2)*hiddenStates+j] += alphat1Ns2 * temp20;
					beta_news2 += temp20;
					
					//j+1
					double temp21 = ab[(yt*hiddenStates + s+2)*hiddenStates + j+1] * beta1;
					a_new[(s+2)*hiddenStates+j+1] += alphat1Ns2 * temp21;
					beta_news2 += temp21;
					
									
					//j+2
					double temp22 = ab[(yt*hiddenStates + s+2)*hiddenStates + j+2] * beta2;
					a_new[(s+2)*hiddenStates+j+2] += alphat1Ns2 * temp22;
					beta_news2 += temp22;
					
					//j+3
					double temp23 = ab[(yt*hiddenStates + s+2)*hiddenStates + j+3] * beta3;
					a_new[(s+2)*hiddenStates+j+3] += alphat1Ns2 * temp23;
					beta_news2 += temp23;
					
					//s+3
					
					//j
					double temp30 = ab[(yt*hiddenStates + s+3)*hiddenStates + j] * beta0;
					a_new[(s+3)*hiddenStates+j] += alphat1Ns3 * temp30;
					beta_news3 += temp30;
					
					//j+1
					double temp31 = ab[(yt*hiddenStates + s+3)*hiddenStates + j+1] * beta1;
					a_new[(s+3)*hiddenStates+j+1] += alphat1Ns3 * temp31;
					beta_news3 += temp31;
					
									
					//j+2
					double temp32 = ab[(yt*hiddenStates + s+3)*hiddenStates + j+2] * beta2;
					a_new[(s+3)*hiddenStates+j+2] += alphat1Ns3 * temp32;
					beta_news3 += temp32;
					
					//j+3
					double temp33 = ab[(yt*hiddenStates + s+3)*hiddenStates + j+3] * beta3;
					a_new[(s+3)*hiddenStates+j+3] += alphat1Ns3 * temp33;
					beta_news3 += temp33;
				}
				//s
				double ps0 =alphat1Ns0*beta_news0/* *ct[t-1]*/;  
				stateProb[s] = ps0;
				beta_new[s] = beta_news0*ctt;
				//if you use real gamma you have to divide with ct[t-1]
				gamma_sum[s]+= ps0 /* /ct[t-1] */ ;
            			b_new[yt1*hiddenStates+s]+=ps0;
            			
            			//s+1
				double ps1 =alphat1Ns1*beta_news1/* *ct[t-1]*/;  
				stateProb[s+1] = ps1;
				beta_new[s+1] = beta_news1*ctt;
				//if you use real gamma you have to divide with ct[t-1]
				gamma_sum[s+1]+= ps1 /* /ct[t-1] */ ;
        			b_new[yt1*hiddenStates+ s+1]+=ps1;
				
				//s+2
				double ps2 =alphat1Ns2*beta_news2/* *ct[t-1]*/;  
				stateProb[s+2] = ps2;
				beta_new[s+2] = beta_news2*ctt;
				//if you use real gamma you have to divide with ct[t-1]
				gamma_sum[s+2]+= ps2 /* /ct[t-1] */ ;
        			b_new[yt1*hiddenStates+ s+2]+=ps2;
			
			
				//s+3
				double ps3 =alphat1Ns3*beta_news3/* *ct[t-1]*/;  
				stateProb[s+3] = ps3;
				beta_new[s+3] = beta_news3*ctt;
				//if you use real gamma you have to divide with ct[t-1]
				gamma_sum[s+3]+= ps3 /* /ct[t-1] */ ;
        			b_new[yt1*hiddenStates+ s+3]+=ps3;
			
	
			}	
			double * temp = beta_new;
			beta_new = beta;
			beta = temp;
        		yt=yt1;		
    		}
	
		do{
//BAUM-WELCH
	        	int yt = observations[T-1];
		        //add remaining parts of the sum of gamma 
		        for(int s = 0; s < hiddenStates; s+=4){
				double gamma_Ts0 = gamma_T[s];
				//if you use real gamma you have to divide by ct[t-1]
				double gamma_sums0 = gamma_sum[s];
				double gamma_tot0 = gamma_Ts0 + gamma_sums0 /* /ct[T-1] */;
				gamma_T[s] = 1./gamma_tot0;
				gamma_sum[s] = 1./gamma_sums0;
		        	b_new[yt*hiddenStates+s]+=gamma_Ts0;
	   
				double gamma_Ts1 = gamma_T[s+1];
				//if you use real gamma you have to divide by ct[t-1]
				double gamma_sums1 = gamma_sum[s+1];
				double gamma_tot1 = gamma_Ts1 + gamma_sums1 /* /ct[T-1] */;
				gamma_T[s+1] = 1./gamma_tot1;
				gamma_sum[s+1] = 1./gamma_sums1;
		        	b_new[yt*hiddenStates+s+1]+=gamma_Ts1;
	   
				double gamma_Ts2 = gamma_T[s+2];
				//if you use real gamma you have to divide by ct[t-1]
				double gamma_sums2 = gamma_sum[s+2];
				double gamma_tot2 = gamma_Ts2 + gamma_sums2 /* /ct[T-1] */;
				gamma_T[s+2] = 1./gamma_tot2;
				gamma_sum[s+2] = 1./gamma_sums2;
		        	b_new[yt*hiddenStates+s+2]+=gamma_Ts2;
	   
				double gamma_Ts3 = gamma_T[s+3];
				//if you use real gamma you have to divide by ct[t-1]
				double gamma_sums3 = gamma_sum[s+3];
				double gamma_tot3 = gamma_Ts3 + gamma_sums3 /* /ct[T-1] */;
				gamma_T[s+3] = 1./gamma_tot3;
				gamma_sum[s+3] = 1./gamma_sums3;
		        	b_new[yt*hiddenStates+s+3]+=gamma_Ts3;
	       	 }

		        //compute new emission matrix
		        for(int v = 0; v < differentObservables; v+=4){
			        for(int s = 0; s < hiddenStates; s+=4){
					double gamma_sums_inv=gamma_T[s];
					double gamma_sums_inv1=gamma_T[s+1];
					double gamma_sums_inv2=gamma_T[s+2];
					double gamma_sums_inv3=gamma_T[s+3];
					//v,s
				        emissionMatrix[v*hiddenStates + s] = b_new[v*hiddenStates + s] * gamma_sums_inv;
				        b_new[v*hiddenStates + s] = 0.0;
					//v,s+1
				        emissionMatrix[v*hiddenStates + s+1] = b_new[v*hiddenStates + s+1] * gamma_sums_inv1;
				        b_new[v*hiddenStates + s+1] = 0.0;
					//v,s+2
				        emissionMatrix[v*hiddenStates + s+2] = b_new[v*hiddenStates + s+2] * gamma_sums_inv2;
				        b_new[v*hiddenStates + s+2] = 0.0;
					//v,s+3
				        emissionMatrix[v*hiddenStates + s+3] = b_new[v*hiddenStates + s+3] * gamma_sums_inv3;
				        b_new[v*hiddenStates + s+3] = 0.0;
					
					//v+1,s
				        emissionMatrix[(v+1)*hiddenStates + s] = b_new[(v+1)*hiddenStates + s] * gamma_sums_inv;
				        b_new[(v+1)*hiddenStates + s] = 0.0;
					//v+1,s+1
				        emissionMatrix[(v+1)*hiddenStates + s+1] = b_new[(v+1)*hiddenStates + s+1] * gamma_sums_inv1;
				        b_new[(v+1)*hiddenStates + s+1] = 0.0;
					//v+1,s+2
				        emissionMatrix[(v+1)*hiddenStates + s+2] = b_new[(v+1)*hiddenStates + s+2] * gamma_sums_inv2;
				        b_new[(v+1)*hiddenStates + s+2] = 0.0;
					//v+1,s+3
				        emissionMatrix[(v+1)*hiddenStates + s+3] = b_new[(v+1)*hiddenStates + s+3] * gamma_sums_inv3;
				        b_new[(v+1)*hiddenStates + s+3] = 0.0;
					
					//v+2,s
				        emissionMatrix[(v+2)*hiddenStates + s] = b_new[(v+2)*hiddenStates + s] * gamma_sums_inv;
				        b_new[(v+2)*hiddenStates + s] = 0.0;
					//v+2,s+1
				        emissionMatrix[(v+2)*hiddenStates + s+1] = b_new[(v+2)*hiddenStates + s+1] * gamma_sums_inv1;
				        b_new[(v+2)*hiddenStates + s+1] = 0.0;
					//v+2,s+2
				        emissionMatrix[(v+2)*hiddenStates + s+2] = b_new[(v+2)*hiddenStates + s+2] * gamma_sums_inv2;
				        b_new[(v+2)*hiddenStates + s+2] = 0.0;
					//v+2,s+3
				        emissionMatrix[(v+2)*hiddenStates + s+3] = b_new[(v+2)*hiddenStates + s+3] * gamma_sums_inv3;
				        b_new[(v+2)*hiddenStates + s+3] = 0.0;
					
					//v+3,s
				        emissionMatrix[(v+3)*hiddenStates + s] = b_new[(v+3)*hiddenStates + s] * gamma_sums_inv;
				        b_new[(v+3)*hiddenStates + s] = 0.0;
					//v+3,s+1
				        emissionMatrix[(v+3)*hiddenStates + s+1] = b_new[(v+3)*hiddenStates + s+1] * gamma_sums_inv1;
				        b_new[(v+3)*hiddenStates + s+1] = 0.0;
					//v+3,s+2
				        emissionMatrix[(v+3)*hiddenStates + s+2] = b_new[(v+3)*hiddenStates + s+2] * gamma_sums_inv2;
				        b_new[(v+3)*hiddenStates + s+2] = 0.0;
					//v+3,s+3
				        emissionMatrix[(v+3)*hiddenStates + s+3] = b_new[(v+3)*hiddenStates + s+3] * gamma_sums_inv3 ;
				        b_new[(v+3)*hiddenStates + s+3] = 0.0;
			        }
		        }

		        //FORWARD

		        double ctt = 0.0;
		        //compute alpha(0) and scaling factor for t = 0
		        int y0 = observations[0];
		        for(int s = 0; s < hiddenStates; s+=4){
			//s
			        double alphas = stateProb[s] * emissionMatrix[y0*hiddenStates + s];
			        ctt += alphas;
			        alpha[s] = alphas;
					//s+1
			        double alphas1 = stateProb[s+1] * emissionMatrix[y0*hiddenStates + s+1];
			        ctt += alphas1;
			        alpha[s+1] = alphas1;
					//s+2
			        double alphas2 = stateProb[s+2] * emissionMatrix[y0*hiddenStates + s+2];
			        ctt += alphas2;
			        alpha[s+2] = alphas2;
					//s+3
			        double alphas3 = stateProb[s+3] * emissionMatrix[y0*hiddenStates + s+3];
			        ctt += alphas3;
			        alpha[s+3] = alphas3;
		        }
		        
		        ctt = 1.0 / ctt;
	
		        //scale alpha(0)
		        for(int s = 0; s < hiddenStates; s+=4){
			        alpha[s] *= ctt;
			        alpha[s+1] *= ctt;
			        alpha[s+2] *= ctt;
			        alpha[s+3] *= ctt;
		        }
		        //print_matrix(alpha,N,T);
		        ct[0] = ctt;
	
		        //Compute alpha(1) and scale transitionMatrix
		        ctt = 0.0;	
		        yt = observations[1];	
		        for(int s = 0; s<hiddenStates; s+=4){// s=new_state
			        double alphatNs = 0;
			        double alphatNs1 = 0;
			        double alphatNs2 = 0;
			        double alphatNs3 = 0;
			        for(int j = 0; j < hiddenStates; j+=4){//j=old_states
					double gamma_sums_inv = gamma_sum[j];
					double gamma_sums_inv1 = gamma_sum[j+1];
					double gamma_sums_inv2 = gamma_sum[j+2];
					double gamma_sums_inv3 = gamma_sum[j+3];
					double alphaFactor = alpha[0*hiddenStates + j];
					double alphaFactor1 = alpha[0*hiddenStates + j+1];
					double alphaFactor2 = alpha[0*hiddenStates + j+2];
					double alphaFactor3 = alpha[0*hiddenStates + j+3];
						//s
						//j
				        double aj0Ns0 =  a_new[j*hiddenStates + s] * gamma_sums_inv;
				        transitionMatrix[j*hiddenStates + s] = aj0Ns0;
				        alphatNs += alphaFactor * aj0Ns0;
				        a_new[j*hiddenStates + s] = 0.0;
						//j+1
				        double aj1Ns0 =  a_new[(j+1)*hiddenStates + s] * gamma_sums_inv1;
				        transitionMatrix[(j+1)*hiddenStates + s] = aj1Ns0;
				        alphatNs += alphaFactor1 * aj1Ns0;
				        a_new[(j+1)*hiddenStates + s] = 0.0;
						//j+2
				        double aj2Ns0 =  a_new[(j+2)*hiddenStates + s] * gamma_sums_inv2;
				        transitionMatrix[(j+2)*hiddenStates + s] = aj2Ns0;
				        alphatNs += alphaFactor2 * aj2Ns0;
				        a_new[(j+2)*hiddenStates + s] = 0.0;
						//j+3
				        double aj3Ns0 =  a_new[(j+3)*hiddenStates + s] * gamma_sums_inv3;
				        transitionMatrix[(j+3)*hiddenStates + s] = aj3Ns0;
				        alphatNs += alphaFactor3 * aj3Ns0;
				        a_new[(j+3)*hiddenStates + s] = 0.0;
						
						//s+1
						//j
				        double aj0Ns1 =  a_new[j*hiddenStates + s+1] * gamma_sums_inv;
				        transitionMatrix[j*hiddenStates + s+1] = aj0Ns1;
				        alphatNs1 += alphaFactor * aj0Ns1;
				        a_new[j*hiddenStates + s+1] = 0.0;
						//j+1
				        double aj1Ns1 =  a_new[(j+1)*hiddenStates + s+1] * gamma_sums_inv1;
				        transitionMatrix[(j+1)*hiddenStates + s+1] = aj1Ns1;
				        alphatNs1 += alphaFactor1 * aj1Ns1;
				        a_new[(j+1)*hiddenStates + s+1] = 0.0;
						//j+2
				        double aj2Ns1 =  a_new[(j+2)*hiddenStates + s+1] * gamma_sums_inv2;
				        transitionMatrix[(j+2)*hiddenStates + s+1] = aj2Ns1;
				        alphatNs1 += alphaFactor2 * aj2Ns1;
				        a_new[(j+2)*hiddenStates + s+1] = 0.0;
						//j+3
				        double aj3Ns1 =  a_new[(j+3)*hiddenStates + s+1] * gamma_sums_inv3;
				        transitionMatrix[(j+3)*hiddenStates + s+1] = aj3Ns1;
				        alphatNs1 += alphaFactor3 * aj3Ns1;
				        a_new[(j+3)*hiddenStates + s+1] = 0.0;
						
						//s+2
						//j
				        double aj0Ns2 =  a_new[j*hiddenStates + s+2] * gamma_sums_inv;
				        transitionMatrix[j*hiddenStates + s+2] = aj0Ns2;
				        alphatNs2 += alphaFactor * aj0Ns2;
				        a_new[j*hiddenStates + s+2] = 0.0;
						//j+1
				        double aj1Ns2 =  a_new[(j+1)*hiddenStates + s+2] * gamma_sums_inv1;
				        transitionMatrix[(j+1)*hiddenStates + s+2] = aj1Ns2;
				        alphatNs2 += alphaFactor1 * aj1Ns2;
				        a_new[(j+1)*hiddenStates + s+2] = 0.0;
						//j+2
				        double aj2Ns2 =  a_new[(j+2)*hiddenStates + s+2] * gamma_sums_inv2;
				        transitionMatrix[(j+2)*hiddenStates + s+2] = aj2Ns2;
				        alphatNs2 += alphaFactor2 * aj2Ns2;
				        a_new[(j+2)*hiddenStates + s+2] = 0.0;
						//j+3
				        double aj3Ns2 =  a_new[(j+3)*hiddenStates + s+2] * gamma_sums_inv3;
				        transitionMatrix[(j+3)*hiddenStates + s+2] = aj3Ns2;
				        alphatNs2 += alphaFactor3 * aj3Ns2;
				        a_new[(j+3)*hiddenStates + s+2] = 0.0;
					
					//s+3
					//j
				        double aj0Ns3 =  a_new[j*hiddenStates + s+3] * gamma_sums_inv;
				        transitionMatrix[j*hiddenStates + s+3] = aj0Ns3;
				        alphatNs3 += alphaFactor * aj0Ns3;
				        a_new[j*hiddenStates + s+3] = 0.0;
						//j+1
				        double aj1Ns3 =  a_new[(j+1)*hiddenStates + s+3] * gamma_sums_inv1;
				        transitionMatrix[(j+1)*hiddenStates + s+3] = aj1Ns3;
				        alphatNs3 += alphaFactor1 * aj1Ns3;
				        a_new[(j+1)*hiddenStates + s+3] = 0.0;
						//j+2
				        double aj2Ns3 =  a_new[(j+2)*hiddenStates + s+3] * gamma_sums_inv2;
				        transitionMatrix[(j+2)*hiddenStates + s+3] = aj2Ns3;
				        alphatNs3 += alphaFactor2 * aj2Ns3;
				        a_new[(j+2)*hiddenStates + s+3] = 0.0;
						//j+3
				        double aj3Ns3 =  a_new[(j+3)*hiddenStates + s+3] * gamma_sums_inv3;
				        transitionMatrix[(j+3)*hiddenStates + s+3] = aj3Ns3;
				        alphatNs3 += alphaFactor3 * aj3Ns3;
				        a_new[(j+3)*hiddenStates + s+3] = 0.0;
			        }
					//s
			        alphatNs *= emissionMatrix[yt*hiddenStates + s];
			        ctt += alphatNs;
			        alpha[1*hiddenStates + s] = alphatNs;
					//s+1
			        alphatNs1 *= emissionMatrix[yt*hiddenStates + s+1];
			        ctt += alphatNs1;
			        alpha[1*hiddenStates + s+1] = alphatNs1;
					//s+2
			        alphatNs2 *= emissionMatrix[yt*hiddenStates + s+2];
			        ctt += alphatNs2;
			        alpha[1*hiddenStates + s+2] = alphatNs2;
					//s+3
			        alphatNs3 *= emissionMatrix[yt*hiddenStates + s+3];
			        ctt += alphatNs3;
			        alpha[1*hiddenStates + s+3] = alphatNs3;
	       	}
		        //scaling factor for t 
		        ctt = 1.0 / ctt;
		        
		        //scale alpha(t)
		        for(int s = 0; s<hiddenStates; s+=4){// s=new_state
			        alpha[1*hiddenStates + s] *= ctt;
			        alpha[1*hiddenStates + s+1] *= ctt;
			        alpha[1*hiddenStates + s+2] *= ctt;
			        alpha[1*hiddenStates + s+3] *= ctt;
		        }
		        ct[1] = ctt;
	
				
		        for(int t = 2; t < T-1; t++){
			        ctt = 0.0;	
			        yt = observations[t];	
			        for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				        double alphatNs = 0;
				        double alphatNs1 = 0;
				        double alphatNs2 = 0;
				        double alphatNs3 = 0;
				        //alpha[s*T + t] = 0;
				        for(int j = 0; j < hiddenStates; j+=4){//j=old_states
						double alphaFactor=alpha[(t-1)*hiddenStates + j];
						double alphaFactor1=alpha[(t-1)*hiddenStates + j+1];
						double alphaFactor2=alpha[(t-1)*hiddenStates + j+2];
						double alphaFactor3=alpha[(t-1)*hiddenStates + j+3];
							//s
					        alphatNs += alphaFactor * transitionMatrix[j*hiddenStates + s];
					        alphatNs += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s];
					        alphatNs += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s];
					        alphatNs += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s];
							//s+1
					        alphatNs1 += alphaFactor * transitionMatrix[j*hiddenStates + s+1];
					        alphatNs1 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+1];
					        alphatNs1 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+1];
					        alphatNs1 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+1];
							//s+2
					        alphatNs2 += alphaFactor * transitionMatrix[j*hiddenStates + s+2];
					        alphatNs2 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+2];
					        alphatNs2 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+2];
					        alphatNs2 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+2];
							//s+3
					        alphatNs3 += alphaFactor * transitionMatrix[j*hiddenStates + s+3];
					        alphatNs3 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+3];
					        alphatNs3 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+3];
					        alphatNs3 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+3];
				        }
						//s
				        alphatNs *= emissionMatrix[yt*hiddenStates + s];
				        ctt += alphatNs;
				        alpha[t*hiddenStates + s] = alphatNs;
						
						//s+1
				        alphatNs1 *= emissionMatrix[yt*hiddenStates + s+1];
				        ctt += alphatNs1;
				        alpha[t*hiddenStates + s+1] = alphatNs1;
						
						//s+2
				        alphatNs2 *= emissionMatrix[yt*hiddenStates + s+2];
				        ctt += alphatNs2;
				        alpha[t*hiddenStates + s+2] = alphatNs2;
						
						//s+3
				        alphatNs3 *= emissionMatrix[yt*hiddenStates + s+3];
				        ctt += alphatNs3;
				        alpha[t*hiddenStates + s+3] = alphatNs3;
						
						
			        }
			        //scaling factor for t 
			        ctt = 1.0 / ctt;
			        
			        //scale alpha(t)
			        for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				        alpha[t*hiddenStates + s] *= ctt;
				        alpha[t*hiddenStates + s+1] *= ctt;
				        alpha[t*hiddenStates + s+2] *= ctt;
				        alpha[t*hiddenStates + s+3] *= ctt;
			        }
			        ct[t] = ctt;
		        }
				
				
		        //compute alpha(T-1)
		        ctt = 0.0;	
		        yt = observations[T-1];	
		        for(int s = 0; s<hiddenStates; s+=4){// s=new_state
			        double alphatNs = 0;
			        double alphatNs1 = 0;
			        double alphatNs2 = 0;
			        double alphatNs3 = 0;
			        //alpha[s*T + t] = 0;
			        for(int j = 0; j < hiddenStates; j+=4){//j=old_states
						double alphaFactor=alpha[(T-2)*hiddenStates + j];
						double alphaFactor1=alpha[(T-2)*hiddenStates + j+1];
						double alphaFactor2=alpha[(T-2)*hiddenStates + j+2];
						double alphaFactor3=alpha[(T-2)*hiddenStates + j+3];
							//s
					        alphatNs += alphaFactor * transitionMatrix[j*hiddenStates + s];
					        alphatNs += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s];
					        alphatNs += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s];
					        alphatNs += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s];
							//s+1
					        alphatNs1 += alphaFactor * transitionMatrix[j*hiddenStates + s+1];
					        alphatNs1 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+1];
					        alphatNs1 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+1];
					        alphatNs1 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+1];
							//s+2
					        alphatNs2 += alphaFactor * transitionMatrix[j*hiddenStates + s+2];
					        alphatNs2 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+2];
					        alphatNs2 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+2];
					        alphatNs2 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+2];
							//s+3
					        alphatNs3 += alphaFactor * transitionMatrix[j*hiddenStates + s+3];
					        alphatNs3 += alphaFactor1 * transitionMatrix[(j+1)*hiddenStates + s+3];
					        alphatNs3 += alphaFactor2 * transitionMatrix[(j+2)*hiddenStates + s+3];
					        alphatNs3 += alphaFactor3 * transitionMatrix[(j+3)*hiddenStates + s+3];
			        }
					//s
			        alphatNs *= emissionMatrix[yt*hiddenStates + s];
			        ctt += alphatNs;
			        alpha[(T-1)*hiddenStates + s] = alphatNs;
					
					//s+1
			        alphatNs1 *= emissionMatrix[yt*hiddenStates + s+1];
			        ctt += alphatNs1;
			        alpha[(T-1)*hiddenStates + s+1] = alphatNs1;
					
					//s+2
			        alphatNs2 *= emissionMatrix[yt*hiddenStates + s+2];
			        ctt += alphatNs2;
			        alpha[(T-1)*hiddenStates + s+2] = alphatNs2;
					
					//s+3
			        alphatNs3 *= emissionMatrix[yt*hiddenStates + s+3];
			        ctt += alphatNs3;
			        alpha[(T-1)*hiddenStates + s+3] = alphatNs3;
		        }
		        //scaling factor for T-1
		        ctt = 1.0 / ctt;
			        
		        //scale alpha(t)
		        for(int s = 0; s<hiddenStates; s+=4){// s=new_state
					//s
			        double alphaT1Ns = alpha[(T-1) * hiddenStates + s]*ctt;
			        alpha[(T-1)*hiddenStates+s] = alphaT1Ns;
			        gamma_T[s] = alphaT1Ns /* *ct[T-1]*/;
					//s+1
			        alphaT1Ns = alpha[(T-1) * hiddenStates + s+1]*ctt;
			        alpha[(T-1)*hiddenStates + s+1] = alphaT1Ns;
			        gamma_T[s+1] = alphaT1Ns /* *ct[T-1]*/;
					//s+2
			        alphaT1Ns = alpha[(T-1) * hiddenStates + s+2]*ctt;
			        alpha[(T-1)*hiddenStates + s+2] = alphaT1Ns;
			        gamma_T[s+2] = alphaT1Ns /* *ct[T-1]*/;
					//s+3
			        alphaT1Ns = alpha[(T-1) * hiddenStates + s+3]*ctt;
			        alpha[(T-1)*hiddenStates + s+3] = alphaT1Ns;
			        gamma_T[s+3] = alphaT1Ns /* *ct[T-1]*/;
		        }
		        ct[T-1] = ctt;
	
		        //FUSED BACKWARD and UPDATE STEP
	
		        for(int s = 0; s < hiddenStates; s+=4){
			        beta[s] = /* 1* */ctt;
			        gamma_sum[s] = 0.0;
			        beta[s+1] = /* 1* */ctt;
			        gamma_sum[s+1] = 0.0;
			        beta[s+2] = /* 1* */ctt;
			        gamma_sum[s+2] = 0.0;
			        beta[s+3] = /* 1* */ctt;
			        gamma_sum[s+3] = 0.0;
		        }
		        
		        
		        
		        
		        	
			for(int v = 0; v < differentObservables; v++){
				for(int s = 0; s < hiddenStates; s+=4){
					for(int j = 0; j < hiddenStates; j+=4){
						ab[(v*hiddenStates + s) * hiddenStates + j] = transitionMatrix[s*hiddenStates + j] * emissionMatrix[v*hiddenStates +j];
						ab[(v*hiddenStates + s) * hiddenStates + j+1] = transitionMatrix[s*hiddenStates + j+1] * emissionMatrix[v*hiddenStates +j+1];
						ab[(v*hiddenStates + s) * hiddenStates + j+2] = transitionMatrix[s*hiddenStates + j+2] * emissionMatrix[v*hiddenStates +j+2];
						ab[(v*hiddenStates + s) * hiddenStates + j+3] = transitionMatrix[s*hiddenStates + j+3] * emissionMatrix[v*hiddenStates +j+3];
						
						ab[(v*hiddenStates + s+1) * hiddenStates + j] = transitionMatrix[(s+1)*hiddenStates + j] * emissionMatrix[v*hiddenStates +j];
						ab[(v*hiddenStates + s+1) * hiddenStates + j+1] = transitionMatrix[(s+1)*hiddenStates + j+1] * emissionMatrix[v*hiddenStates +j+1];
						ab[(v*hiddenStates + s+1) * hiddenStates + j+2] = transitionMatrix[(s+1)*hiddenStates + j+2] * emissionMatrix[v*hiddenStates +j+2];
						ab[(v*hiddenStates + s+1) * hiddenStates + j+3] = transitionMatrix[(s+1)*hiddenStates + j+3] * emissionMatrix[v*hiddenStates +j+3];
							
						ab[(v*hiddenStates + s+2) * hiddenStates + j] = transitionMatrix[(s+2)*hiddenStates + j] * emissionMatrix[v*hiddenStates +j];
						ab[(v*hiddenStates + s+2) * hiddenStates + j+1] = transitionMatrix[(s+2)*hiddenStates + j+1] * emissionMatrix[v*hiddenStates +j+1];
						ab[(v*hiddenStates + s+2) * hiddenStates + j+2] = transitionMatrix[(s+2)*hiddenStates + j+2] * emissionMatrix[v*hiddenStates +j+2];
						ab[(v*hiddenStates + s+2) * hiddenStates + j+3] = transitionMatrix[(s+2)*hiddenStates + j+3] * emissionMatrix[v*hiddenStates +j+3];
						
						ab[(v*hiddenStates + s+3) * hiddenStates + j] = transitionMatrix[(s+3)*hiddenStates + j] * emissionMatrix[v*hiddenStates +j];
						ab[(v*hiddenStates + s+3) * hiddenStates + j+1] = transitionMatrix[(s+3)*hiddenStates + j+1] * emissionMatrix[v*hiddenStates +j+1];
						ab[(v*hiddenStates + s+3) * hiddenStates + j+2] = transitionMatrix[(s+3)*hiddenStates + j+2] * emissionMatrix[v*hiddenStates +j+2];
						ab[(v*hiddenStates + s+3) * hiddenStates + j+3] = transitionMatrix[(s+3)*hiddenStates + j+3] * emissionMatrix[v*hiddenStates +j+3];
						
							
					}
				}
			}
		        
		        
		        
		        
		        
	
		        	//compute sum of xi and gamma from t= 0...T-2
        	  	yt = observations[T-1];
			for(int t = T-1; t > 0; t--){
				const int yt1 = observations[t-1];
				const double ctt = ct[t-1];
				for(int s = 0; s < hiddenStates ; s+=4){
					double beta_news0 = 0.0;
					double alphat1Ns0 = alpha[(t-1)*hiddenStates + s];
					double beta_news1 = 0.0;
					double alphat1Ns1 = alpha[(t-1)*hiddenStates + s+1];
					double beta_news2 = 0.0;
					double alphat1Ns2 = alpha[(t-1)*hiddenStates + s+2];
					double beta_news3 = 0.0;
					double alphat1Ns3 = alpha[(t-1)*hiddenStates + s+3];
					for(int j = 0; j < hiddenStates; j+=4){
						
            					double beta0 = beta[j];
						double beta1 = beta[j+1];
						double beta2 = beta[j+2];
						double beta3 = beta[j+3];
					
						//s
						//j
						double temp00 = ab[(yt*hiddenStates + s)*hiddenStates + j] * beta0;
						a_new[s*hiddenStates+j] += alphat1Ns0 * temp00;
						beta_news0 += temp00;
						
						//j+1
						double temp01 = ab[(yt*hiddenStates + s)*hiddenStates + j+1] * beta1;
						a_new[s*hiddenStates+j+1] += alphat1Ns0 * temp01;
						beta_news0 += temp01;
						
										
						//j+2
						double temp02 = ab[(yt*hiddenStates + s)*hiddenStates + j+2] * beta2;
						a_new[s*hiddenStates+j+2] += alphat1Ns0 * temp02;
						beta_news0 += temp02;
						
						//j+3
						double temp03 = ab[(yt*hiddenStates + s)*hiddenStates + j+3] * beta3;
						a_new[s*hiddenStates+j+3] += alphat1Ns0 * temp03;
						beta_news0 += temp03;
						
						
						//s+1
						
						//j
						double temp10 = ab[(yt*hiddenStates + s+1)*hiddenStates + j] * beta0;
						a_new[(s+1)*hiddenStates+j] += alphat1Ns1 * temp10;
						beta_news1 += temp10;
						
						//j+1
						double temp11 = ab[(yt*hiddenStates + s+1)*hiddenStates + j+1] * beta1;
						a_new[(s+1)*hiddenStates+j+1] += alphat1Ns1 * temp11;
						beta_news1 += temp11;
						
										
						//j+2
						double temp12 = ab[(yt*hiddenStates + s+1)*hiddenStates + j+2] * beta2;
						a_new[(s+1)*hiddenStates+j+2] += alphat1Ns1 * temp12;
						beta_news1 += temp12;
						
						//j+3
						double temp13 = ab[(yt*hiddenStates + s+1)*hiddenStates + j+3] * beta3;
						a_new[(s+1)*hiddenStates+j+3] += alphat1Ns1 * temp13;
						beta_news1 += temp13;
						
						//s+2
					
						//j
						double temp20 = ab[(yt*hiddenStates + s+2)*hiddenStates + j] * beta0;
						a_new[(s+2)*hiddenStates+j] += alphat1Ns2 * temp20;
						beta_news2 += temp20;
						
						//j+1
						double temp21 = ab[(yt*hiddenStates + s+2)*hiddenStates + j+1] * beta1;
						a_new[(s+2)*hiddenStates+j+1] += alphat1Ns2 * temp21;
						beta_news2 += temp21;
						
										
						//j+2
						double temp22 = ab[(yt*hiddenStates + s+2)*hiddenStates + j+2] * beta2;
						a_new[(s+2)*hiddenStates+j+2] += alphat1Ns2 * temp22;
						beta_news2 += temp22;
						
						//j+3
						double temp23 = ab[(yt*hiddenStates + s+2)*hiddenStates + j+3] * beta3;
						a_new[(s+2)*hiddenStates+j+3] += alphat1Ns2 * temp23;
						beta_news2 += temp23;
						
						//s+3
						
						//j
						double temp30 = ab[(yt*hiddenStates + s+3)*hiddenStates + j] * beta0;
						a_new[(s+3)*hiddenStates+j] += alphat1Ns3 * temp30;
						beta_news3 += temp30;
						
						//j+1
						double temp31 = ab[(yt*hiddenStates + s+3)*hiddenStates + j+1] * beta1;
						a_new[(s+3)*hiddenStates+j+1] += alphat1Ns3 * temp31;
						beta_news3 += temp31;
						
										
						//j+2
						double temp32 = ab[(yt*hiddenStates + s+3)*hiddenStates + j+2] * beta2;
						a_new[(s+3)*hiddenStates+j+2] += alphat1Ns3 * temp32;
						beta_news3 += temp32;
						
						//j+3
						double temp33 = ab[(yt*hiddenStates + s+3)*hiddenStates + j+3] * beta3;
						a_new[(s+3)*hiddenStates+j+3] += alphat1Ns3 * temp33;
						beta_news3 += temp33;
					}
					//s
					double ps0 =alphat1Ns0*beta_news0/* *ct[t-1]*/;  
					stateProb[s] = ps0;
					beta_new[s] = beta_news0*ctt;
					//if you use real gamma you have to divide with ct[t-1]
					gamma_sum[s]+= ps0 /* /ct[t-1] */ ;
            				b_new[yt1*hiddenStates+s]+=ps0;
            				
            				//s+1
					double ps1 =alphat1Ns1*beta_news1/* *ct[t-1]*/;  
					stateProb[s+1] = ps1;
					beta_new[s+1] = beta_news1*ctt;
					//if you use real gamma you have to divide with ct[t-1]
					gamma_sum[s+1]+= ps1 /* /ct[t-1] */ ;
        				b_new[yt1*hiddenStates+ s+1]+=ps1;
					
					//s+2
					double ps2 =alphat1Ns2*beta_news2/* *ct[t-1]*/;  
					stateProb[s+2] = ps2;
					beta_new[s+2] = beta_news2*ctt;
					//if you use real gamma you have to divide with ct[t-1]
					gamma_sum[s+2]+= ps2 /* /ct[t-1] */ ;
        				b_new[yt1*hiddenStates+ s+2]+=ps2;
				
				
					//s+3
					double ps3 =alphat1Ns3*beta_news3/* *ct[t-1]*/;  
					stateProb[s+3] = ps3;
					beta_new[s+3] = beta_news3*ctt;
					//if you use real gamma you have to divide with ct[t-1]
					gamma_sum[s+3]+= ps3 /* /ct[t-1] */ ;
        				b_new[yt1*hiddenStates+ s+3]+=ps3;
				
		
				}	
				double * temp = beta_new;
				beta_new = beta;
				beta = temp;
        			yt=yt1;
	        	}
            		steps+=1;
			
//FINISHING
	        //log likelihood
	        double oldLogLikelihood=logLikelihood;

	        double newLogLikelihood = 0.0;
	        //evidence with alpha only:

	        for(int time = 0; time < T; time++){
		        newLogLikelihood -= log2(ct[time]);
	        }
	        
	        logLikelihood=newLogLikelihood;

	        disparance=newLogLikelihood-oldLogLikelihood;

		}while (disparance>EPSILON && steps<maxSteps);
//Final Scaling




		yt = observations[T-1];
		//add remaining parts of the sum of gamma 
		for(int s = 0; s < hiddenStates; s+=4){
			double gamma_Ts0 = gamma_T[s];
			//if you use real gamma you have to divide by ct[t-1]
			double gamma_sums0 = gamma_sum[s];
			double gamma_tot0 = gamma_Ts0 + gamma_sums0 /* /ct[T-1] */;
			gamma_T[s] = 1./gamma_tot0;
			gamma_sum[s] = 1./gamma_sums0;
	       	b_new[yt*hiddenStates+s]+=gamma_Ts0;
	   
			double gamma_Ts1 = gamma_T[s+1];
			//if you use real gamma you have to divide by ct[t-1]
			double gamma_sums1 = gamma_sum[s+1];
			double gamma_tot1 = gamma_Ts1 + gamma_sums1 /* /ct[T-1] */;
			gamma_T[s+1] = 1./gamma_tot1;
			gamma_sum[s+1] = 1./gamma_sums1;
	        	b_new[yt*hiddenStates+s+1]+=gamma_Ts1;
	  
			double gamma_Ts2 = gamma_T[s+2];
			//if you use real gamma you have to divide by ct[t-1]
			double gamma_sums2 = gamma_sum[s+2];
			double gamma_tot2 = gamma_Ts2 + gamma_sums2 /* /ct[T-1] */;
			gamma_T[s+2] = 1./gamma_tot2;
			gamma_sum[s+2] = 1./gamma_sums2;
	        	b_new[yt*hiddenStates+s+2]+=gamma_Ts2;
	  
			double gamma_Ts3 = gamma_T[s+3];
			//if you use real gamma you have to divide by ct[t-1]
			double gamma_sums3 = gamma_sum[s+3];
			double gamma_tot3 = gamma_Ts3 + gamma_sums3 /* /ct[T-1] */;
			gamma_T[s+3] = 1./gamma_tot3;
			gamma_sum[s+3] = 1./gamma_sums3;
	        	b_new[yt*hiddenStates+s+3]+=gamma_Ts3;
	  
		}

	    	//compute new transition matrix
	    	for(int s = 0; s < hiddenStates; s+=4){
			double gamma_sums_inv = gamma_sum[s];
			double gamma_sums_inv1 = gamma_sum[s+1];
			double gamma_sums_inv2 = gamma_sum[s+2];
		    	double gamma_sums_inv3 = gamma_sum[s+3];
		    	for(int j = 0; j < hiddenStates; j+=4){
				//s
				transitionMatrix[s*hiddenStates + j] = a_new[s*hiddenStates + j]*gamma_sums_inv;
				transitionMatrix[s*hiddenStates + j+1] = a_new[s*hiddenStates + j+1]*gamma_sums_inv;
				transitionMatrix[s*hiddenStates + j+2] = a_new[s*hiddenStates + j+2]*gamma_sums_inv;
				transitionMatrix[s*hiddenStates + j+3] = a_new[s*hiddenStates + j+3]*gamma_sums_inv;
				//s+1
				transitionMatrix[(s+1)*hiddenStates + j] = a_new[(s+1)*hiddenStates + j]*gamma_sums_inv1;
				transitionMatrix[(s+1)*hiddenStates + j+1] = a_new[(s+1)*hiddenStates + j+1]*gamma_sums_inv1;
				transitionMatrix[(s+1)*hiddenStates + j+2] = a_new[(s+1)*hiddenStates + j+2]*gamma_sums_inv1;
				transitionMatrix[(s+1)*hiddenStates + j+3] = a_new[(s+1)*hiddenStates + j+3]*gamma_sums_inv1;
				//s+2
				transitionMatrix[(s+2)*hiddenStates + j] = a_new[(s+2)*hiddenStates + j]*gamma_sums_inv2;
			    	transitionMatrix[(s+2)*hiddenStates + j+1] = a_new[(s+2)*hiddenStates + j+1]*gamma_sums_inv2;
			    	transitionMatrix[(s+2)*hiddenStates + j+2] = a_new[(s+2)*hiddenStates + j+2]*gamma_sums_inv2;
			    	transitionMatrix[(s+2)*hiddenStates + j+3] = a_new[(s+2)*hiddenStates + j+3]*gamma_sums_inv2;
				//s+3
			    	transitionMatrix[(s+3)*hiddenStates + j] = a_new[(s+3)*hiddenStates + j]*gamma_sums_inv3;
			    	transitionMatrix[(s+3)*hiddenStates + j+1] = a_new[(s+3)*hiddenStates + j+1]*gamma_sums_inv3;
			    	transitionMatrix[(s+3)*hiddenStates + j+2] = a_new[(s+3)*hiddenStates + j+2]*gamma_sums_inv3;
			    	transitionMatrix[(s+3)*hiddenStates + j+3] = a_new[(s+3)*hiddenStates + j+3]*gamma_sums_inv3;
		   	 }
	    	}
		//compute new emission matrix
		for(int v = 0; v < differentObservables; v+=4){
			for(int s = 0; s < hiddenStates; s+=4){
			
				double gamma_T0 = gamma_T[s];
				double gamma_T1 = gamma_T[s+1];
				double gamma_T2 = gamma_T[s+2];
				double gamma_T3 = gamma_T[s+3];
				
				emissionMatrix[v*hiddenStates + s] = b_new[v*hiddenStates + s] * gamma_T0;
				
				emissionMatrix[v*hiddenStates + s+1] = b_new[v*hiddenStates + s+1] * gamma_T1;
				
				emissionMatrix[v*hiddenStates + s+2] = b_new[v*hiddenStates + s+2] * gamma_T2;
					
				emissionMatrix[v*hiddenStates + s+3] = b_new[v*hiddenStates + s+3] * gamma_T3;
				
				
				
				emissionMatrix[(v+1)*hiddenStates + s] = b_new[(v+1)*hiddenStates + s] * gamma_T0;
				
				emissionMatrix[(v+1)*hiddenStates + s+1] = b_new[(v+1)*hiddenStates + s+1] * gamma_T1;
					
				emissionMatrix[(v+1)*hiddenStates + s+2] = b_new[(v+1)*hiddenStates + s+2] * gamma_T2;
				
				emissionMatrix[(v+1)*hiddenStates + s+3] = b_new[(v+1)*hiddenStates + s+3] * gamma_T3;
				
				
				
				emissionMatrix[(v+2)*hiddenStates + s] = b_new[(v+2)*hiddenStates + s] * gamma_T0;
				
				emissionMatrix[(v+2)*hiddenStates + s+1] = b_new[(v+2)*hiddenStates + s+1] * gamma_T1;
					
				emissionMatrix[(v+2)*hiddenStates + s+2] = b_new[(v+2)*hiddenStates + s+2] * gamma_T2;
					
				emissionMatrix[(v+2)*hiddenStates + s+3] = b_new[(v+2)*hiddenStates + s+3] * gamma_T3;
				
					
				emissionMatrix[(v+3)*hiddenStates + s] = b_new[(v+3)*hiddenStates + s] * gamma_T0;
				
				emissionMatrix[(v+3)*hiddenStates + s+1] = b_new[(v+3)*hiddenStates + s+1] * gamma_T1;
					
				emissionMatrix[(v+3)*hiddenStates + s+2] = b_new[(v+3)*hiddenStates + s+2] * gamma_T2;
				
				emissionMatrix[(v+3)*hiddenStates + s+3] = b_new[(v+3)*hiddenStates + s+3] * gamma_T3;
	
					
					
		
				}
			}
		
		






	    
		cycles = stop_tsc(start);
        	cycles = cycles/steps;

		transpose(emissionMatrix,differentObservables,hiddenStates);

		/*
		//Show results
		printf(" %i \n", steps);
		print_matrix(transitionMatrix,hiddenStates,hiddenStates);
		print_matrix(emissionMatrix, hiddenStates,differentObservables);
		print_vector(stateProb, hiddenStates);
		*/


		//emissionMatrix is not in state major order
		transpose(emissionMatrixTesting, differentObservables,hiddenStates);
        	tested_implementation(hiddenStates, differentObservables, T, transitionMatrixTesting, emissionMatrixTesting, stateProbTesting, observations);
		
		/*
		//Show tested results
		printf("tested \n");
		print_matrix(transitionMatrixTesting,hiddenStates,hiddenStates);
		print_matrix(emissionMatrixTesting, hiddenStates,differentObservables);
		print_vector(stateProbTesting, hiddenStates);
		*/


		if (similar(transitionMatrixTesting,transitionMatrix,hiddenStates,hiddenStates) && similar(emissionMatrixTesting,emissionMatrix,differentObservables,hiddenStates)){
			runs[run]=cycles;
		}else{	
			
		  	free(groundTransitionMatrix);
			free(groundEmissionMatrix);
			free(observations);
			free(transitionMatrix);
			free(emissionMatrix);
			free(stateProb);
   			free(ct);
			free(gamma_T);
			free(gamma_sum);
			free(a_new);
			free(b_new);
  			free(transitionMatrixSafe);
			free(emissionMatrixSafe);
   			free(stateProbSafe);
			free(transitionMatrixTesting);
			free(emissionMatrixTesting);
			free(stateProbTesting);
           		free(beta);
            		free(beta_new);
            		free(alpha);
            		free(ab);
			printf("Something went wrong! \n");
			return -1;//error Jan
		}	

	}

	qsort (runs, maxRuns, sizeof (double), compare_doubles);
  	double medianTime = runs[maxRuns/2];
	printf("Median Time: \t %lf cycles \n", medianTime); 

	//write_result(transitionMatrix, emissionMatrix, observations, stateProb, steps, hiddenStates, differentObservables, T);
        
    	free(groundTransitionMatrix);
	free(groundEmissionMatrix);
	free(observations);
	free(transitionMatrix);
	free(emissionMatrix);
	free(stateProb);
   	free(ct);
	free(gamma_T);
	free(gamma_sum);
	free(a_new);
	free(b_new);
  	free(transitionMatrixSafe);
	free(emissionMatrixSafe);
   	free(stateProbSafe);
	free(transitionMatrixTesting);
	free(emissionMatrixTesting);
	free(stateProbTesting);
    	free(beta);
	free(beta_new);
	free(alpha);
	free(ab);
	return 0; 
} 
