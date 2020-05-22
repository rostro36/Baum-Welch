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

//generate a random number [0,1] and return the index...
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
void makeMatrix(const int dim1,const int dim2, double* const matrix){

	for (int row=0;row<dim1;row++){
		//make probabilites for one row
		makeProbabilities(matrix + row*dim2,dim2);
	}
}


void initial_step(double* const a, double* const b, double* const p, const int* const y, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int N, const int K, const int T){

	double* beta = (double*) malloc(T  * sizeof(double));
	double* beta_new = (double*) malloc(T * sizeof(double));
	double* alpha = (double*) malloc(N * T * sizeof(double));
	double* ab = (double*) malloc(N * N * (T-1) * sizeof(double));

	//FORWARD

	for(int row = 0 ; row < N; row++){
		for(int col =row+1; col < N; col++){
			double temp = a[col*N+row];
			a[col * N + row]  = a[row * N + col];
			a[row*N + col] = temp;
		}
	}
	

	double ct0 = 0.0;
	//compute alpha(0) and scaling factor for t = 0
	int y0 = y[0];
	for(int s = 0; s < N; s++){
		double alphas = p[s] * b[y0*N + s];
		ct0 += alphas;
		alpha[s] = alphas;
	}
	
	ct0 = 1.0 / ct0;

	//scale alpha(0)
	for(int s = 0; s < N; s++){
		alpha[s] *= ct0;
	}
	ct[0] = ct0;

	for(int t = 1; t < T-1; t++){
		double ctt = 0.0;	
		const int yt = y[t];	
		for(int s = 0; s<N; s++){// s=new_state
			double alphatNs = 0;
			for(int j = 0; j < N; j++){//j=old_states
				alphatNs += alpha[(t-1)*N + j] * a[s*N + j];
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
	double ctt = 0.0;	
	int yt = y[T-1];	
	for(int s = 0; s<N; s++){// s=new_state
		double alphatNs = 0;
		//alpha[s*T + t] = 0;
		for(int j = 0; j < N; j++){//j=old_states
			alphatNs += alpha[(T-2)*N + j] * a[s*N + j];	
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
		for(int j = 0; j < N; j++){
			a_new[s*N + j] =0.0;
		}
	}

	for(int v = 0;  v < K; v++){
		for(int s = 0; s < N; s++){
			b_new[v*N + s] = 0.0;
		}
	}
	for(int row = 0 ; row < N; row++){
		for(int col =row+1; col < N; col++){
			double temp = a[col*N+row];
			a[col * N + row]  = a[row * N + col];
			a[row*N + col] = temp;
		}
	}
	
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			for(int j = 0; j < N; j++){
				ab[(v*N + s) * N + j] = a[s*N + j] * b[v*N +j];
			}
		}
	}

    yt = y[T-1];
	//compute sum of xi and gamma from t= 0...T-2
	for(int t = T-1; t > 0; t--){
		const int yt1 = y[t-1];
		const double ctt = ct[t-1];
		for(int s = 0; s < N ; s++){
			double beta_news = 0.0;
			double alphat1Ns = alpha[(t-1)*N + s];
			for(int j = 0; j < N; j++){
				double temp =ab[(yt*N + s)*N + j] * beta[j];
				
				double xi_sjt = alphat1Ns * temp;
				a_new[s*N+j] +=xi_sjt;
				beta_news += temp;
				
			}
			double ps =alphat1Ns*beta_news/* *ct[t-1]*/;  
			p[s] = ps;
			beta_new[s] = beta_news*ctt;

			//if you use real gamma you have to divide with ct[t-1]
			gamma_sum[s]+= ps /* /ct[t-1] */ ;
            b_new[yt1*N+s]+=ps;
		}
		

		double * temp = beta_new;
		beta_new = beta;
		beta = temp;
		yt=yt1;
	}

	free(beta);
	free(beta_new);
	free(alpha);
	free(ab);
	return;

}
myInt64 bw(double* const transitionMatrix, double* const emissionMatrix, double* const stateProb, const int* const observations, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int hiddenStates, const int differentObservables, const int T,double* beta, double* beta_new, double* alpha, double* ab){
        double logLikelihood=-DBL_MAX;
        double disparance;
		//only needed for testing with R
		//write_init(transitionMatrix, emissionMatrix, observations, stateProb, hiddenStates, differentObservables, T);
        
        int steps=1;
	myInt64 start = start_tsc();
//FORWARD

	for(int row = 0 ; row < hiddenStates; row++){
		for(int col =row+1; col < hiddenStates; col++){
			double temp = transitionMatrix[col*hiddenStates+row];
			transitionMatrix[col * hiddenStates+ row]  = transitionMatrix[row * hiddenStates + col];
			transitionMatrix[row*hiddenStates + col] = temp;
		}
	}
	

	double ct0 = 0.0;
	//compute alpha(0) and scaling factor for t = 0
	int y0 = observations[0];
	for(int s = 0; s < hiddenStates; s++){
		double alphas = stateProb[s] * emissionMatrix[y0*hiddenStates + s];
		ct0 += alphas;
		alpha[s] = alphas;
	}
	
	ct0 = 1.0 / ct0;

	//scale alpha(0)
	for(int s = 0; s < hiddenStates; s++){
		alpha[s] *= ct0;
	}
	ct[0] = ct0;

	for(int t = 1; t < T-1; t++){
		double ctt = 0.0;	
		const int yt = observations[t];	
		for(int s = 0; s<hiddenStates; s++){// s=new_state
			double alphatNs = 0;
			for(int j = 0; j < hiddenStates; j++){//j=old_states
				alphatNs += alpha[(t-1)*hiddenStates + j] * transitionMatrix[s*hiddenStates + j];
			}
			alphatNs *= emissionMatrix[yt*hiddenStates + s];
			ctt += alphatNs;
			alpha[t*hiddenStates + s] = alphatNs;
		}
		//scaling factor for t 
		ctt = 1.0 / ctt;
		
		//scale alpha(t)
		for(int s = 0; s<hiddenStates; s++){// s=new_state
			alpha[t*hiddenStates+s] *= ctt;
		}
		ct[t] = ctt;
	}
	double ctt = 0.0;	
	int yt = observations[T-1];	
	for(int s = 0; s<hiddenStates; s++){// s=new_state
		double alphatNs = 0;
		for(int j = 0; j < hiddenStates; j++){//j=old_states
			alphatNs += alpha[(T-2)*hiddenStates + j] * transitionMatrix[s*hiddenStates + j];	
		}

		alphatNs *= emissionMatrix[yt*hiddenStates + s];
		ctt += alphatNs;
		alpha[(T-1)*hiddenStates + s] = alphatNs;
	}
	//scaling factor for T-1
	ctt = 1.0 / ctt;
		
	//scale alpha(t)
	for(int s = 0; s<hiddenStates; s++){// s=new_state
		double alphaT1Ns = alpha[(T-1) * hiddenStates + s]*ctt;
		alpha[(T-1)*hiddenStates+s] = alphaT1Ns;
		gamma_T[s] = alphaT1Ns /* *ct[T-1]*/;
	}
	ct[T-1] = ctt;


	//FUSED BACKWARD and UPDATE STEP

	for(int s = 0; s < hiddenStates; s++){
		beta[s] = /* 1* */ctt;
		gamma_sum[s] = 0.0;
		for(int j = 0; j < hiddenStates; j++){
			a_new[s*hiddenStates + j] =0.0;
		}
	}

	for(int v = 0;  v < differentObservables; v++){
		for(int s = 0; s < hiddenStates; s++){
			b_new[v*hiddenStates + s] = 0.0;
		}
	}
	for(int row = 0 ; row < hiddenStates; row++){
		for(int col =row+1; col < hiddenStates; col++){
			double temp = transitionMatrix[col*hiddenStates+row];
			transitionMatrix[col * hiddenStates + row]  = transitionMatrix[row * hiddenStates + col];
			transitionMatrix[row*hiddenStates + col] = temp;
		}
	}
	
	for(int v = 0; v < differentObservables; v++){
		for(int s = 0; s < hiddenStates; s++){
			for(int j = 0; j < hiddenStates; j++){
				ab[(v*hiddenStates + s) * hiddenStates + j] = transitionMatrix[s*hiddenStates + j] * emissionMatrix[v*hiddenStates +j];
			}
		}
	}

    	yt = observations[T-1];
	//compute sum of xi and gamma from t= 0...T-2
	for(int t = T-1; t > 0; t--){
		const int yt1 = observations[t-1];
		const double ctt = ct[t-1];
		for(int s = 0; s < hiddenStates; s++){
			double beta_news = 0.0;
			double alphat1Ns = alpha[(t-1)*hiddenStates + s];
			for(int j = 0; j < hiddenStates; j++){
				double temp =ab[(yt*hiddenStates + s)*hiddenStates + j] * beta[j];
				
				double xi_sjt = alphat1Ns * temp;
				a_new[s*hiddenStates+j] +=xi_sjt;
				beta_news += temp;
			}
			double ps =alphat1Ns*beta_news/* *ct[t-1]*/;  
			stateProb[s] = ps;
			beta_new[s] = beta_news*ctt;

			//if you use real gamma you have to divide with ct[t-1]
			gamma_sum[s]+= ps /* /ct[t-1] */ ;
            b_new[yt1*hiddenStates+s]+=ps;
		}
		
		double * temp = beta_new;
		beta_new = beta;
		beta = temp;
		yt=yt1;
	}

		do{
//Baum-Welch
	        int yt = observations[T-1];
	        //add remaining parts of the sum of gamma 
	        for(int s = 0; s < hiddenStates; s++){
		        double gamma_Ts = gamma_T[s];
		        //if you use real gamma you have to divide by ct[t-1]
		        double gamma_sums = gamma_sum[s];
		        double gamma_tot = gamma_Ts + gamma_sums /* /ct[T-1] */;
		        gamma_T[s] = 1./gamma_tot;
		        gamma_sum[s] = 1./gamma_sums;
                b_new[yt*hiddenStates+s]+=gamma_Ts;
	        }

	        //compute new emission matrix
	        for(int v = 0; v < differentObservables; v++){
		        for(int s = 0; s < hiddenStates; s++){
			        emissionMatrix[v*hiddenStates + s] = b_new[v*hiddenStates + s] * gamma_T[s];
			        b_new[v*hiddenStates + s] = 0.0;
		        }
	        }

	        //FORWARD
	        

	        //Transpose a_new. Note that it is not necessary to transpose matrix a.

	        const int block_size = 4;

	        for(int by = 0; by < hiddenStates; by+=block_size){
		        const int end = by + block_size;
		        for(int i = by; i < end-1; i++){
			        for(int j = i+1; j < end; j++){
					        double temp = a_new[i*hiddenStates+j];
					        a_new[i * hiddenStates + j]  = a_new[j * hiddenStates + i];
					        a_new[j*hiddenStates + i] = temp;			
			        }
		        }
		        for(int bx = end; bx < hiddenStates; bx+= block_size){
			        const int end_x = bx + block_size;
			        for(int i = by; i < end; i++){
				        for(int j = bx; j < end_x; j++){
					        double temp = a_new[j*hiddenStates+i];
					        a_new[j * hiddenStates + i]  = a_new[i * hiddenStates + j];
					        a_new[i*hiddenStates + j] = temp;
				        }
			        }
		        }	
	        }

	        double ctt = 0.0;
	        //compute alpha(0) and scaling factor for t = 0
	        int y0 = observations[0];
	        for(int s = 0; s < hiddenStates; s++){
		        double alphas = stateProb[s] * emissionMatrix[y0*hiddenStates + s];
		        ctt += alphas;
		        alpha[s] = alphas;
	        }
	        
	        ctt = 1.0 / ctt;

	        //scale alpha(0)
	        for(int s = 0; s < hiddenStates; s++){
		        alpha[s] *= ctt;
	        }
	        //print_matrix(alpha,N,T);
	        ct[0] = ctt;

	        //Compute alpha(1) and scale transitionMatrix
	        ctt = 0.0;	
	        yt = observations[1];	
	        for(int s = 0; s<hiddenStates-1; s++){// s=new_state
		        double alphatNs = 0;
		        for(int j = 0; j < hiddenStates; j++){//j=old_states
			        double asNj =  a_new[s*hiddenStates + j] * gamma_sum[j];
			        a_new[s*hiddenStates+j] = 0.0;
			        transitionMatrix[s*hiddenStates + j] = asNj;
			        alphatNs += alpha[0*hiddenStates + j] * asNj;
		        }
		        alphatNs *= emissionMatrix[yt*hiddenStates + s];
		        ctt += alphatNs;
		        alpha[1*hiddenStates + s] = alphatNs;
	        }
	        
	        // One iteration seperate to set gamma_sum to zero
	        double alphatNs = 0;
	        for(int j = 0; j < hiddenStates; j++){//j=old_states
		        double gamma_sumj = gamma_sum[j];
		        gamma_sum[j] =0.0;
		        double asNj =  a_new[(hiddenStates-1)*hiddenStates + j] * gamma_sumj;
		        transitionMatrix[(hiddenStates-1)*hiddenStates + j] = asNj;
		        alphatNs += alpha[0*hiddenStates + j] * asNj;
		        a_new[(hiddenStates-1)*hiddenStates+j] = 0.0;
	        }
	        alphatNs *= emissionMatrix[yt*hiddenStates + (hiddenStates-1)];
	        ctt += alphatNs;
	        alpha[1*hiddenStates + (hiddenStates-1)] = alphatNs;
	        
	        //scaling factor for t 
	        ctt = 1.0 / ctt;
	        
	        //scale alpha(t)
	        for(int s = 0; s<hiddenStates; s++){// s=new_state
		        alpha[1*hiddenStates+s] *= ctt;
	        }
	        ct[1] = ctt;

	        for(int t = 2; t < T-1; t++){
		        ctt = 0.0;	
		        yt = observations[t];	
		        for(int s = 0; s<hiddenStates; s++){// s=new_state
			        double alphatNs = 0;
			        for(int j = 0; j < hiddenStates; j++){//j=old_states
				        alphatNs += alpha[(t-1)*hiddenStates + j] *transitionMatrix[s*hiddenStates + j];
			        }
			        alphatNs *= emissionMatrix[yt*hiddenStates + s];
			        ctt += alphatNs;
			        alpha[t*hiddenStates + s] = alphatNs;
		        }
		        //scaling factor for t 
		        ctt = 1.0 / ctt;
		        
		        //scale alpha(t)
		        for(int s = 0; s<hiddenStates; s++){// s=new_state
			        alpha[t*hiddenStates+s] *= ctt;
		        }
		        ct[t] = ctt;
	        }
	        //compute alpha(T-1)
	        ctt = 0.0;	
	        yt = observations[T-1];	
	        for(int s = 0; s<hiddenStates; s++){// s=new_state
		        double alphatNs = 0;
		        for(int j = 0; j < hiddenStates; j++){//j=old_states
			        alphatNs += alpha[(T-2)*hiddenStates + j] * transitionMatrix[s*hiddenStates + j];
		        }

		        alphatNs *= emissionMatrix[yt*hiddenStates + s];
		        ctt += alphatNs;
		        alpha[(T-1)*hiddenStates + s] = alphatNs;
	        }
	        //scaling factor for T-1
	        ctt = 1.0 / ctt;
		        
	        //scale alpha(t)
	        for(int s = 0; s<hiddenStates; s++){// s=new_state
		        double alphaT1Ns = alpha[(T-1) * hiddenStates + s]*ctt;
		        alpha[(T-1)*hiddenStates+s] = alphaT1Ns;

		        gamma_T[s] = alphaT1Ns /* *ct[T-1]*/;
	        }
	        ct[T-1] = ctt;

	        //FUSED BACKWARD and UPDATE STEP

	        for(int by = 0; by < hiddenStates; by+=block_size){
		        const int end = by + block_size;
		        for(int i = by; i < end-1; i++){
			        for(int j = i+1; j < end; j++){
					        double temp = transitionMatrix[i*hiddenStates+j];
					        transitionMatrix[i * hiddenStates + j]  = transitionMatrix[j * hiddenStates + i];
					        transitionMatrix[j*hiddenStates + i] = temp;				
			        }
		        }
		        for(int bx = end; bx < hiddenStates; bx+= block_size){
			        const int end_x = block_size + bx;
			        for(int i = by; i < end; i++){
				        for(int j = bx; j < end_x; j++){
					        double temp = transitionMatrix[j*hiddenStates+i];
					        transitionMatrix[j * hiddenStates + i]  = transitionMatrix[i * hiddenStates + j];
					        transitionMatrix[i*hiddenStates + j] = temp;
				        }
			        }
		        }	
	        }
	        
	        for(int v = 0; v < differentObservables; v++){
		        for(int s = 0; s < hiddenStates; s++){
			        for(int j = 0; j < hiddenStates; j++){
				        ab[(v*hiddenStates + s) * hiddenStates + j] = transitionMatrix[s*hiddenStates + j] * emissionMatrix[v*hiddenStates +j];
			        }
		        }
	        }

	        for(int s = 0; s < hiddenStates; s++){
		        beta[s] = /* 1* */ctt;
	        }

	        //compute sum of xi and gamma from t= 0...T-2
            yt = observations[T-1];
	        for(int t = T-1; t > 0; t--){
		        const int yt1 = observations[t-1];
		        ctt = ct[t-1];
		        for(int s = 0; s < hiddenStates ; s++){
			        double beta_news = 0.0;
			        double alphat1Ns = alpha[(t-1)*hiddenStates + s];
			        for(int j = 0; j < hiddenStates; j++){
				        double temp = ab[(yt*hiddenStates + s)*hiddenStates + j] * beta[j];
				        a_new[s*hiddenStates+j] +=alphat1Ns * temp;
				        beta_news += temp;
			        }
			        double ps =alphat1Ns*beta_news/* *ct[t-1]*/;  
			        stateProb[s] = ps;
			        beta_new[s] = beta_news*ctt;

			        //if you use real gamma you have to divide with ct[t-1]
			        gamma_sum[s]+= ps /* /ct[t-1] */ ;
                  	b_new[yt1*hiddenStates+s]+=ps;
		        }
		        double * temp = beta_new;
		        beta_new = beta;
		        beta = temp;
		        yt=yt1;	
	        }
            steps+=1;
//Finishing
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
//Final scale		
	        //compute new transition matrix
	        for(int s = 0; s < hiddenStates; s++){
		        double gamma_sums_inv = 1./gamma_sum[s];
		        for(int j = 0; j < hiddenStates; j++){
			        transitionMatrix[s*hiddenStates+j] = a_new[s*hiddenStates+j]*gamma_sums_inv;
		        }
	        }

	        yt =observations[T-1];
	        //add remaining parts of the sum of gamma 
	        for(int s = 0; s < hiddenStates; s++){	
		        double gamma_Ts = gamma_T[s];
		        //if you use real gamma you have to divide by ct[t-1]
		        double gamma_tot = gamma_Ts + gamma_sum[s] /* /ct[T-1] */;
		        gamma_T[s] = 1./gamma_tot;
                b_new[yt*hiddenStates+s]+=gamma_Ts;
	        }

	        //compute new emission matrix
	        for(int v = 0; v < differentObservables; v++){
		        for(int s = 0; s < hiddenStates; s++){
			        emissionMatrix[v*hiddenStates + s] = b_new[v*hiddenStates + s] * gamma_T[s];
		        }
	        }
		myInt64 cycles = stop_tsc(start);
        cycles = cycles/steps;
        return cycles;
}
void baum_welch(double* const a, double* const b, double* const p, const int* const y, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int N, const int K, const int T){

	double* beta = (double*) malloc(N  * sizeof(double));
	double* beta_new = (double*) malloc(N * sizeof(double));
	double* alpha = (double*) malloc(N * T * sizeof(double));
	double* ab = (double*) malloc(N * N * K * sizeof(double));

	int yt = y[T-1];
	//add remaining parts of the sum of gamma 
	for(int s = 0; s < N; s++){
		double gamma_Ts = gamma_T[s];
		//if you use real gamma you have to divide by ct[t-1]
		double gamma_sums = gamma_sum[s];
		double gamma_tot = gamma_Ts + gamma_sums /* /ct[T-1] */;
		gamma_T[s] = 1./gamma_tot;
		gamma_sum[s] = 1./gamma_sums;
        	b_new[yt*N+s]+=gamma_Ts;
	}

	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] * gamma_T[s];
			b_new[v*N + s] = 0.0;
		}
	}

	//FORWARD
	

	//Transpose a_new. Note that it is not necessary to transpose matrix a.

	const int block_size = 4;

	for(int by = 0; by < N; by+=block_size){
		const int end = by + block_size;
		for(int i = by; i < end-1; i++){
			for(int j = i+1; j < end; j++){
					double temp = a_new[i*N+j];
					a_new[i * N + j]  = a_new[j * N + i];
					a_new[j*N + i] = temp;			
			}
		}
		for(int bx = end; bx < N; bx+= block_size){
			const int end_x = bx + block_size;
			for(int i = by; i < end; i++){
				for(int j = bx; j < end_x; j++){
					double temp = a_new[j*N+i];
					a_new[j * N + i]  = a_new[i * N + j];
					a_new[i*N + j] = temp;
				}
			}
		}	
	}

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
	for(int s = 0; s<N-1; s++){// s=new_state
		double alphatNs = 0;
		for(int j = 0; j < N; j++){//j=old_states
			double asNj =  a_new[s*N + j] * gamma_sum[j];
			a_new[s*N+j] = 0.0;
			a[s*N + j] = asNj;
			alphatNs += alpha[0*N + j] * asNj;
		}
		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[1*N + s] = alphatNs;
	}
	
	// One iteration seperate to set gamma_sum to zero
	double alphatNs = 0;
	for(int j = 0; j < N; j++){//j=old_states
		double gamma_sumj = gamma_sum[j];
		gamma_sum[j] =0.0;
		double asNj =  a_new[(N-1)*N + j] * gamma_sumj;
		a[(N-1)*N + j] = asNj;
		alphatNs += alpha[0*N + j] * asNj;
		a_new[(N-1)*N+j] = 0.0;
	}
	alphatNs *= b[yt*N + (N-1)];
	ctt += alphatNs;
	alpha[1*N + (N-1)] = alphatNs;
	
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
			for(int j = 0; j < N; j++){//j=old_states
				alphatNs += alpha[(t-1)*N + j] *a[s*N + j];
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
		for(int j = 0; j < N; j++){//j=old_states
			alphatNs += alpha[(T-2)*N + j] * a[s*N + j];
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

	for(int by = 0; by < N; by+=block_size){
		const int end = by + block_size;
		for(int i = by; i < end-1; i++){
			for(int j = i+1; j < end; j++){
					double temp = a[i*N+j];
					a[i * N + j]  = a[j * N + i];
					a[j*N + i] = temp;				
			}
		}
		for(int bx = end; bx < N; bx+= block_size){
			const int end_x = block_size + bx;
			for(int i = by; i < end; i++){
				for(int j = bx; j < end_x; j++){
					double temp = a[j*N+i];
					a[j * N + i]  = a[i * N + j];
					a[i*N + j] = temp;
				}
			}
		}	
	}
	
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			for(int j = 0; j < N; j++){
				ab[(v*N + s) * N + j] = a[s*N + j] * b[v*N +j];
			}
		}
	}

	for(int s = 0; s < N; s++){
		beta[s] = /* 1* */ctt;
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
				double temp = ab[(yt*N + s)*N + j] * beta[j];
				a_new[s*N+j] +=alphat1Ns * temp;
				beta_news += temp;
				
			}
			double ps =alphat1Ns*beta_news/* *ct[t-1]*/;  
			p[s] = ps;
			beta_new[s] = beta_news*ctt;

			//if you use real gamma you have to divide with ct[t-1]
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
	free(ab);
	
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
		double gamma_tot = gamma_Ts + gamma_sum[s] /* /ct[T-1] */;
		gamma_T[s] = 1./gamma_tot;
        b_new[yt*N+s]+=gamma_Ts;
	}

	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] * gamma_T[s];
		}
	}
}
//Jan
int finished(const double* const ct, double* const l,const int N,const int T){

	//log likelihood
	double oldLogLikelihood=*l;

	double newLogLikelihood = 0.0;
	//evidence with alpha only:

	for(int time = 0; time < T; time++){
		newLogLikelihood -= log2(ct[time]);
	}
	
	*l=newLogLikelihood;
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
	const int cachegrind_runs = 1;
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

    	double* beta = (double*) malloc(hiddenStates  * sizeof(double));
	double* beta_new = (double*) malloc(hiddenStates * sizeof(double));
	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* ab = (double*) malloc(hiddenStates * hiddenStates * differentObservables * sizeof(double));
	
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
	
	for (int run=0; run<cachegrind_runs; run++){

		//init transition Matrix, emission Matrix and initial state distribution random
		memcpy(transitionMatrix, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
   		memcpy(emissionMatrix, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
        	memcpy(stateProb, stateProbSafe, hiddenStates * sizeof(double));
		
        	myInt64 cycles=bw(transitionMatrix, emissionMatrix, stateProb, observations, gamma_sum, gamma_T,a_new,b_new, ct, hiddenStates, differentObservables, T,beta, beta_new, alpha, ab);


		/*
		//Show results
		printf(" %i \n", steps);
		print_matrix(transitionMatrix,hiddenStates,hiddenStates);
		print_matrix(emissionMatrix, hiddenStates,differentObservables);
		print_vector(stateProb, hiddenStates);
		*/

		/*
		//Show tested results
		printf("tested \n");
		print_matrix(transitionMatrixTesting,hiddenStates,hiddenStates);
		print_matrix(emissionMatrixTesting, hiddenStates,differentObservables);
		print_vector(stateProbTesting, hiddenStates);
		*/


		runs[run]=cycles;
	

	}

	qsort (runs, cachegrind_runs, sizeof (double), compare_doubles);
  	double medianTime = runs[(int)(cachegrind_runs/2)];
	printf("Median Time: \t %lf cycles \n", medianTime); 

	//used for testing
	memcpy(transitionMatrixTesting, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
	memcpy(emissionMatrixTesting, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
	memcpy(stateProbTesting, stateProbSafe, hiddenStates * sizeof(double));

	//write_result(transitionMatrix, emissionMatrix, observations, stateProb, steps, hiddenStates, differentObservables, T);

	transpose(emissionMatrix,differentObservables,hiddenStates);
	//emissionMatrix is not in state major order
	transpose(emissionMatrixTesting, differentObservables,hiddenStates);
        tested_implementation(hiddenStates, differentObservables, T, transitionMatrixTesting, emissionMatrixTesting, stateProbTesting, observations);
		

	if (!similar(transitionMatrixTesting,transitionMatrix,hiddenStates,hiddenStates) && similar(emissionMatrixTesting,emissionMatrix,differentObservables,hiddenStates)){
		printf("Something went wrong !");	
		
	}        
	
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

	return 0; 
} 
