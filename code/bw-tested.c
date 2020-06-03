#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <float.h> //for DOUBL_MAX

#include "tested.h"


void tested_set_zero(double* const a, const int rows, const int cols){
	for(int row = 0 ; row < rows; row++){
		for(int col = 0; col < cols; col++){
			a[row * cols + col] = 0.0;
		}
	}
}

void tested_forward(const double* const a, const double* const p, const double* const b, double* const alpha,  const int * const y, double* const ct, const int N, const int K, const int T){

	ct[0]=0.0;
	for(int s = 0; s < N; s++){
		alpha[s*T] = p[s] * b[s*K + y[0]];
		ct[0] += alpha[s*T];
	}
	
	//scaling factor for t = 0
	ct[0] = 1.0 / ct[0];

	//scale alpha(0)
	for(int s = 0; s < N; s++){
		alpha[s*T] *= ct[0];
	}

	for(int t = 1; t < T; t++){
		ct[t]=0.0;
		for(int s = 0; s<N; s++){// s=new_state
			alpha[s*T + t] = 0;
			for(int j = 0; j < N; j++){//j=old_states

				alpha[s*T + t] += alpha[j*T + t-1] * a[j*N + s];
			}

			alpha[s*T + t] *= b[s*K + y[t]];
			ct[t] += alpha[s*T + t];
		}
		//scaling factor for t 
		ct[t] = 1.0 / ct[t];
		
		//scale alpha(t)
		for(int s = 0; s<N; s++){// s=new_state
			alpha[s*T + t] *= ct[t];
		}
		
	}


	return;
}


void tested_backward(const double* const a, const double* const b, double* const beta, const int * const y, const double * const ct, const int N, const int K, const int T ){
	for(int s = 1; s < N+1; s++){
		beta[s*T-1] = /* 1* */ct[T-1];
	}

	for(int t = T-1; t > 0; t--){
		for(int s = 0; s < N; s++){//s=older state
       			beta[s*T + t-1] = 0.;
			for(int j = 0; j < N; j++){//j=newer state
				beta[s*T + t-1] += beta[j*T + t ] * a[s*N + j] * b[j*K + y[t]];
			}
			beta[s*T + t-1] *= ct[t-1];
		}
	}
	return;
}

void tested_update(double* const a, double* const p, double* const b, const double* const alpha, const double* const beta, double* const gamma, double* const xi, const int* const y, const double* const ct,const int N, const int K, const int T){


	double xi_sum, gamma_sum_numerator, gamma_sum_denominator;

	//gamma needs t = 0 ... T and not like xi from 0...T-1
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

	for(int s = 0; s < N; s++){
		// new pi
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



void tested_evidence_testing(const double* const alpha, const double* const beta,const double* const a,const double* const b,const int* const y, const double* const ct, const int N, const int T,int K){
	
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
}


int tested_finished(const double* const alpha,const double* const beta, const double* const ct, double* const l,const int N,const int T,double EPSILON){

	
	//log likelihood
	double oldLogLikelihood=*l;

	double newLogLikelihood = 0.0;
	//evidence with alpha only:

	for(int time = 0; time < T; time++){
		newLogLikelihood -= log2(ct[time]);
	}
	
	*l=newLogLikelihood;

	//printf("log likelihood %.10lf , Epsilon %.10lf result %.10lf \n", newLogLikelihood, EPSILON,newLogLikelihood-oldLogLikelihood);
	return (newLogLikelihood-oldLogLikelihood)<EPSILON;
}



int tested_similar(const double * const a, const double * const b , const int N, const int M, const double DELTA){
	//Frobenius norm
	double sum=0.0;
	double abs=0.0;
	for(int i=0;i<N;i++){
		for(int j=0;j<M;j++){
			abs=a[i*M+j]-b[i*M+j];
			sum+=abs*abs;
		}
	}
	printf("Frobenius norm = %.10lf delta = %.10lf\n", sqrt(sum), DELTA);
	return sqrt(sum)<DELTA; 
}


void tested_implementation(int hiddenStates, int differentObservables, int T, double* transitionMatrix, double* emissionMatrix, double* stateProb, int* observations,const double EPSILON,const double DELTA){
	
	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * (T-1) * sizeof(double)); 
	double* ct = (double*) malloc(T*sizeof(double));
	    
	double logLikelihood=-DBL_MAX;
    
    	int minima=10;
    	int variableSteps=100-cbrt(hiddenStates*differentObservables*T)/3;
 	int maxSteps=minima < variableSteps ? variableSteps : minima;
 	minima=1;    
    	variableSteps=10-log10(hiddenStates*differentObservables*T);

	tested_set_zero(alpha,hiddenStates,T);
	tested_set_zero(beta,hiddenStates,T);
	tested_set_zero(gamma,hiddenStates,T);
	tested_set_zero(xi,hiddenStates*hiddenStates,T-1);
	tested_set_zero(ct,1,T);

	int steps=0;
	do{
		tested_forward(transitionMatrix, stateProb, emissionMatrix, alpha, observations, ct, hiddenStates, differentObservables, T);
		tested_backward(transitionMatrix, emissionMatrix, beta,observations, ct, hiddenStates, differentObservables, T);
		tested_update(transitionMatrix, stateProb, emissionMatrix, alpha, beta, gamma, xi, observations, ct, hiddenStates, differentObservables, T);
		steps+=1;

	}while (!tested_finished(alpha, beta, ct, &logLikelihood, hiddenStates, T, EPSILON) && steps<maxSteps);
	free(alpha);
	free(beta);
	free(gamma);
	free(xi);
	free(ct);
};
