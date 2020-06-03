#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <float.h> //for DOUBL_MAX
#include "tsc_x86.h"

#include "io.h"
#include "tested.h"
#include "util.h"

#include "mkl.h"


double EPSILON = 1e-4;
#define DELTA 1e-2
#define BUFSIZE 1<<26   // ~60 MB



void initial_step(double* const a, double* const b, double* const p, const int* const y, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int N, const int K, const int T){
	
	double* beta = (double*) malloc(N  * sizeof(double));
	double* beta_new = (double*) malloc(N * sizeof(double));
	double* alpha = (double*) malloc(N * T * sizeof(double));

	//FORWARD

	double ct0 = 0.0;
	//compute alpha(0) and scaling factor for t = 0
    
	int y0 = y[0];
	for(int s = 0; s < N; s++){
		double alphas = p[s] * b[y0*N + s];
		ct0 += alphas;
		alpha[s] = alphas;
	}
	
	ct0 = 1.0 / ct0;
    
    	cblas_dscal(N,ct0,alpha,1);
	ct[0] = ct0;

	for(int t = 1; t < T-1; t++){
		double ctt = 0.0;	
		const int yt = y[t];	
		for(int s = 0; s<N; s++){// s=new_state
			double alphatNs = cblas_ddot(N,alpha+(t-1)*N,1,a+s,N);
            		//double alphatNs=0;
			//for(int j = 0; j < N; j++){//j=old_states
			//	alphatNs += alpha[(t-1)*N + j] * a[j*N + s];
			//}
			alphatNs *= b[yt*N + s];
			ctt += alphatNs;
			alpha[t*N + s] = alphatNs;
		}
		//scaling factor for t 
		ctt = 1.0 / ctt;
		
		//scale alpha(t)
        	cblas_dscal(N,ctt,alpha+t*N,1);
		//for(int s = 0; s<N; s++){// s=new_state
		//	alpha[t*N+s] *= ctt;
		//}
		ct[t] = ctt;
	}
		
	
	double ctt = 0.0;	
	const int yt = y[T-1];	
	for(int s = 0; s<N; s++){// s=new_state
		double alphatNs = 0;
		//alpha[s*T + t] = 0;
       	alphatNs=cblas_ddot(N,alpha+(T-2)*N,1,a+s,N);		
        	//for(int j = 0; j < N; j++){//j=old_states
			//	alphatNs += alpha[(T-2)*N + j] * a[j*N + s];	
		//}

		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[(T-1)*N + s] = alphatNs;
	}
	//scaling factor for T-1
	ctt = 1.0 / ctt;
		
	//scale alpha(t)
	//for(int s = 0; s<N; s++){// s=new_state
	//	alpha[(T-1)*N+s] *= ctt;
	//	//XXX Last iteration explicit because of this line
	//	gamma_T[s] = alpha[(T-1)*N + s] /* *ct[T-1]*/;
	//}
    	cblas_dscal(N,ctt,alpha+(T-1)*N,1);
    	cblas_dcopy(N,alpha+(T-1)*N,1,gamma_T,1);
    	ct[T-1] = ctt;

	//FUSED BACKWARD and UPDATE STEP

	for(int s = 0; s < N; s++){
		beta[s] = /* 1* */ct[T-1];
		gamma_sum[s] = 0.0;
		for(int j = 0; j < N; j++){
			a_new[s*N + j] =0.0;
		}
		for(int v = 0;  v < K; v++){
			b_new[v*N + s] = 0.0;
		}
	}
    
	//compute sum of xi and gamma from t= 0...T-2
	for(int t = T-1; t > 0; t--){
		for(int s = 0; s < N ; s++){
			p[s] = 0.0;
			beta_new[s]=0.0;
			for(int j = 0; j < N; j++){
				double temp = a[s*N +j] * beta[j] * b[y[t]*N + j];
				
				double xi_sjt = alpha[(t-1)*N + s] * temp;
				a_new[s*N+j] +=xi_sjt;
				beta_new[s] += temp;
				
			}
			p[s] = alpha[(t-1)*N+s]*beta_new[s]/* *ct[t-1]*/;
			beta_new[s] *= ct[t-1];
			gamma_sum[s]+= p[s] /* /ct[t-1] */ ;

			for(int v = 0; v < K; v++){
				int indicator = (int)(y[t-1] == v);
				b_new[v*N + s] += (double)(indicator)*p[s];
			}
		}
		double * temp = beta_new;
		beta_new = beta;
		beta = temp;	
	}

	free(beta);
	free(beta_new);
	free(alpha);
	return;

}
myInt64 bw(double* const a, double* const b, double* const p, const int* const y, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int N, const int K, const int T, double* beta, double* beta_new ,double* alpha, volatile unsigned char* buf, int maxSteps){
     double logLikelihood=-DBL_MAX;
     double disparance;
     int steps=1;
       	_flush_cache(buf,BUFSIZE); // ensure the cache is cold
		myInt64 start = start_tsc();

	double ct0 = 0.0;
	//compute alpha(0) and scaling factor for t = 0
    
	int y0 = y[0];
	for(int s = 0; s < N; s++){
		double alphas = p[s] * b[y0*N + s];
		ct0 += alphas;
		alpha[s] = alphas;
	}
	
	ct0 = 1.0 / ct0;
    
    cblas_dscal(N,ct0,alpha,1);
	ct[0] = ct0;

	for(int t = 1; t < T-1; t++){
		double ctt = 0.0;	
		const int yt = y[t];	
		for(int s = 0; s<N; s++){// s=new_state
			double alphatNs = cblas_ddot(N,alpha+(t-1)*N,1,a+s,N);
			alphatNs *= b[yt*N + s];
			ctt += alphatNs;
			alpha[t*N + s] = alphatNs;
		}
		//scaling factor for t 
		ctt = 1.0 / ctt;
		
		//scale alpha(t)
        cblas_dscal(N,ctt,alpha+t*N,1);
		ct[t] = ctt;
	}
		
	
	double ctt = 0.0;	
	const int yt = y[T-1];	
	for(int s = 0; s<N; s++){// s=new_state
		double alphatNs = 0;
       	alphatNs=cblas_ddot(N,alpha+(T-2)*N,1,a+s,N);
		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[(T-1)*N + s] = alphatNs;
	}
	//scaling factor for T-1
	    ctt = 1.0 / ctt;
    	cblas_dscal(N,ctt,alpha+(T-1)*N,1);
    	cblas_dcopy(N,alpha+(T-1)*N,1,gamma_T,1);
    	ct[T-1] = ctt;

	//FUSED BACKWARD and UPDATE STEP

	for(int s = 0; s < N; s++){
		beta[s] = /* 1* */ct[T-1];
		gamma_sum[s] = 0.0;
		for(int j = 0; j < N; j++){
			a_new[s*N + j] =0.0;
		}
		for(int v = 0;  v < K; v++){
			b_new[v*N + s] = 0.0;
		}
	}
    
	//compute sum of xi and gamma from t= 0...T-2
	for(int t = T-1; t > 0; t--){
		for(int s = 0; s < N ; s++){
			p[s] = 0.0;
			beta_new[s]=0.0;
			for(int j = 0; j < N; j++){
				double temp = a[s*N +j] * beta[j] * b[y[t]*N + j];
				double xi_sjt = alpha[(t-1)*N + s] * temp;
				a_new[s*N+j] +=xi_sjt;
				beta_new[s] += temp;
				
			}
			p[s] = alpha[(t-1)*N+s]*beta_new[s]/* *ct[t-1]*/;
			beta_new[s] *= ct[t-1];
			gamma_sum[s]+= p[s] /* /ct[t-1] */ ;

			for(int v = 0; v < K; v++){
				int indicator = (int)(y[t-1] == v);
				b_new[v*N + s] += (double)(indicator)*p[s];
			}
		}
		double * temp = beta_new;
		beta_new = beta;
		beta = temp;	
	}


		do{
    	cblas_daxpy(N,1,gamma_T,1,b_new+y[T-1]*N,1);
    	cblas_daxpy(N,1,gamma_sum,1,gamma_T,1);
	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] / gamma_T[s];
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
    cblas_dscal(N,ctt,alpha,1);
	
    	ct[0] = ctt;
	//Compute alpha(1) and scale transitionMatrix
	ctt = 0.0;	
	int yt = y[1];	
	for(int s = 0; s<N; s++){// s=new_state
		double alphatNs = 0;
		//alpha[s*T + t] = 0;
		for(int j = 0; j < N; j++){//j=old_states
			double ajNs =  a_new[j*N + s] / gamma_sum[j];
			a[j*N + s] = ajNs;
			alphatNs += alpha[0*N + j] * ajNs;
		}
		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[1*N + s] = alphatNs;
	}
	//scaling factor for t 
	ctt = 1.0 / ctt;

    cblas_dscal(N,ctt,alpha+N,1);
	ct[1] = ctt;

	for(int t = 2; t < T-1; t++){
		ctt = 0.0;	
		yt = y[t];	
		for(int s = 0; s<N; s++){// s=new_state
            double alphatNs=cblas_ddot(N,alpha+(t-1)*N,1,a+s,N);
			alphatNs *= b[yt*N + s];
			ctt += alphatNs;
			alpha[t*N + s] = alphatNs;
		}
		//scaling factor for t 
		ctt = 1.0 / ctt;
		cblas_dscal(N,ctt,alpha+t*N,1);
		ct[t] = ctt;
	}
		
	//compute alpha(T-1)
	ctt = 0.0;	
	yt = y[T-1];	
	for(int s = 0; s<N; s++){// s=new_state
        	double alphatNs=cblas_ddot(N,alpha+(T-2)*N,1,a+s,N);
		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[(T-1)*N + s] = alphatNs;
	}
	//scaling factor for T-1
	ctt = 1.0 / ctt;

    cblas_dscal(N,ctt,alpha+(T-1)*N,1);
    cblas_dcopy(N,alpha+(T-1)*N,1,gamma_T,1);
    ct[T-1] = ctt;
    	
	//FUSED BACKWARD and UPDATE STEP

	for(int s = 0; s < N; s++){
		beta[s] = /* 1* */ct[T-1];
		gamma_sum[s] = 0.0;
		for(int j = 0; j < N; j++){
			a_new[s*N + j] =0.0;
		}
		for(int v = 0;  v < K; v++){
			b_new[v*N + s] = 0.0;
		}
	}
	//compute sum of xi and gamma from t= 0...T-2
	for(int t = T-1; t > 0; t--){
		for(int s = 0; s < N ; s++){
			p[s] = 0.0;
			beta_new[s]=0.0;
			for(int j = 0; j < N; j++){

				double temp = a[s*N +j] * beta[j] * b[y[t]*N + j];
				double xi_sjt = alpha[(t-1)*N + s] * temp;
				a_new[s*N+j] +=xi_sjt;
				beta_new[s] += temp;
				
			}
			p[s] = alpha[(t-1)*N+s]*beta_new[s]/* *ct[t-1]*/;
			beta_new[s] *= ct[t-1];

			//if you use real gamma you have to divide with ct[t-1]
			gamma_sum[s]+= p[s] /* /ct[t-1] */ ;
            b_new[y[t-1]*N+s]+=p[s];
        }
		double * temp = beta_new;
		beta_new = beta;
		beta = temp;	
	}
        steps+=1;
            		
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


		//compute new transition matrix
	for(int s = 0; s < N; s++){
        	double denom=1/gamma_sum[s];
        	cblas_dscal(N,denom,a_new+s*N,1);
	}
	cblas_dcopy(N*N,a_new,1,a,1);

    cblas_daxpy(N,1,gamma_T,1,gamma_sum,1);
    cblas_daxpy(N,1,gamma_T,1,b_new+y[T-1]*N,1);
	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] / gamma_sum[s];
		}
	}

		myInt64 cycles = stop_tsc(start);
        return cycles/steps;

}

void baum_welch(double* const a, double* const b, double* const p, const int* const y, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int N, const int K, const int T){

	double* beta = (double*) malloc(N  * sizeof(double));
	double* beta_new = (double*) malloc(N * sizeof(double));
	double* alpha = (double*) malloc(N * T * sizeof(double));


	//add remaining parts of the sum of gamma 
	/*
	for(int s = 0; s < N; s++){
		//if you use real gamma you have to divide by ct[t-1]
		double gamma_Ts = gamma_T[s];
		gamma_T[s] += gamma_sum[s];// /ct[T-1] 
    		b_new[y[T-1]*N+s]+=gamma_Ts;
	}
	*/
    	cblas_daxpy(N,1,gamma_T,1,b_new+y[T-1]*N,1);
    	cblas_daxpy(N,1,gamma_sum,1,gamma_T,1);
	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] / gamma_T[s];
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
    	cblas_dscal(N,ctt,alpha,1);
	//for(int s = 0; s < N; s++){
		//alpha[s] *= ctt;
		//printf("%lf %lf %lf \n", alpha[s*T], p[s], b[s*K+y[0]]);
	//}
	//print_matrix(alpha,N,T);
	
    	ct[0] = ctt;

	//a[i*N+j] = a_new[i*N+j]/gamma_sum_1[i];
	//Compute alpha(1) and scale transitionMatrix
	ctt = 0.0;	
	int yt = y[1];	
	for(int s = 0; s<N; s++){// s=new_state
		double alphatNs = 0;
		//alpha[s*T + t] = 0;
		for(int j = 0; j < N; j++){//j=old_states
			double ajNs =  a_new[j*N + s] / gamma_sum[j];
			a[j*N + s] = ajNs;
			alphatNs += alpha[0*N + j] * ajNs;
		}
		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[1*N + s] = alphatNs;
	}
	//scaling factor for t 
	ctt = 1.0 / ctt;
	
	//scale alpha(t)
	//for(int s = 0; s<N; s++){// s=new_state
	//	alpha[1*N+s] *= ctt;
	//}
    	cblas_dscal(N,ctt,alpha+N,1);
	ct[1] = ctt;

	for(int t = 2; t < T-1; t++){
		ctt = 0.0;	
		yt = y[t];	
		for(int s = 0; s<N; s++){// s=new_state
			//double alphatNs = 0;
			//for(int j = 0; j < N; j++){//j=old_states
			//	alphatNs += alpha[(t-1)*N + j] * a[j*N + s];
			//}
            		double alphatNs=cblas_ddot(N,alpha+(t-1)*N,1,a+s,N);
			alphatNs *= b[yt*N + s];
			ctt += alphatNs;
			alpha[t*N + s] = alphatNs;
		}
		//scaling factor for t 
		ctt = 1.0 / ctt;
		cblas_dscal(N,ctt,alpha+t*N,1);
		//scale alpha(t)
		//for(int s = 0; s<N; s++){// s=new_state
		//	alpha[t*N+s] *= ctt;
		//}
		ct[t] = ctt;
	}
		
	//compute alpha(T-1)
	ctt = 0.0;	
	yt = y[T-1];	
	for(int s = 0; s<N; s++){// s=new_state
		//double alphatNs = 0;
		//for(int j = 0; j < N; j++){//j=old_states
		//	alphatNs += alpha[(T-2)*N + j] * a[j*N + s];		
		//}
        	double alphatNs=cblas_ddot(N,alpha+(T-2)*N,1,a+s,N);
		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[(T-1)*N + s] = alphatNs;
	}
	//scaling factor for T-1
	ctt = 1.0 / ctt;
		
//	//scale alpha(t)
//	for(int s = 0; s<N; s++){// s=new_state
//		alpha[(T-1)*N+s] *= ctt;
//		gamma_T[s] = alpha[(T-1)*N + s] /* *ct[T-1]*/;
//	}
//	ct[T-1] = ctt;

    	cblas_dscal(N,ctt,alpha+(T-1)*N,1);
    	cblas_dcopy(N,alpha+(T-1)*N,1,gamma_T,1);
    	ct[T-1] = ctt;
    	
	//FUSED BACKWARD and UPDATE STEP

	for(int s = 0; s < N; s++){
		beta[s] = /* 1* */ct[T-1];
		gamma_sum[s] = 0.0;
		for(int j = 0; j < N; j++){
			a_new[s*N + j] =0.0;
		}
		for(int v = 0;  v < K; v++){
			b_new[v*N + s] = 0.0;
		}
	}
	//compute sum of xi and gamma from t= 0...T-2
	for(int t = T-1; t > 0; t--){
		for(int s = 0; s < N ; s++){
			
			p[s] = 0.0;
			beta_new[s]=0.0;
			for(int j = 0; j < N; j++){

				double temp = a[s*N +j] * beta[j] * b[y[t]*N + j];
				double xi_sjt = alpha[(t-1)*N + s] * temp;
				a_new[s*N+j] +=xi_sjt;
				beta_new[s] += temp;
				
			}
			p[s] = alpha[(t-1)*N+s]*beta_new[s]/* *ct[t-1]*/;
			beta_new[s] *= ct[t-1];

			//if you use real gamma you have to divide with ct[t-1]
			gamma_sum[s]+= p[s] /* /ct[t-1] */ ;
            		b_new[y[t-1]*N+s]+=p[s];
//			for(int v = 0; v < K; v++){
//				int indicator = (int)(y[t-1] == v);
//				//if you use real gamma you have to divide by ct[t-1]
//				b_new[v*N + s] += (double)(indicator)*p[s] /* /ct[t-1]*/;
//			}
		}
		
		
		//print_vector(beta,N);
		double * temp = beta_new;
		beta_new = beta;
		beta = temp;	
	}

	//print_matrix(alpha,T,N);

	free(alpha);
	free(beta);
	free(beta_new);
	
	return;
}

void final_scaling(double* const a, double* const b, double* const p, const int* const y, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int N, const int K, const int T){
	//compute new transition matrix
	for(int s = 0; s < N; s++){
        	double denom=1/gamma_sum[s];
        	cblas_dscal(N,denom,a_new+s*N,1);
		//for(int j = 0; j < N; j++){
		//	a[s*N+j] = a_new[s*N+j]/gamma_sum[s];
		//}
	}
	cblas_dcopy(N*N,a_new,1,a,1);

	//add remaining parts of the sum of gamma 
	//for(int s = 0; s < N; s++){
		//if you use real gamma you have to divide by ct[t-1]
		//gamma_sum[s] += gamma_T[s] /* /ct[T-1] */;
        //b_new[y[T-1]*N+s]+=gamma_T[s];
        /*
		for(int v = 0; v < K; v++){
			int indicator = (int)(y[T-1] == v);
			//if you use real gamma you have to divide by ct[t-1]
			b_new[v*N + s] += indicator*gamma_T[s];
		}
        */
	//}
    	cblas_daxpy(N,1,gamma_T,1,gamma_sum,1);
    	cblas_daxpy(N,1,gamma_T,1,b_new+y[T-1]*N,1);
	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] / gamma_sum[s];
		}
	}
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

	double* gamma_T = (double*) malloc( hiddenStates * sizeof(double));
	double* gamma_sum = (double*) malloc( hiddenStates * sizeof(double));
	
	double* a_new = (double*) malloc(hiddenStates * hiddenStates * sizeof(double));
	double* b_new = (double*) malloc(differentObservables*hiddenStates * sizeof(double));
	
	double* ct = (double*) malloc(T*sizeof(double));

    double* beta = (double*) malloc(hiddenStates  * sizeof(double));
	double* beta_new = (double*) malloc(hiddenStates * sizeof(double));
	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	
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
	//heatup(transitionMatrix,stateProb,emissionMatrix,observations,hiddenStates,differentObservables,T);
	
	volatile unsigned char* buf = malloc(BUFSIZE*sizeof(char));
	
	for (int run=0; run<maxRuns; run++){

		//init transition Matrix, emission Matrix and initial state distribution random
		memcpy(transitionMatrix, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
   		memcpy(emissionMatrix, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
      		memcpy(stateProb, stateProbSafe, hiddenStates * sizeof(double));

		runs[run]=bw(transitionMatrix,emissionMatrix,stateProb,observations, gamma_sum, gamma_T, a_new, b_new, ct,hiddenStates,differentObservables, T, beta, beta_new ,alpha, buf, maxSteps);

	}

	qsort (runs, maxRuns, sizeof (double), compare_doubles);
  	double medianTime = runs[maxRuns/2];
	printf("Median Time: \t %lf cycles \n", medianTime); 
	
	//used for testing
	memcpy(transitionMatrixTesting, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
	memcpy(emissionMatrixTesting, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
	memcpy(stateProbTesting, stateProbSafe, hiddenStates * sizeof(double));

	
	transpose(emissionMatrix,differentObservables,hiddenStates);
	//emissionMatrix is not in state major order
	transpose(emissionMatrixTesting, differentObservables,hiddenStates);
	tested_implementation(hiddenStates, differentObservables, T, transitionMatrixTesting, emissionMatrixTesting, stateProbTesting, observations,EPSILON, DELTA);

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
	free((void*)buf);
    	free(beta);
	free(beta_new);
	free(alpha);
	return 0; 
} 
