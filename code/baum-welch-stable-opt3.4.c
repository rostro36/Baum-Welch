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
			//printf(" %lf %lf \n ", 	transpose[col * rows + row] , a[row * cols + col]);
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


void initial_step(double* const a, double* const b, double* const p, const int* const y, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int N, const int K, const int T){
	

	//XXX It is not optimal to create three arrays in each iteration.
	//This is only used to demonstarte the use of the pointer swap at the end
	//When inlined the arrays should be generated at the beginning of the main function like the other arrays we use
	//Then this next three lines can be deleted 
	//Rest needs no modification
	double* beta = (double*) malloc(T  * sizeof(double));
	double* beta_new = (double*) malloc(T * sizeof(double));
	double* alpha = (double*) malloc(N * T * sizeof(double));

	//FORWARD

	double ct0 = 0.0;
	//ct[0]=0.0;
	//compute alpha(0) and scaling factor for t = 0
	int y0 = y[0];
	for(int s = 0; s < N; s++){
		double alphas = p[s] * b[y0*N + s];
		ct0 += alphas;
		alpha[s] = alphas;
		//printf("%lf %lf %lf \n", alpha[s], p[s], b[s*K+y[0]]);
	}
	
	ct0 = 1.0 / ct0;

	//scale alpha(0)
	for(int s = 0; s < N; s++){
		alpha[s] *= ct0;
		//printf("%lf %lf %lf \n", alpha[s*T], p[s], b[s*K+y[0]]);
	}
	//print_matrix(alpha,N,T);
	ct[0] = ct0;

	for(int t = 1; t < T-1; t++){
		double ctt = 0.0;	
		const int yt = y[t];	
		for(int s = 0; s<N; s++){// s=new_state
			double alphatNs = 0;
			//alpha[s*T + t] = 0;
			for(int j = 0; j < N; j++){//j=old_states
				alphatNs += alpha[(t-1)*N + j] * a[j*N + s];
				//printf("%lf %lf %lf %lf %i \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t+1]],y[t]);
			}
			alphatNs *= b[yt*N + s];
			//print_matrix(alpha,N,T);
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
	const int yt = y[T-1];	
	for(int s = 0; s<N; s++){// s=new_state
		double alphatNs = 0;
		//alpha[s*T + t] = 0;
		for(int j = 0; j < N; j++){//j=old_states
			alphatNs += alpha[(T-2)*N + j] * a[j*N + s];
			//printf("%lf %lf %lf %lf %i \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t+1]],y[t]);

			
		}

		alphatNs *= b[yt*N + s];	
		//print_matrix(alpha,N,T);
		ctt += alphatNs;
		alpha[(T-1)*N + s] = alphatNs;
	}
	//scaling factor for T-1
	ctt = 1.0 / ctt;
		
	//scale alpha(t)
	for(int s = 0; s<N; s++){// s=new_state
		double alphaT1Ns = alpha[(T-1) * N + s]*ctt;
		alpha[(T-1)*N+s] = alphaT1Ns;
		//XXX Last iteration explicit because of this line
		gamma_T[s] = alphaT1Ns /* *ct[T-1]*/;
	}
	ct[T-1] = ctt;
	//print_matrix(alpha,T,N);
	//print_matrix(ct,1,T);


	//FUSED BACKWARD and UPDATE STEP

	for(int s = 0; s < N; s++){
		beta[s] = /* 1* */ctt;
		//if you use real gamma you have to divide by ct[t-1]
		//the scaling is not necessary
		//without this scaling we have:
		//(T-1)*N mults
		//instead of:
		//(N + (T-1)*N*N )*mult + ((T-1)*N + (T-1)*N*K + N + N*K)*div
		//gamma_T[s] = alpha[(T-1)*N + s] /* *ct[T-1]*/;
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

	//print_matrix(beta,1,N);
	//print_matrix(gamma_T,1,N);

	//compute sum of xi and gamma from t= 0...T-2
	for(int t = T-1; t > 0; t--){
		const int yt = y[t];
		const int yt1 = y[t-1];
		const double ctt = ct[t-1];
		for(int s = 0; s < N ; s++){
			double beta_news = 0.0;
			double alphat1Ns = alpha[(t-1)*N + s];
			//p[s] = 0.0;
			//beta_new[s]=0.0;
			for(int j = 0; j < N; j++){
				double temp = a[s*N +j] * beta[j] * b[yt*N + j];
				
				double xi_sjt = alphat1Ns * temp;
				a_new[s*N+j] +=xi_sjt;
				//XXX NEW COMPUTATION OF BETA DOES NOT NEED THIS
				//to get real gamma you have to scale with ct[t-1] 
				//p[i] += xi_ijt /* *ct[t-1]*/ ;
				//printf("%lf %lf %lf %lf %lf , ", xi_ijt,alpha[(t-1)*N + i] , a[i*N + j] , beta[ j] , b[j*K + y[t]]);

				//XXX NEW COMPUTATION OF BETA DOES NEED THIS
				//Cost equal as computing p[i]
				//printf("%lf , ", beta[ j] );
				beta_news += temp;
				
			}
			//XXX NEW COMPUTATION OF BETA DOES NEED THIS
			//Cost better as beta = gamma / alpha because we replace division by multiplication
			//to get real gamma you have to scale with ct[t-1]
			double ps =alphat1Ns*beta_news/* *ct[t-1]*/;  
			p[s] = ps;
			beta_new[s] = beta_news*ctt;

			//XXX NEW COMPUTATION OF BETA DOES NOT NEED THIS
			//printf("\n\n");
			//beta_new[i] = p[i] * ct[t-1] / alpha[(t-1)*N+i];

			//if you use real gamma you have to divide with ct[t-1]
			gamma_sum[s]+= ps /* /ct[t-1] */ ;
            b_new[yt1*N+s]+=ps;
            /*
			for(int v = 0; v < K; v++){
				int indicator = (int)(yt1 == v);
				//if you use real gamma you have to divide by ct[t-1]
				b_new[v*N + s] += (double)(indicator)*ps;
				
				//printf(" %i %lf \n ", indicator,p[i]);
			}
			*/
			//printf(" %lf %lf %lf \n", p[i],  ct[t-1],alpha[(t-1)*N+i]);
		}
		//printf("T = %li\n",t);
		

		double * temp = beta_new;
		beta_new = beta;
		beta = temp;
		
		
		
		/*
		//This needs additional accesses to p[i] and alpha[(t-1)*N +i]
		for(int i = 0; i < N; i++){
			//if you use real gamma you do not have to scale with ct[t-1]
			beta[i] =p[i] *ct[t-1] / alpha[(t-1)*N + i] ;
		}
		*/	

		//printf("beta \n");
		//print_matrix(beta,1,N);
		//printf("gamma \n");
		//print_matrix(gamma_t,1,N);
		//printf("gamma sum \n");		
		//print_matrix(gamma_sum,1,N);

		
	}

	free(beta);
	free(beta_new);
	free(alpha);
	return;

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
        /*
		for(int v = 0; v < K; v++){
			int indicator = (int)(yt == v);
			//if you use real gamma you have to divide by ct[t-1]
			b_new[v*N + s] += indicator*gamma_Ts;
		}
        */
	}

	const int block_size_x = 4;
	const int block_size_y = 4;

	//print_matrix(b,K,N);
	//printf("\n");
	for(int i = 0; i<K; i+=block_size_y){
		for(int j = 0; j < N; j+= block_size_x){
			for(int v = i; v < i + block_size_y; v++){
				for(int s = j; s < j + block_size_x; s++){
					//printf("hello %lf \n", b[v*N + s]);
					b[v*N + s] = b_new[v*N +s] / gamma_T[s];
					b_new[v*N + s] = 0.0;
				}
			}
		}
	} 

	/*
	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] / gamma_T[s];
			//printf(" %lf %lf %lf \n", b[i*K + v], b_new[i*K + v] , gamma_sum[i]);
			b_new[v*N + s] = 0.0;
		}
	}
	*/

	//print_matrix(b,N,K);

	//FORWARD

	double ctt = 0.0;
	//ct[0]=0.0;
	//compute alpha(0) and scaling factor for t = 0
	int y0 = y[0];
	for(int s = 0; s < N; s++){
		double alphas = p[s] * b[y0*N + s];
		ctt += alphas;
		alpha[s] = alphas;
		//printf("%lf %lf %lf \n", alpha[s], p[s], b[s*K+y[0]]);
	}
	
	ctt = 1.0 / ctt;

	//scale alpha(0)
	for(int s = 0; s < N; s++){
		alpha[s] *= ctt;
		//printf("%lf %lf %lf \n", alpha[s*T], p[s], b[s*K+y[0]]);
	}
	//print_matrix(alpha,N,T);
	ct[0] = ctt;

	//a[i*N+j] = a_new[i*N+j]/gamma_sum_1[i];

	//Compute alpha(1) and scale transitionMatrix
	ctt = 0.0;	
	yt = y[1];	
	for(int s = 0; s<N; s++){// s=new_state
		double alphatNs = 0;
		//alpha[s*T + t] = 0;
		for(int j = 0; j < N; j++){//j=old_states
			double ajNs =  a_new[j*N + s] / gamma_sum[j];
			a[j*N + s] = ajNs;
			alphatNs += alpha[0*N + j] * ajNs;
			//printf("%lf %lf %lf %lf %i \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t+1]],y[t]);
			a_new[j*N+s] = 0.0;
		}
		alphatNs *= b[yt*N + s];
		//print_matrix(alpha,N,T);
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
				//printf("%lf %lf %lf %lf %i \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t+1]],y[t]);
			}
			alphatNs *= b[yt*N + s];
			//print_matrix(alpha,N,T);
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
			//printf("%lf %lf %lf %lf %i \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t+1]],y[t]);

			
		}

		alphatNs *= b[yt*N + s];	
		//print_matrix(alpha,N,T);
		ctt += alphatNs;
		alpha[(T-1)*N + s] = alphatNs;
	}
	//scaling factor for T-1
	ctt = 1.0 / ctt;
		
	//scale alpha(t)
	for(int s = 0; s<N; s++){// s=new_state
		double alphaT1Ns = alpha[(T-1) * N + s]*ctt;
		alpha[(T-1)*N+s] = alphaT1Ns;
		//XXX Last iteration explicit because of this line
		gamma_T[s] = alphaT1Ns /* *ct[T-1]*/;
	}
	ct[T-1] = ctt;
	//print_matrix(alpha,T,N);
	//print_matrix(ct,1,T);

	//FUSED BACKWARD and UPDATE STEP

	for(int s = 0; s < N; s++){
		beta[s] = /* 1* */ctt;
		//if you use real gamma you have to divide by ct[t-1]
		//the scaling is not necessary
		//without this scaling we have:
		//(T-1)*N mults
		//instead of:
		//(N + (T-1)*N*N )*mult + ((T-1)*N + (T-1)*N*K + N + N*K)*div
		//gamma_T[s] = alpha[(T-1)*N + s] /* *ct[T-1]*/;
		gamma_sum[s] = 0.0;
	}
	//print_matrix(beta,1,N);
	//print_matrix(gamma_T,1,N);

	//compute sum of xi and gamma from t= 0...T-2
	for(int t = T-1; t > 0; t--){
		const int yt = y[t];
		const int yt1 = y[t-1];
		ctt = ct[t-1];
		for(int s = 0; s < N ; s++){
			double beta_news = 0.0;
			double alphat1Ns = alpha[(t-1)*N + s];
			//p[s] = 0.0;
			//beta_new[s]=0.0;
			for(int j = 0; j < N; j++){
				double temp = a[s*N +j] * beta[j] * b[yt*N + j];
				
				double xi_sjt = alphat1Ns * temp;
				a_new[s*N+j] +=xi_sjt;
				//XXX NEW COMPUTATION OF BETA DOES NOT NEED THIS
				//to get real gamma you have to scale with ct[t-1] 
				//p[i] += xi_ijt /* *ct[t-1]*/ ;
				//printf("%lf %lf %lf %lf %lf , ", xi_ijt,alpha[(t-1)*N + i] , a[i*N + j] , beta[ j] , b[j*K + y[t]]);

				//XXX NEW COMPUTATION OF BETA DOES NEED THIS
				//Cost equal as computing p[i]
				//printf("%lf , ", beta[ j] );
				beta_news += temp;
				
			}
			//XXX NEW COMPUTATION OF BETA DOES NEED THIS
			//Cost better as beta = gamma / alpha because we replace division by multiplication
			//to get real gamma you have to scale with ct[t-1]
			double ps =alphat1Ns*beta_news/* *ct[t-1]*/;  
			p[s] = ps;
			beta_new[s] = beta_news*ctt;

			//XXX NEW COMPUTATION OF BETA DOES NOT NEED THIS
			//printf("\n\n");
			//beta_new[i] = p[i] * ct[t-1] / alpha[(t-1)*N+i];

			//if you use real gamma you have to divide with ct[t-1]
			gamma_sum[s]+= ps /* /ct[t-1] */ ;
            b_new[yt1*N+s]+=ps;
            /*
			for(int v = 0; v < K; v++){
				int indicator = (int)(yt1 == v);
				//if you use real gamma you have to divide by ct[t-1]
				b_new[v*N + s] += (double)(indicator)*ps;
				
				//printf(" %i %lf \n ", indicator,p[i]);
			}
			*/
			//printf(" %lf %lf %lf \n", p[i],  ct[t-1],alpha[(t-1)*N+i]);
		}
		//printf("T = %li\n",t);
		

		double * temp = beta_new;
		beta_new = beta;
		beta = temp;
		
		
		
		/*
		//This needs additional accesses to p[i] and alpha[(t-1)*N +i]
		for(int i = 0; i < N; i++){
			//if you use real gamma you do not have to scale with ct[t-1]
			beta[i] =p[i] *ct[t-1] / alpha[(t-1)*N + i] ;
		}
		*/	

		//printf("beta \n");
		//print_matrix(beta,1,N);
		//printf("gamma \n");
		//print_matrix(gamma_t,1,N);
		//printf("gamma sum \n");		
		//print_matrix(gamma_sum,1,N);
		
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
			a[s*N+j] = a_new[s*N+j]*gamma_sums_inv;//gamma_sum[s];
			//printf(" %lf %lf %lf \n", a[i*N + j],  a_new[i*N+j],gamma_sum[i]);
		}
	}
	//print_matrix(a,N,N);

	int yt =y[T-1];
	//add remaining parts of the sum of gamma 
	for(int s = 0; s < N; s++){	
		double gamma_Ts = gamma_T[s];
		//if you use real gamma you have to divide by ct[t-1]
		gamma_sum[s] += gamma_Ts /* /ct[T-1] */;
        b_new[yt*N+s]+=gamma_Ts;
        /*
		for(int v = 0; v < K; v++){
			int indicator = (int)(yt == v);
			//if you use real gamma you have to divide by ct[t-1]
			b_new[v*N + s] += indicator*gamma_Ts;
		}
        */
	}

	const int block_size_x = 4;
	const int block_size_y = 4;

	//print_matrix(b,K,N);
	//printf("\n");
	for(int i = 0; i<K; i+=block_size_y){
		for(int j = 0; j < N; j+= block_size_x){
			for(int v = i; v < i + block_size_y; v++){
				for(int s = j; s < j + block_size_x; s++){
					//printf("hello %lf \n", b[v*N + s]);
					b[v*N + s] = b_new[v*N +s] / gamma_T[s];
					b_new[v*N + s] = 0.0;
				}
			}
		}
	} 

	/*
	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] / gamma_sum[s];
			//printf(" %lf %lf %lf \n", b[i*K + v], b_new[i*K + v] , gamma_sum[i]);
		}
	}
	*/
	//print_matrix(b,N,K);

}
//Jan
int finished(const double* const ct, double* const l,const int N,const int T){

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
	
        int steps=0;
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

		initial_step(transitionMatrix, emissionMatrix, stateProb, observations, gamma_sum, gamma_T,a_new,b_new,ct, hiddenStates, differentObservables, T);

		//for(int i = 0; i < 1; i++){
		do{

			baum_welch(transitionMatrix, emissionMatrix, stateProb, observations, gamma_sum, gamma_T,a_new,b_new,ct, hiddenStates, differentObservables, T);

            		steps+=1;
		}while (!finished(ct, &logLikelihood, hiddenStates, T) && steps<maxSteps);
		
		final_scaling(transitionMatrix, emissionMatrix, stateProb, observations, gamma_sum, gamma_T,a_new,b_new,ct, hiddenStates, differentObservables, T);

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
            //DEBUG OFF
			//printf("run %i: \t %llu cycles \n",run, cycles);
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

