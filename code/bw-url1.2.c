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
	double* ab = (double*) malloc(N * N * (T-1) * sizeof(double));

	//FORWARD

	for(int row = 0 ; row < N; row++){
		for(int col =row+1; col < N; col++){
			double temp = a[col*N+row];
			a[col * N + row]  = a[row * N + col];
			a[row*N + col] = temp;
			//printf(" %lf %lf \n ", 	transpose[col * rows + row] , a[row * cols + col]);
		}
	}
	

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
				alphatNs += alpha[(t-1)*N + j] * a[s*N + j];
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
	int yt = y[T-1];	
	for(int s = 0; s<N; s++){// s=new_state
		double alphatNs = 0;
		//alpha[s*T + t] = 0;
		for(int j = 0; j < N; j++){//j=old_states
			alphatNs += alpha[(T-2)*N + j] * a[s*N + j];
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
	//print_matrix(gamma_T,1,N);	for(int row = 0 ; row < N; row++){
	for(int row = 0 ; row < N; row++){
		for(int col =row+1; col < N; col++){
			double temp = a[col*N+row];
			a[col * N + row]  = a[row * N + col];
			a[row*N + col] = temp;
			//printf(" %lf %lf \n ", 	transpose[col * rows + row] , a[row * cols + col]);
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
			//p[s] = 0.0;
			//beta_new[s]=0.0;
			for(int j = 0; j < N; j++){
				double temp =ab[(yt*N + s)*N + j] * beta[j]; //a[s*N +j] * beta[j] * b[yt*N + j];
				
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
		yt=yt1;
		
		
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
	free(ab);
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
	double* ab = (double*) malloc(N * N * (T-1) * sizeof(double));

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
        /*
		for(int v = 0; v < K; v++){
			//int indicator = (int)(yt == v);
			//if you use real gamma you have to divide by ct[t-1]
			b_new[v*N + s] += indicator[(T-1)*K + v]*gamma_Ts;
		}
        */
	}

	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] * gamma_T[s];
			//printf(" %lf %lf %lf \n", b[i*K + v], b_new[i*K + v] , gamma_sum[i]);
			b_new[v*N + s] = 0.0;
		}
	}
	//print_matrix(b,N,K);

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
					//printf("temp = %lf \n", temp);
					//print_matrix(a_new,N,N);			
			}
		}
		for(int bx = end; bx < N; bx+= block_size){
			const int end_x = bx + block_size;
			for(int i = by; i < end; i++){
				for(int j = bx; j < end_x; j++){
					double temp = a_new[j*N+i];
					a_new[j * N + i]  = a_new[i * N + j];
					a_new[i*N + j] = temp;
					//print_matrix(a_new,N,N);
				}
			}
		}	
	}
	//print_matrix(a_new,N,N);


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
	for(int s = 0; s<N-1; s++){// s=new_state
		double alphatNs = 0;
		//alpha[s*T + t] = 0;
		for(int j = 0; j < N; j++){//j=old_states
			double asNj =  a_new[s*N + j] * gamma_sum[j];
			a_new[s*N+j] = 0.0;
			a[s*N + j] = asNj;
			alphatNs += alpha[0*N + j] * asNj;
			//printf("%lf %lf %lf %lf %i \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t+1]],y[t]);
		}
		alphatNs *= b[yt*N + s];
		//print_matrix(alpha,N,T);
		ctt += alphatNs;
		alpha[1*N + s] = alphatNs;
	}
	
	// One iteration seperate to set gamma_sum to zero
	double alphatNs = 0;
	//alpha[s*T + t] = 0;
	for(int j = 0; j < N; j++){//j=old_states
		double gamma_sumj = gamma_sum[j];
		gamma_sum[j] =0.0;
		double asNj =  a_new[(N-1)*N + j] * gamma_sumj;
		a[(N-1)*N + j] = asNj;
		alphatNs += alpha[0*N + j] * asNj;
		//printf("%lf %lf %lf %lf %i \n", alpha[s*T + t], alpha[s*T + t-1], a[j*N+s], b[s*K+y[t+1]],y[t]);
		a_new[(N-1)*N+j] = 0.0;
	}
	alphatNs *= b[yt*N + (N-1)];
	//print_matrix(alpha,N,T);
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
			//alpha[s*T + t] = 0;
			for(int j = 0; j < N; j++){//j=old_states
				alphatNs += alpha[(t-1)*N + j] *a[s*N + j];
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
			alphatNs += alpha[(T-2)*N + j] * a[s*N + j];
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
	
	
	//print_matrix(beta,1,N);
	//print_matrix(gamma_T,1,N);

	//compute sum of xi and gamma from t= 0...T-2
    	yt = y[T-1];
	for(int t = T-1; t > 0; t--){
		const int yt1 = y[t-1];
		ctt = ct[t-1];
		for(int s = 0; s < N ; s++){
			double beta_news = 0.0;
			double alphat1Ns = alpha[(t-1)*N + s];
			//p[s] = 0.0;
			//beta_new[s]=0.0;
			for(int j = 0; j < N; j++){
				double temp = ab[(yt*N + s)*N + j] * beta[j];//a[s*N +j] * beta[j] * b[yt*N + j];
				
				//double xi_sjt = alphat1Ns * temp;
				a_new[s*N+j] +=alphat1Ns * temp;
				
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
				//int indicator = (int)(yt1 == v);
				//if you use real gamma you have to divide by ct[t-1]
				b_new[v*N + s] += indicator[(t-1)*K + v]*ps;
				
				//printf(" %i %lf \n ", indicator,p[i]);
			}
			*/
			//printf(" %lf %lf %lf \n", p[i],  ct[t-1],alpha[(t-1)*N+i]);
		}
		//printf("T = %li\n",t);
		

		double * temp = beta_new;
		beta_new = beta;
		beta = temp;
		yt=yt1;
		
		
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
	free(ab);
	
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
		double gamma_tot = gamma_Ts + gamma_sum[s] /* /ct[T-1] */;
		gamma_T[s] = 1./gamma_tot;
        	b_new[yt*N+s]+=gamma_Ts;
        /*
		for(int v = 0; v < K; v++){
			//int indicator = (int)(yt == v);
			//if you use real gamma you have to divide by ct[t-1]
			b_new[v*N + s] += indicator[(T-1)*K + v]*gamma_Ts;
		}
        */
	}

	//compute new emission matrix
	for(int v = 0; v < K; v++){
		for(int s = 0; s < N; s++){
			b[v*N + s] = b_new[v*N + s] * gamma_T[s];
			//printf(" %lf %lf %lf \n", b[i*K + v], b_new[i*K + v] , gamma_sum[i]);
		}
	}
	//print_matrix(b,N,K);

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

	double* beta = (double*) malloc(T  * sizeof(double));
	double* beta_new = (double*) malloc(T * sizeof(double));
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
        
        
        
        
        
        	const int unroll_inc = 4;
        
        	const int block_size = 4;
        
		for(int by = 0; by < hiddenStates; by+=block_size){
			const int end = by + block_size;
			
			//Diagonal 4x4 blocks
			double temp01 = transitionMatrix[by*hiddenStates+by+1];
			transitionMatrix[by * hiddenStates + by+1]  = transitionMatrix[(by+1) * hiddenStates + by];
			transitionMatrix[(by+1)*hiddenStates+ by] = temp01;	
			
			double temp02 = transitionMatrix[by*hiddenStates+by+2];
			transitionMatrix[by * hiddenStates + by+2]  = transitionMatrix[(by+2) * hiddenStates + by];
			transitionMatrix[(by+2)*hiddenStates+ by] = temp02;	
			
			double temp03 = transitionMatrix[by*hiddenStates+by+3];
			transitionMatrix[by * hiddenStates + by+3]  = transitionMatrix[(by+3) * hiddenStates + by];
			transitionMatrix[(by+3)*hiddenStates+ by] = temp03;	
			
			double temp12 = transitionMatrix[(by+1)*hiddenStates+by+2];
			transitionMatrix[(by+1) * hiddenStates + by+2]  = transitionMatrix[(by+2)* hiddenStates + (by+1)];
			transitionMatrix[(by+2)*hiddenStates+ (by+1)] = temp12;	
			
			double temp13 = transitionMatrix[(by+1)*hiddenStates+by+3];
			transitionMatrix[(by+1) * hiddenStates + by+3]  = transitionMatrix[(by+3) * hiddenStates + (by+1)];
			transitionMatrix[(by+3)*hiddenStates+ (by+1)] = temp13;	

			double temp23 = transitionMatrix[(by+2)*hiddenStates+by+3];
			transitionMatrix[(by+2) * hiddenStates + by+3]  = transitionMatrix[(by+3) * hiddenStates + (by+2)];
			transitionMatrix[(by+3)*hiddenStates+ (by+2)] = temp23;	

			
			//Offdiagonal blocks
			for(int bx = end; bx < hiddenStates; bx+= block_size){
		
				double temp00 = transitionMatrix[by*hiddenStates+bx];
				transitionMatrix[by * hiddenStates + bx]  = transitionMatrix[bx * hiddenStates + by];
				transitionMatrix[bx*hiddenStates+ by] = temp00;	
	
				double temp01 = transitionMatrix[by*hiddenStates+bx + 1];
				transitionMatrix[by * hiddenStates + bx + 1]  = transitionMatrix[(bx + 1) * hiddenStates + by];
				transitionMatrix[(bx + 1)*hiddenStates+ by] = temp01;	
				
				double temp02 = transitionMatrix[by*hiddenStates+bx + 2];
				transitionMatrix[by * hiddenStates + bx + 2]  = transitionMatrix[(bx + 2)  * hiddenStates + by];
				transitionMatrix[(bx + 2) *hiddenStates+ by] = temp02;	
				
				double temp03 = transitionMatrix[by*hiddenStates+bx + 3];
				transitionMatrix[by * hiddenStates + bx + 3]  = transitionMatrix[(bx + 3) * hiddenStates + by];
				transitionMatrix[(bx + 3) *hiddenStates+ by] = temp03;
						
						
				double temp10 = transitionMatrix[(by+1)*hiddenStates+bx];
				transitionMatrix[(by+1) * hiddenStates + bx]  = transitionMatrix[bx * hiddenStates + (by+1)];
				transitionMatrix[bx*hiddenStates+ (by+1)] = temp10;	
			
				double temp11 = transitionMatrix[(by+1)*hiddenStates+bx + 1];
				transitionMatrix[(by+1) * hiddenStates + bx + 1]  = transitionMatrix[(bx + 1) * hiddenStates + (by+1)];
				transitionMatrix[(bx + 1)*hiddenStates+ (by+1)] = temp11;	
					
				double temp12 = transitionMatrix[(by+1)*hiddenStates+bx + 2];
				transitionMatrix[(by+1) * hiddenStates + bx + 2]  = transitionMatrix[(bx + 2)  * hiddenStates + (by+1)];
				transitionMatrix[(bx + 2) *hiddenStates+ (by+1)] = temp12;	
					
				double temp13 = transitionMatrix[(by+1)*hiddenStates+bx + 3];
				transitionMatrix[(by+1) * hiddenStates + bx + 3]  = transitionMatrix[(bx + 3) * hiddenStates + (by+1)];
				transitionMatrix[(bx + 3) *hiddenStates+ (by+1)] = temp13;
						
						
				double temp20 = transitionMatrix[(by+2)*hiddenStates+bx];
				transitionMatrix[(by+2)* hiddenStates + bx]  = transitionMatrix[bx * hiddenStates + (by+2)];
				transitionMatrix[bx*hiddenStates+ (by+2)] = temp20;	
					
				double temp21 = transitionMatrix[(by+2)*hiddenStates+bx + 1];
				transitionMatrix[(by+2) * hiddenStates + bx + 1]  = transitionMatrix[(bx + 1) * hiddenStates + (by+2)];
				transitionMatrix[(bx + 1)*hiddenStates+(by+2)] = temp21;	
					
				double temp22 = transitionMatrix[(by+2)*hiddenStates+bx + 2];
				transitionMatrix[(by+2) * hiddenStates + bx + 2]  = transitionMatrix[(bx + 2)  * hiddenStates + (by+2)];
				transitionMatrix[(bx + 2) *hiddenStates+ (by+2)] = temp22;	
					
				double temp23 = transitionMatrix[(by+2)*hiddenStates+bx + 3];
				transitionMatrix[(by+2)* hiddenStates + bx + 3]  = transitionMatrix[(bx + 3) * hiddenStates + (by+2)];
				transitionMatrix[(bx + 3) *hiddenStates+ (by+2)] = temp23;
					
						
				double temp30 = transitionMatrix[(by+3)*hiddenStates+bx];
				transitionMatrix[(by+3) * hiddenStates + bx]  = transitionMatrix[bx * hiddenStates +(by+3)];
				transitionMatrix[bx*hiddenStates+ (by+3)] = temp30;	
					
				double temp31 = transitionMatrix[(by+3)*hiddenStates+bx + 1];
				transitionMatrix[(by+3) * hiddenStates + bx + 1]  = transitionMatrix[(bx + 1) * hiddenStates + (by+3)];
				transitionMatrix[(bx + 1)*hiddenStates+ (by+3)] = temp31;	
				
				double temp32 = transitionMatrix[(by+3)*hiddenStates+bx + 2];
				transitionMatrix[(by+3) * hiddenStates + bx + 2]  = transitionMatrix[(bx + 2)  * hiddenStates + (by+3)];
				transitionMatrix[(bx + 2) *hiddenStates+ (by+3)] = temp32;	
					
				double temp33 = transitionMatrix[(by+3)*hiddenStates+bx + 3];
				transitionMatrix[(by+3) * hiddenStates + bx + 3]  = transitionMatrix[(bx + 3) * hiddenStates + (by+3)];
				transitionMatrix[(bx + 3) *hiddenStates+ (by+3)] = temp33;
						
			
			}	
		}
	
	

		double ct0 = 0.0;
		
		//compute alpha(0) and scaling factor for t = 0
		int y0 = observations[0];
		for(int s = 0; s < hiddenStates; s+=unroll_inc){
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
		for(int s = 0; s < hiddenStates; s+=unroll_inc){
			//s
			alpha[s] *= ct0;
			//s+1
			alpha[s+1] *= ct0;
			//s+2
			alpha[s+2] *= ct0;
			//s+3
			alpha[s+3] *= ct0;
		}
		
		ct[0] = ct0;

		for(int t = 1; t < T-1; t++){
			double ctt = 0.0;	
			const int yt = observations[t];	
			for(int s = 0; s<hiddenStates; s+=unroll_inc){// s=new_state
				double alphatNs0 = 0;
				double alphatNs1 = 0;
				double alphatNs2 = 0;
				double alphatNs3 = 0;
				
				for(int j = 0; j < hiddenStates; j+=unroll_inc){//j=old_states
				
					double alphaFactor0 = alpha[(t-1)*hiddenStates + j];
					double alphaFactor1 = alpha[(t-1)*hiddenStates + j+1];
					double alphaFactor2 = alpha[(t-1)*hiddenStates + j+2]; 
					double alphaFactor3 = alpha[(t-1)*hiddenStates + j+3]; 
				
					alphatNs0 += alphaFactor0 * transitionMatrix[s*hiddenStates + j];
					alphatNs0 += alphaFactor1 * transitionMatrix[s*hiddenStates + j+1];
					alphatNs0 += alphaFactor2 * transitionMatrix[s*hiddenStates + j+2];
					alphatNs0 += alphaFactor3 * transitionMatrix[s*hiddenStates + j+3];
					
					alphatNs1 += alphaFactor0 * transitionMatrix[(s+1)*hiddenStates + j];
					alphatNs1 += alphaFactor1 * transitionMatrix[(s+1)*hiddenStates + j+1];
					alphatNs1 += alphaFactor2 * transitionMatrix[(s+1)*hiddenStates + j+2];
					alphatNs1 += alphaFactor3 * transitionMatrix[(s+1)*hiddenStates + j+3];
					
					alphatNs2 += alphaFactor0 * transitionMatrix[(s+2)*hiddenStates + j];
					alphatNs2 += alphaFactor1 * transitionMatrix[(s+2)*hiddenStates + j+1];
					alphatNs2 += alphaFactor2 * transitionMatrix[(s+2)*hiddenStates + j+2];
					alphatNs2 += alphaFactor3 * transitionMatrix[(s+2)*hiddenStates + j+3];
					
					alphatNs3 += alphaFactor0 * transitionMatrix[(s+3)*hiddenStates + j];
					alphatNs3 += alphaFactor1 * transitionMatrix[(s+3)*hiddenStates + j+1];
					alphatNs3 += alphaFactor2 * transitionMatrix[(s+3)*hiddenStates + j+2];
					alphatNs3 += alphaFactor3 * transitionMatrix[(s+3)*hiddenStates + j+3];
				}
				alphatNs0 *= emissionMatrix[yt*hiddenStates + s];
				ctt += alphatNs0;
				alpha[t*hiddenStates + s] = alphatNs0;
				
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
			for(int s = 0; s<hiddenStates; s+=unroll_inc){// s=new_state
				alpha[t*hiddenStates+s] *= ctt;
				alpha[t*hiddenStates+s+1] *= ctt;
				alpha[t*hiddenStates+s+2] *= ctt;
				alpha[t*hiddenStates+s+3] *= ctt;
			}
			ct[t] = ctt;
		}
		
	
		double ctt = 0.0;	
		int yt = observations[T-1];	

		
		for(int s = 0; s<hiddenStates; s+=unroll_inc){// s=new_state
			double alphatNs0 = 0;
			double alphatNs1 = 0;
			double alphatNs2 = 0;
			double alphatNs3 = 0;
			
			for(int j = 0; j < hiddenStates; j+=unroll_inc){//j=old_states
			
				double alphaFactor0 = alpha[(T-2)*hiddenStates + j];
				double alphaFactor1 = alpha[(T-2)*hiddenStates + j+1];
				double alphaFactor2 = alpha[(T-2)*hiddenStates + j+2]; 
				double alphaFactor3 = alpha[(T-2)*hiddenStates + j+3]; 
			
				alphatNs0 += alphaFactor0 * transitionMatrix[s*hiddenStates + j];
				alphatNs0 += alphaFactor1 * transitionMatrix[s*hiddenStates + j+1];
				alphatNs0 += alphaFactor2 * transitionMatrix[s*hiddenStates + j+2];
				alphatNs0 += alphaFactor3 * transitionMatrix[s*hiddenStates + j+3];
				
				alphatNs1 += alphaFactor0 * transitionMatrix[(s+1)*hiddenStates + j];
				alphatNs1 += alphaFactor1 * transitionMatrix[(s+1)*hiddenStates + j+1];
				alphatNs1 += alphaFactor2 * transitionMatrix[(s+1)*hiddenStates + j+2];
				alphatNs1 += alphaFactor3 * transitionMatrix[(s+1)*hiddenStates + j+3];
				
				alphatNs2 += alphaFactor0 * transitionMatrix[(s+2)*hiddenStates + j];
				alphatNs2 += alphaFactor1 * transitionMatrix[(s+2)*hiddenStates + j+1];
				alphatNs2 += alphaFactor2 * transitionMatrix[(s+2)*hiddenStates + j+2];
				alphatNs2 += alphaFactor3 * transitionMatrix[(s+2)*hiddenStates + j+3];
					
				alphatNs3 += alphaFactor0 * transitionMatrix[(s+3)*hiddenStates + j];
				alphatNs3 += alphaFactor1 * transitionMatrix[(s+3)*hiddenStates + j+1];
				alphatNs3 += alphaFactor2 * transitionMatrix[(s+3)*hiddenStates + j+2];
				alphatNs3 += alphaFactor3 * transitionMatrix[(s+3)*hiddenStates + j+3];
			}
			alphatNs0 *= emissionMatrix[yt*hiddenStates + s];
			ctt += alphatNs0;
			alpha[(T-1)*hiddenStates + s] = alphatNs0;
			
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
		for(int s = 0; s<hiddenStates; s+=unroll_inc){// s=new_state
			//s
			double alphaT1Ns0 = alpha[(T-1) * hiddenStates + s]*ctt;
			alpha[(T-1)*hiddenStates + s] = alphaT1Ns0;
			gamma_T[s] = alphaT1Ns0 /* *ct[T-1]*/;
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

		for(int s = 0; s < hiddenStates; s+=unroll_inc){
			beta[s] = /* 1* */ctt;
			beta[s+1] = /* 1* */ctt;
			beta[s+2] = /* 1* */ctt;
			beta[s+3] = /* 1* */ctt;
			gamma_sum[s] = 0.0;
			gamma_sum[s+1] = 0.0;
			gamma_sum[s+2] = 0.0;
			gamma_sum[s+3] = 0.0;
			for(int j = 0; j < hiddenStates; j+=unroll_inc){
				a_new[s*hiddenStates + j] =0.0;
				a_new[s*hiddenStates + j+1] =0.0;
				a_new[s*hiddenStates + j+2] =0.0;
				a_new[s*hiddenStates + j+3] =0.0;
								
				a_new[(s+1)*hiddenStates + j] =0.0;
				a_new[(s+1)*hiddenStates + j+1] =0.0;
				a_new[(s+1)*hiddenStates + j+2] =0.0;
				a_new[(s+1)*hiddenStates + j+3] =0.0;
			
				a_new[(s+2)*hiddenStates + j] =0.0;
				a_new[(s+2)*hiddenStates + j+1] =0.0;
				a_new[(s+2)*hiddenStates + j+2] =0.0;
				a_new[(s+2)*hiddenStates + j+3] =0.0;
			
				a_new[(s+3)*hiddenStates + j] =0.0;
				a_new[(s+3)*hiddenStates + j+1] =0.0;
				a_new[(s+3)*hiddenStates + j+2] =0.0;
				a_new[(s+3)*hiddenStates + j+3] =0.0;
			}
		}


		for(int v = 0;  v < differentObservables; v+=unroll_inc){
			for(int s = 0; s < hiddenStates; s+=unroll_inc){
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

		//Transpose transitionMatrix
		for(int by = 0; by < hiddenStates; by+=block_size){
			const int end = by + block_size;
			
			//Diagonal 4x4 blocks
			double temp01 = transitionMatrix[by*hiddenStates+by+1];
			transitionMatrix[by * hiddenStates + by+1]  = transitionMatrix[(by+1) * hiddenStates + by];
			transitionMatrix[(by+1)*hiddenStates+ by] = temp01;	
			
			double temp02 = transitionMatrix[by*hiddenStates+by+2];
			transitionMatrix[by * hiddenStates + by+2]  = transitionMatrix[(by+2) * hiddenStates + by];
			transitionMatrix[(by+2)*hiddenStates+ by] = temp02;	
			
			double temp03 = transitionMatrix[by*hiddenStates+by+3];
			transitionMatrix[by * hiddenStates + by+3]  = transitionMatrix[(by+3) * hiddenStates + by];
			transitionMatrix[(by+3)*hiddenStates+ by] = temp03;	
			
			double temp12 = transitionMatrix[(by+1)*hiddenStates+by+2];
			transitionMatrix[(by+1) * hiddenStates + by+2]  = transitionMatrix[(by+2)* hiddenStates + (by+1)];
			transitionMatrix[(by+2)*hiddenStates+ (by+1)] = temp12;	
			
			double temp13 = transitionMatrix[(by+1)*hiddenStates+by+3];
			transitionMatrix[(by+1) * hiddenStates + by+3]  = transitionMatrix[(by+3) * hiddenStates + (by+1)];
			transitionMatrix[(by+3)*hiddenStates+ (by+1)] = temp13;	

			double temp23 = transitionMatrix[(by+2)*hiddenStates+by+3];
			transitionMatrix[(by+2) * hiddenStates + by+3]  = transitionMatrix[(by+3) * hiddenStates + (by+2)];
			transitionMatrix[(by+3)*hiddenStates+ (by+2)] = temp23;	

			
			//Offdiagonal blocks
			for(int bx = end; bx < hiddenStates; bx+= block_size){
				const int end_x = block_size + bx;
				for(int i = by; i < end; i+=unroll_inc){
					for(int j = bx; j < end_x; j+=unroll_inc){
						double temp00 = transitionMatrix[i*hiddenStates+j];
						transitionMatrix[i * hiddenStates + j]  = transitionMatrix[j * hiddenStates + i];
						transitionMatrix[j*hiddenStates+ i] = temp00;	
					
						double temp01 = transitionMatrix[i*hiddenStates+j + 1];
						transitionMatrix[i * hiddenStates + j + 1]  = transitionMatrix[(j + 1) * hiddenStates + i];
						transitionMatrix[(j + 1)*hiddenStates+ i] = temp01;	
					
						double temp02 = transitionMatrix[i*hiddenStates+j + 2];
						transitionMatrix[i * hiddenStates + j + 2]  = transitionMatrix[(j + 2)  * hiddenStates + i];
						transitionMatrix[(j + 2) *hiddenStates+ i] = temp02;	
					
						double temp03 = transitionMatrix[i*hiddenStates+j + 3];
						transitionMatrix[i * hiddenStates + j + 3]  = transitionMatrix[(j + 3) * hiddenStates + i];
						transitionMatrix[(j + 3) *hiddenStates+ i] = temp03;
						
						
						double temp10 = transitionMatrix[(i+1)*hiddenStates+j];
						transitionMatrix[(i+1) * hiddenStates + j]  = transitionMatrix[j * hiddenStates + (i+1)];
						transitionMatrix[j*hiddenStates+ (i+1)] = temp10;	
					
						double temp11 = transitionMatrix[(i+1)*hiddenStates+j + 1];
						transitionMatrix[(i+1) * hiddenStates + j + 1]  = transitionMatrix[(j + 1) * hiddenStates + (i+1)];
						transitionMatrix[(j + 1)*hiddenStates+ (i+1)] = temp11;	
					
						double temp12 = transitionMatrix[(i+1)*hiddenStates+j + 2];
						transitionMatrix[(i+1) * hiddenStates + j + 2]  = transitionMatrix[(j + 2)  * hiddenStates + (i+1)];
						transitionMatrix[(j + 2) *hiddenStates+ (i+1)] = temp12;	
					
						double temp13 = transitionMatrix[(i+1)*hiddenStates+j + 3];
						transitionMatrix[(i+1) * hiddenStates + j + 3]  = transitionMatrix[(j + 3) * hiddenStates + (i+1)];
						transitionMatrix[(j + 3) *hiddenStates+ (i+1)] = temp13;
						
						
						double temp20 = transitionMatrix[(i+2)*hiddenStates+j];
						transitionMatrix[(i+2)* hiddenStates + j]  = transitionMatrix[j * hiddenStates + (i+2)];
						transitionMatrix[j*hiddenStates+ (i+2)] = temp20;	
					
						double temp21 = transitionMatrix[(i+2)*hiddenStates+j + 1];
						transitionMatrix[(i+2) * hiddenStates + j + 1]  = transitionMatrix[(j + 1) * hiddenStates + (i+2)];
						transitionMatrix[(j + 1)*hiddenStates+(i+2)] = temp21;	
					
						double temp22 = transitionMatrix[(i+2)*hiddenStates+j + 2];
						transitionMatrix[(i+2) * hiddenStates + j + 2]  = transitionMatrix[(j + 2)  * hiddenStates + (i+2)];
						transitionMatrix[(j + 2) *hiddenStates+ (i+2)] = temp22;	
					
						double temp23 = transitionMatrix[(i+2)*hiddenStates+j + 3];
						transitionMatrix[(i+2)* hiddenStates + j + 3]  = transitionMatrix[(j + 3) * hiddenStates + (i+2)];
						transitionMatrix[(j + 3) *hiddenStates+ (i+2)] = temp23;
						
						
						double temp30 = transitionMatrix[(i+3)*hiddenStates+j];
						transitionMatrix[(i+3) * hiddenStates + j]  = transitionMatrix[j * hiddenStates +(i+3)];
						transitionMatrix[j*hiddenStates+ (i+3)] = temp30;	
					
						double temp31 = transitionMatrix[(i+3)*hiddenStates+j + 1];
						transitionMatrix[(i+3) * hiddenStates + j + 1]  = transitionMatrix[(j + 1) * hiddenStates + (i+3)];
						transitionMatrix[(j + 1)*hiddenStates+ (i+3)] = temp31;	
					
						double temp32 = transitionMatrix[(i+3)*hiddenStates+j + 2];
						transitionMatrix[(i+3) * hiddenStates + j + 2]  = transitionMatrix[(j + 2)  * hiddenStates + (i+3)];
						transitionMatrix[(j + 2) *hiddenStates+ (i+3)] = temp32;	
					
						double temp33 = transitionMatrix[(i+3)*hiddenStates+j + 3];
						transitionMatrix[(i+3) * hiddenStates + j + 3]  = transitionMatrix[(j + 3) * hiddenStates + (i+3)];
						transitionMatrix[(j + 3) *hiddenStates+ (i+3)] = temp33;
						
							
					}
				}
			}	
		}
	
	
		for(int v = 0; v < differentObservables; v++){
			for(int s = 0; s < hiddenStates; s+=unroll_inc){
				for(int j = 0; j < hiddenStates; j+=unroll_inc){
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
		
		
		
   		yt = observations[T-1];
		for(int t = T-1; t > 0; t--){
			const int yt1 = observations[t-1];
			const double ctt = ct[t-1];
			for(int s = 0; s < hiddenStates ; s+=unroll_inc){
				double beta_news0 = 0.0;
				double alphat1Ns0 = alpha[(t-1)*hiddenStates + s];
				double beta_news1 = 0.0;
				double alphat1Ns1 = alpha[(t-1)*hiddenStates + s+1];
				double beta_news2 = 0.0;
				double alphat1Ns2 = alpha[(t-1)*hiddenStates + s+2];
				double beta_news3 = 0.0;
				double alphat1Ns3 = alpha[(t-1)*hiddenStates + s+3];
				for(int j = 0; j < hiddenStates; j+=unroll_inc){
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
        

		//for(int i = 0; i < 1; i++){
		do{
		
		
		
		
			yt = observations[T-1];
			//add remaining parts of the sum of gamma 
			for(int s = 0; s < hiddenStates; s+=unroll_inc){
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
			for(int v = 0; v < differentObservables; v+=unroll_inc){
				for(int s = 0; s < hiddenStates; s+=unroll_inc){
				
					double gamma_T0 = gamma_T[s];
					double gamma_T1 = gamma_T[s+1];
					double gamma_T2 = gamma_T[s+2];
					double gamma_T3 = gamma_T[s+3];
				
					emissionMatrix[v*hiddenStates + s] = b_new[v*hiddenStates + s] * gamma_T0;
					b_new[v*hiddenStates + s] = 0.0;
					
					emissionMatrix[v*hiddenStates + s+1] = b_new[v*hiddenStates + s+1] * gamma_T1;
					b_new[v*hiddenStates + s+1] = 0.0;
					
					emissionMatrix[v*hiddenStates + s+2] = b_new[v*hiddenStates + s+2] * gamma_T2;
					b_new[v*hiddenStates + s+2] = 0.0;
					
					emissionMatrix[v*hiddenStates + s+3] = b_new[v*hiddenStates + s+3] * gamma_T3;
					b_new[v*hiddenStates + s+3] = 0.0;
					
					
					
					emissionMatrix[(v+1)*hiddenStates + s] = b_new[(v+1)*hiddenStates + s] * gamma_T0;
					b_new[(v+1)*hiddenStates + s] = 0.0;
					
					emissionMatrix[(v+1)*hiddenStates + s+1] = b_new[(v+1)*hiddenStates + s+1] * gamma_T1;
					b_new[(v+1)*hiddenStates + s+1] = 0.0;
					
					emissionMatrix[(v+1)*hiddenStates + s+2] = b_new[(v+1)*hiddenStates + s+2] * gamma_T2;
					b_new[(v+1)*hiddenStates + s+2] = 0.0;
					
					emissionMatrix[(v+1)*hiddenStates + s+3] = b_new[(v+1)*hiddenStates + s+3] * gamma_T3;
					b_new[(v+1)*hiddenStates + s+3] = 0.0;
					
					
					
					emissionMatrix[(v+2)*hiddenStates + s] = b_new[(v+2)*hiddenStates + s] * gamma_T0;
					b_new[(v+2)*hiddenStates + s] = 0.0;
					
					emissionMatrix[(v+2)*hiddenStates + s+1] = b_new[(v+2)*hiddenStates + s+1] * gamma_T1;
					b_new[(v+2)*hiddenStates + s+1] = 0.0;
					
					emissionMatrix[(v+2)*hiddenStates + s+2] = b_new[(v+2)*hiddenStates + s+2] * gamma_T2;
					b_new[(v+2)*hiddenStates + s+2] = 0.0;
					
					emissionMatrix[(v+2)*hiddenStates + s+3] = b_new[(v+2)*hiddenStates + s+3] * gamma_T3;
					b_new[(v+2)*hiddenStates + s+3] = 0.0;
					
					
					
					emissionMatrix[(v+3)*hiddenStates + s] = b_new[(v+3)*hiddenStates + s] * gamma_T0;
					b_new[(v+3)*hiddenStates + s] = 0.0;
					
					emissionMatrix[(v+3)*hiddenStates + s+1] = b_new[(v+3)*hiddenStates + s+1] * gamma_T1;
					b_new[(v+3)*hiddenStates + s+1] = 0.0;
					
					emissionMatrix[(v+3)*hiddenStates + s+2] = b_new[(v+3)*hiddenStates + s+2] * gamma_T2;
					b_new[(v+3)*hiddenStates + s+2] = 0.0;
					
					emissionMatrix[(v+3)*hiddenStates + s+3] = b_new[(v+3)*hiddenStates + s+3] * gamma_T3;
					b_new[(v+3)*hiddenStates + s+3] = 0.0;
		
					
					
		
				}
			}

			//FORWARD

			//Transpose a_new. Note that it is not necessary to transpose matrix a.
			//Transpose transitionMatrix
			    
			for(int by = 0; by < hiddenStates; by+=block_size){
				const int end = by + block_size;
			
				//Diagonal 4x4 blocks
				double temp01 = a_new[by*hiddenStates+by+1];
				a_new[by * hiddenStates + by+1]  = a_new[(by+1) * hiddenStates + by];
				a_new[(by+1)*hiddenStates+ by] = temp01;	
				
				double temp02 = a_new[by*hiddenStates+by+2];
				a_new[by * hiddenStates + by+2]  = a_new[(by+2) * hiddenStates + by];
				a_new[(by+2)*hiddenStates+ by] = temp02;	
				
				double temp03 = a_new[by*hiddenStates+by+3];
				a_new[by * hiddenStates + by+3]  = a_new[(by+3) * hiddenStates + by];
				a_new[(by+3)*hiddenStates+ by] = temp03;	
			
				double temp12 = a_new[(by+1)*hiddenStates+by+2];
				a_new[(by+1) * hiddenStates + by+2]  = a_new[(by+2)* hiddenStates + (by+1)];
				a_new[(by+2)*hiddenStates+ (by+1)] = temp12;	
				
				double temp13 = a_new[(by+1)*hiddenStates+by+3];
				a_new[(by+1) * hiddenStates + by+3]  = a_new[(by+3) * hiddenStates + (by+1)];
				a_new[(by+3)*hiddenStates+ (by+1)] = temp13;	
	
				double temp23 = a_new[(by+2)*hiddenStates+by+3];
				a_new[(by+2) * hiddenStates + by+3]  = a_new[(by+3) * hiddenStates + (by+2)];
				a_new[(by+3)*hiddenStates+ (by+2)] = temp23;	
	
				
				//Offdiagonal blocks
				for(int bx = end; bx < hiddenStates; bx+= block_size){
			
					double temp00 = a_new[by*hiddenStates+bx];
					a_new[by * hiddenStates + bx]  = a_new[bx * hiddenStates + by];
					a_new[bx*hiddenStates+ by] = temp00;	
		
					double temp01 = a_new[by*hiddenStates+bx + 1];
					a_new[by * hiddenStates + bx + 1]  = a_new[(bx + 1) * hiddenStates + by];
					a_new[(bx + 1)*hiddenStates+ by] = temp01;	
					
					double temp02 = a_new[by*hiddenStates+bx + 2];
					a_new[by * hiddenStates + bx + 2]  = a_new[(bx + 2)  * hiddenStates + by];
					a_new[(bx + 2) *hiddenStates+ by] = temp02;	
					
					double temp03 = a_new[by*hiddenStates+bx + 3];
					a_new[by * hiddenStates + bx + 3]  = a_new[(bx + 3) * hiddenStates + by];
					a_new[(bx + 3) *hiddenStates+ by] = temp03;
							
							
					double temp10 = a_new[(by+1)*hiddenStates+bx];
					a_new[(by+1) * hiddenStates + bx]  = a_new[bx * hiddenStates + (by+1)];
					a_new[bx*hiddenStates+ (by+1)] = temp10;	
				
					double temp11 = a_new[(by+1)*hiddenStates+bx + 1];
					a_new[(by+1) * hiddenStates + bx + 1]  = a_new[(bx + 1) * hiddenStates + (by+1)];
					a_new[(bx + 1)*hiddenStates+ (by+1)] = temp11;	
						
					double temp12 = a_new[(by+1)*hiddenStates+bx + 2];
					a_new[(by+1) * hiddenStates + bx + 2]  = a_new[(bx + 2)  * hiddenStates + (by+1)];
					a_new[(bx + 2) *hiddenStates+ (by+1)] = temp12;	
						
					double temp13 = a_new[(by+1)*hiddenStates+bx + 3];
					a_new[(by+1) * hiddenStates + bx + 3]  = a_new[(bx + 3) * hiddenStates + (by+1)];
					a_new[(bx + 3) *hiddenStates+ (by+1)] = temp13;
							
							
					double temp20 = a_new[(by+2)*hiddenStates+bx];
					a_new[(by+2)* hiddenStates + bx]  = a_new[bx * hiddenStates + (by+2)];
					a_new[bx*hiddenStates+ (by+2)] = temp20;	
						
					double temp21 = a_new[(by+2)*hiddenStates+bx + 1];
					a_new[(by+2) * hiddenStates + bx + 1]  = a_new[(bx + 1) * hiddenStates + (by+2)];
					a_new[(bx + 1)*hiddenStates+(by+2)] = temp21;	
						
					double temp22 = a_new[(by+2)*hiddenStates+bx + 2];
					a_new[(by+2) * hiddenStates + bx + 2]  = a_new[(bx + 2)  * hiddenStates + (by+2)];
					a_new[(bx + 2) *hiddenStates+ (by+2)] = temp22;	
						
					double temp23 = a_new[(by+2)*hiddenStates+bx + 3];
					a_new[(by+2)* hiddenStates + bx + 3]  = a_new[(bx + 3) * hiddenStates + (by+2)];
					a_new[(bx + 3) *hiddenStates+ (by+2)] = temp23;
						
							
					double temp30 = a_new[(by+3)*hiddenStates+bx];
					a_new[(by+3) * hiddenStates + bx]  = a_new[bx * hiddenStates +(by+3)];
					a_new[bx*hiddenStates+ (by+3)] = temp30;	
						
					double temp31 = a_new[(by+3)*hiddenStates+bx + 1];
					a_new[(by+3) * hiddenStates + bx + 1]  = a_new[(bx + 1) * hiddenStates + (by+3)];
					a_new[(bx + 1)*hiddenStates+ (by+3)] = temp31;	
					
					double temp32 = a_new[(by+3)*hiddenStates+bx + 2];
					a_new[(by+3) * hiddenStates + bx + 2]  = a_new[(bx + 2)  * hiddenStates + (by+3)];
					a_new[(bx + 2) *hiddenStates+ (by+3)] = temp32;	
							
					double temp33 = a_new[(by+3)*hiddenStates+bx + 3];
					a_new[(by+3) * hiddenStates + bx + 3]  = a_new[(bx + 3) * hiddenStates + (by+3)];
					a_new[(bx + 3) *hiddenStates+ (by+3)] = temp33;
						
			
				}	
			}

			double ctt = 0.0;

			//compute alpha(0) and scaling factor for t = 0
			int y0 = observations[0];
		  
		        for(int s = 0; s < hiddenStates; s+=unroll_inc){
				//s
		  	      double alphas0 = stateProb[s] * emissionMatrix[y0*hiddenStates + s];
		  	      ctt += alphas0;
		  	      alpha[s] = alphas0;
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
	        	for(int s = 0; s < hiddenStates; s+=unroll_inc){
		        	alpha[s] *= ctt;
		        	alpha[s+1] *= ctt;
		        	alpha[s+2] *= ctt;
		        	alpha[s+3] *= ctt;
	        	}
	
			ct[0] = ctt;

			//Compute alpha(1) and scale transitionMatrix
			ctt = 0.0;	
			yt = observations[1];	
			
			for(int s = 0; s<hiddenStates; s+=unroll_inc){// s=new_state
				double alphatNs0 = 0;
				double alphatNs1 = 0;
				double alphatNs2 = 0;
				double alphatNs3 = 0;
			
				for(int j = 0; j < hiddenStates; j+=unroll_inc){//j=old_states
				
					double alphaFactor0 = alpha[j];
					double alphaFactor1 = alpha[j+1];
					double alphaFactor2 = alpha[j+2]; 
					double alphaFactor3 = alpha[j+3];
					
					double gamma_sum0 = gamma_sum[j];
					double gamma_sum1 = gamma_sum[j+1];
					double gamma_sum2 = gamma_sum[j+2];
					double gamma_sum3 = gamma_sum[j+3];
					
					double as0Nj0 = a_new[s*hiddenStates+j] * gamma_sum0; 
					double as0Nj1 = a_new[s*hiddenStates+j+1] * gamma_sum1; 
					double as0Nj2 = a_new[s*hiddenStates+j+2] * gamma_sum2; 
					double as0Nj3 = a_new[s*hiddenStates+j+3] * gamma_sum3; 
					
					double as1Nj0 = a_new[(s+1)*hiddenStates+j] * gamma_sum0; 
					double as1Nj1 = a_new[(s+1)*hiddenStates+j+1] * gamma_sum1; 
					double as1Nj2 = a_new[(s+1)*hiddenStates+j+2] * gamma_sum2; 
					double as1Nj3 = a_new[(s+1)*hiddenStates+j+3] * gamma_sum3; 
					
					double as2Nj0 = a_new[(s+2)*hiddenStates+j] * gamma_sum0; 
					double as2Nj1 = a_new[(s+2)*hiddenStates+j+1] * gamma_sum1; 
					double as2Nj2 = a_new[(s+2)*hiddenStates+j+2] * gamma_sum2; 
					double as2Nj3 = a_new[(s+2)*hiddenStates+j+3] * gamma_sum3; 
					
					double as3Nj0 = a_new[(s+3)*hiddenStates+j] * gamma_sum0; 
					double as3Nj1 = a_new[(s+3)*hiddenStates+j+1] * gamma_sum1; 
					double as3Nj2 = a_new[(s+3)*hiddenStates+j+2] * gamma_sum2; 
					double as3Nj3 = a_new[(s+3)*hiddenStates+j+3] * gamma_sum3; 
				
					a_new[s*hiddenStates+j] = 0.0;
					a_new[s*hiddenStates+j+1] = 0.0;
					a_new[s*hiddenStates+j+2] = 0.0;
					a_new[s*hiddenStates+j+3] = 0.0;
					
					a_new[(s+1)*hiddenStates+j] = 0.0;
					a_new[(s+1)*hiddenStates+j+1] = 0.0;
					a_new[(s+1)*hiddenStates+j+2] = 0.0;
					a_new[(s+1)*hiddenStates+j+3] = 0.0;
					
					a_new[(s+2)*hiddenStates+j] = 0.0;
					a_new[(s+2)*hiddenStates+j+1] = 0.0;
					a_new[(s+2)*hiddenStates+j+2] = 0.0;
					a_new[(s+2)*hiddenStates+j+3] = 0.0;
					
					a_new[(s+3)*hiddenStates+j] = 0.0;
					a_new[(s+3)*hiddenStates+j+1] = 0.0;
					a_new[(s+3)*hiddenStates+j+2] = 0.0;
					a_new[(s+3)*hiddenStates+j+3] = 0.0;
					
					transitionMatrix[s*hiddenStates + j] = as0Nj0;
					transitionMatrix[s*hiddenStates + j+1] = as0Nj1;
					transitionMatrix[s*hiddenStates + j+2] = as0Nj2;
					transitionMatrix[s*hiddenStates + j+3] = as0Nj3;
					
					transitionMatrix[(s+1)*hiddenStates + j] = as1Nj0;
					transitionMatrix[(s+1)*hiddenStates + j+1] = as1Nj1;
					transitionMatrix[(s+1)*hiddenStates + j+2] = as1Nj2;
					transitionMatrix[(s+1)*hiddenStates + j+3] = as1Nj3;
					
					transitionMatrix[(s+2)*hiddenStates + j] = as2Nj0;
					transitionMatrix[(s+2)*hiddenStates + j+1] = as2Nj1;
					transitionMatrix[(s+2)*hiddenStates + j+2] = as2Nj2;
					transitionMatrix[(s+2)*hiddenStates + j+3] = as2Nj3;
					
					transitionMatrix[(s+3)*hiddenStates + j] = as3Nj0;
					transitionMatrix[(s+3)*hiddenStates + j+1] = as3Nj1;
					transitionMatrix[(s+3)*hiddenStates + j+2] = as3Nj2;
					transitionMatrix[(s+3)*hiddenStates + j+3] = as3Nj3;
					
				
					alphatNs0 += alphaFactor0 * as0Nj0;
					alphatNs0 += alphaFactor1 * as0Nj1;
					alphatNs0 += alphaFactor2 * as0Nj1;
					alphatNs0 += alphaFactor3 * as0Nj3;
					
					alphatNs1 += alphaFactor0 * as1Nj0;
					alphatNs1 += alphaFactor1 * as1Nj1;
					alphatNs1 += alphaFactor2 * as1Nj1;
					alphatNs1 += alphaFactor3 * as1Nj3;
					
					alphatNs2 += alphaFactor0 * as2Nj0;
					alphatNs2 += alphaFactor1 * as2Nj1;
					alphatNs2 += alphaFactor2 * as2Nj1;
					alphatNs2 += alphaFactor3 * as2Nj3;
				
					alphatNs3 += alphaFactor0 * as3Nj0;
					alphatNs3 += alphaFactor1 * as3Nj1;
					alphatNs3 += alphaFactor2 * as3Nj1;
					alphatNs3 += alphaFactor3 * as3Nj3;
				}
				alphatNs0 *= emissionMatrix[yt*hiddenStates + s];
				ctt += alphatNs0;
				alpha[1*hiddenStates + s] = alphatNs0;
				
				alphatNs1 *= emissionMatrix[yt*hiddenStates + s+1];
				ctt += alphatNs1;
				alpha[1*hiddenStates + s+1] = alphatNs1;
			
				alphatNs2 *= emissionMatrix[yt*hiddenStates + s+2];
				ctt += alphatNs2;
				alpha[1*hiddenStates + s+2] = alphatNs2;
				
				alphatNs3 *= emissionMatrix[yt*hiddenStates + s+3];
				ctt += alphatNs3;
				alpha[1*hiddenStates + s+3] = alphatNs3;
			}
			
		
			//scaling factor for t 
			ctt = 1.0 / ctt;
	
			//scale alpha(t)
		        for(int s = 0; s<hiddenStates; s+=unroll_inc){// s=new_state
		    	    alpha[1*hiddenStates + s] *= ctt;
		    	    alpha[1*hiddenStates + s+1] *= ctt;
		    	    alpha[1*hiddenStates + s+2] *= ctt;
		    	    alpha[1*hiddenStates + s+3] *= ctt;
	        	}
			ct[1] = ctt;





			for(int t = 1; t < T-1; t++){
				double ctt = 0.0;	
				const int yt = observations[t];	
				for(int s = 0; s<hiddenStates; s+=unroll_inc){// s=new_state
					double alphatNs0 = 0;
					double alphatNs1 = 0;
					double alphatNs2 = 0;
					double alphatNs3 = 0;
				
					for(int j = 0; j < hiddenStates; j+=unroll_inc){//j=old_states
					
						double alphaFactor0 = alpha[(t-1)*hiddenStates + j];
						double alphaFactor1 = alpha[(t-1)*hiddenStates + j+1];
						double alphaFactor2 = alpha[(t-1)*hiddenStates + j+2]; 
						double alphaFactor3 = alpha[(t-1)*hiddenStates + j+3]; 
					
						alphatNs0 += alphaFactor0 * transitionMatrix[s*hiddenStates + j];
						alphatNs0 += alphaFactor1 * transitionMatrix[s*hiddenStates + j+1];
						alphatNs0 += alphaFactor2 * transitionMatrix[s*hiddenStates + j+2];
						alphatNs0 += alphaFactor3 * transitionMatrix[s*hiddenStates + j+3];
						
						alphatNs1 += alphaFactor0 * transitionMatrix[(s+1)*hiddenStates + j];
						alphatNs1 += alphaFactor1 * transitionMatrix[(s+1)*hiddenStates + j+1];
						alphatNs1 += alphaFactor2 * transitionMatrix[(s+1)*hiddenStates + j+2];
						alphatNs1 += alphaFactor3 * transitionMatrix[(s+1)*hiddenStates + j+3];
						
						alphatNs2 += alphaFactor0 * transitionMatrix[(s+2)*hiddenStates + j];
						alphatNs2 += alphaFactor1 * transitionMatrix[(s+2)*hiddenStates + j+1];
						alphatNs2 += alphaFactor2 * transitionMatrix[(s+2)*hiddenStates + j+2];
						alphatNs2 += alphaFactor3 * transitionMatrix[(s+2)*hiddenStates + j+3];
						
						alphatNs3 += alphaFactor0 * transitionMatrix[(s+3)*hiddenStates + j];
						alphatNs3 += alphaFactor1 * transitionMatrix[(s+3)*hiddenStates + j+1];
						alphatNs3 += alphaFactor2 * transitionMatrix[(s+3)*hiddenStates + j+2];
						alphatNs3 += alphaFactor3 * transitionMatrix[(s+3)*hiddenStates + j+3];
					}
					alphatNs0 *= emissionMatrix[yt*hiddenStates + s];
					ctt += alphatNs0;
						alpha[t*hiddenStates + s] = alphatNs0;
					
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
				for(int s = 0; s<hiddenStates; s+=unroll_inc){// s=new_state
					alpha[t*hiddenStates+s] *= ctt;
					alpha[t*hiddenStates+s+1] *= ctt;
					alpha[t*hiddenStates+s+2] *= ctt;
					alpha[t*hiddenStates+s+3] *= ctt;
				}
				ct[t] = ctt;
			}
			
			//compute alpha(T-1)
			ctt = 0.0;	
			yt = observations[T-1];	
			
			for(int s = 0; s<hiddenStates; s+=unroll_inc){// s=new_state
				double alphatNs0 = 0;
				double alphatNs1 = 0;
				double alphatNs2 = 0;
				double alphatNs3 = 0;
			
				for(int j = 0; j < hiddenStates; j+=unroll_inc){//j=old_states
			
					double alphaFactor0 = alpha[(T-2)*hiddenStates + j];
					double alphaFactor1 = alpha[(T-2)*hiddenStates + j+1];
					double alphaFactor2 = alpha[(T-2)*hiddenStates + j+2]; 
					double alphaFactor3 = alpha[(T-2)*hiddenStates + j+3]; 
			
					alphatNs0 += alphaFactor0 * transitionMatrix[s*hiddenStates + j];
					alphatNs0 += alphaFactor1 * transitionMatrix[s*hiddenStates + j+1];
					alphatNs0 += alphaFactor2 * transitionMatrix[s*hiddenStates + j+2];
					alphatNs0 += alphaFactor3 * transitionMatrix[s*hiddenStates + j+3];
				
					alphatNs1 += alphaFactor0 * transitionMatrix[(s+1)*hiddenStates + j];
					alphatNs1 += alphaFactor1 * transitionMatrix[(s+1)*hiddenStates + j+1];
					alphatNs1 += alphaFactor2 * transitionMatrix[(s+1)*hiddenStates + j+2];
					alphatNs1 += alphaFactor3 * transitionMatrix[(s+1)*hiddenStates + j+3];
				
					alphatNs2 += alphaFactor0 * transitionMatrix[(s+2)*hiddenStates + j];
					alphatNs2 += alphaFactor1 * transitionMatrix[(s+2)*hiddenStates + j+1];
					alphatNs2 += alphaFactor2 * transitionMatrix[(s+2)*hiddenStates + j+2];
					alphatNs2 += alphaFactor3 * transitionMatrix[(s+2)*hiddenStates + j+3];
						
					alphatNs3 += alphaFactor0 * transitionMatrix[(s+3)*hiddenStates + j];
					alphatNs3 += alphaFactor1 * transitionMatrix[(s+3)*hiddenStates + j+1];
					alphatNs3 += alphaFactor2 * transitionMatrix[(s+3)*hiddenStates + j+2];
					alphatNs3 += alphaFactor3 * transitionMatrix[(s+3)*hiddenStates + j+3];
				}
				alphatNs0 *= emissionMatrix[yt*hiddenStates + s];
				ctt += alphatNs0;
				alpha[(T-1)*hiddenStates + s] = alphatNs0;
				
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
		        for(int s = 0; s<hiddenStates; s+=unroll_inc){// s=new_state
					//s
			        double alphaT1Ns0 = alpha[(T-1) * hiddenStates + s]*ctt;
			        alpha[(T-1)*hiddenStates+s] = alphaT1Ns0;
			        gamma_T[s] = alphaT1Ns0 /* *ct[T-1]*/;
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

	
			//Transpose transitionMatrix
			    
			for(int by = 0; by < hiddenStates; by+=block_size){
				const int end = by + block_size;
			
				//Diagonal 4x4 blocks
				double temp01 = transitionMatrix[by*hiddenStates+by+1];
				transitionMatrix[by * hiddenStates + by+1]  = transitionMatrix[(by+1) * hiddenStates + by];
				transitionMatrix[(by+1)*hiddenStates+ by] = temp01;	
				
				double temp02 = transitionMatrix[by*hiddenStates+by+2];
				transitionMatrix[by * hiddenStates + by+2]  = transitionMatrix[(by+2) * hiddenStates + by];
				transitionMatrix[(by+2)*hiddenStates+ by] = temp02;	
				
				double temp03 = transitionMatrix[by*hiddenStates+by+3];
				transitionMatrix[by * hiddenStates + by+3]  = transitionMatrix[(by+3) * hiddenStates + by];
				transitionMatrix[(by+3)*hiddenStates+ by] = temp03;	
				
				double temp12 = transitionMatrix[(by+1)*hiddenStates+by+2];
				transitionMatrix[(by+1) * hiddenStates + by+2]  = transitionMatrix[(by+2)* hiddenStates + (by+1)];
				transitionMatrix[(by+2)*hiddenStates+ (by+1)] = temp12;	
				
				double temp13 = transitionMatrix[(by+1)*hiddenStates+by+3];
				transitionMatrix[(by+1) * hiddenStates + by+3]  = transitionMatrix[(by+3) * hiddenStates + (by+1)];
				transitionMatrix[(by+3)*hiddenStates+ (by+1)] = temp13;	

				double temp23 = transitionMatrix[(by+2)*hiddenStates+by+3];
				transitionMatrix[(by+2) * hiddenStates + by+3]  = transitionMatrix[(by+3) * hiddenStates + (by+2)];
				transitionMatrix[(by+3)*hiddenStates+ (by+2)] = temp23;	
			
				//Offdiagonal blocks
				for(int bx = end; bx < hiddenStates; bx+= block_size){
					const int end_x = block_size + bx;
			
			
					double temp00 = transitionMatrix[by*hiddenStates+bx];
					transitionMatrix[by * hiddenStates + bx]  = transitionMatrix[bx * hiddenStates + by];
					transitionMatrix[bx*hiddenStates+ by] = temp00;	
	
					double temp01 = transitionMatrix[by*hiddenStates+bx + 1];
					transitionMatrix[by * hiddenStates + bx + 1]  = transitionMatrix[(bx + 1) * hiddenStates + by];
					transitionMatrix[(bx + 1)*hiddenStates+ by] = temp01;	
					
					double temp02 = transitionMatrix[by*hiddenStates+bx + 2];
					transitionMatrix[by * hiddenStates + bx + 2]  = transitionMatrix[(bx + 2)  * hiddenStates + by];
					transitionMatrix[(bx + 2) *hiddenStates+ by] = temp02;	
				
					double temp03 = transitionMatrix[by*hiddenStates+bx + 3];
					transitionMatrix[by * hiddenStates + bx + 3]  = transitionMatrix[(bx + 3) * hiddenStates + by];
					transitionMatrix[(bx + 3) *hiddenStates+ by] = temp03;
							
						
					double temp10 = transitionMatrix[(by+1)*hiddenStates+bx];
					transitionMatrix[(by+1) * hiddenStates + bx]  = transitionMatrix[bx * hiddenStates + (by+1)];
					transitionMatrix[bx*hiddenStates+ (by+1)] = temp10;	
			
					double temp11 = transitionMatrix[(by+1)*hiddenStates+bx + 1];
					transitionMatrix[(by+1) * hiddenStates + bx + 1]  = transitionMatrix[(bx + 1) * hiddenStates + (by+1)];
					transitionMatrix[(bx + 1)*hiddenStates+ (by+1)] = temp11;	
					
					double temp12 = transitionMatrix[(by+1)*hiddenStates+bx + 2];
					transitionMatrix[(by+1) * hiddenStates + bx + 2]  = transitionMatrix[(bx + 2)  * hiddenStates + (by+1)];
					transitionMatrix[(bx + 2) *hiddenStates+ (by+1)] = temp12;	
					
					double temp13 = transitionMatrix[(by+1)*hiddenStates+bx + 3];
					transitionMatrix[(by+1) * hiddenStates + bx + 3]  = transitionMatrix[(bx + 3) * hiddenStates + (by+1)];
					transitionMatrix[(bx + 3) *hiddenStates+ (by+1)] = temp13;
						
						
					double temp20 = transitionMatrix[(by+2)*hiddenStates+bx];
					transitionMatrix[(by+2)* hiddenStates + bx]  = transitionMatrix[bx * hiddenStates + (by+2)];
					transitionMatrix[bx*hiddenStates+ (by+2)] = temp20;	
					
					double temp21 = transitionMatrix[(by+2)*hiddenStates+bx + 1];
					transitionMatrix[(by+2) * hiddenStates + bx + 1]  = transitionMatrix[(bx + 1) * hiddenStates + (by+2)];
					transitionMatrix[(bx + 1)*hiddenStates+(by+2)] = temp21;	
					
					double temp22 = transitionMatrix[(by+2)*hiddenStates+bx + 2];
					transitionMatrix[(by+2) * hiddenStates + bx + 2]  = transitionMatrix[(bx + 2)  * hiddenStates + (by+2)];
					transitionMatrix[(bx + 2) *hiddenStates+ (by+2)] = temp22;	
					
					double temp23 = transitionMatrix[(by+2)*hiddenStates+bx + 3];
					transitionMatrix[(by+2)* hiddenStates + bx + 3]  = transitionMatrix[(bx + 3) * hiddenStates + (by+2)];
					transitionMatrix[(bx + 3) *hiddenStates+ (by+2)] = temp23;
					
						
					double temp30 = transitionMatrix[(by+3)*hiddenStates+bx];
					transitionMatrix[(by+3) * hiddenStates + bx]  = transitionMatrix[bx * hiddenStates +(by+3)];
					transitionMatrix[bx*hiddenStates+ (by+3)] = temp30;	
					
					double temp31 = transitionMatrix[(by+3)*hiddenStates+bx + 1];
					transitionMatrix[(by+3) * hiddenStates + bx + 1]  = transitionMatrix[(bx + 1) * hiddenStates + (by+3)];
					transitionMatrix[(bx + 1)*hiddenStates+ (by+3)] = temp31;	
					
					double temp32 = transitionMatrix[(by+3)*hiddenStates+bx + 2];
					transitionMatrix[(by+3) * hiddenStates + bx + 2]  = transitionMatrix[(bx + 2)  * hiddenStates + (by+3)];
					transitionMatrix[(bx + 2) *hiddenStates+ (by+3)] = temp32;	
					
					double temp33 = transitionMatrix[(by+3)*hiddenStates+bx + 3];
					transitionMatrix[(by+3) * hiddenStates + bx + 3]  = transitionMatrix[(bx + 3) * hiddenStates + (by+3)];
					transitionMatrix[(bx + 3) *hiddenStates+ (by+3)] = temp33;
						
			
				}	
			}
	
	
	
			for(int v = 0; v < differentObservables; v++){
				for(int s = 0; s < hiddenStates; s+=unroll_inc){
					for(int j = 0; j < hiddenStates; j+=unroll_inc){
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
			
			for(int s = 0; s < hiddenStates; s+=unroll_inc){
			        beta[s] = /* 1* */ctt;
			        beta[s+1] = /* 1* */ctt;
			        beta[s+2] = /* 1* */ctt;
			        beta[s+3] = /* 1* */ctt;
	       	 }
	        
	       	for(int s = 0; s < hiddenStates; s+=unroll_inc){
			        gamma_sum[s] = 0.0;
			        gamma_sum[s+1] = 0.0;
			        gamma_sum[s+2] = 0.0;
			        gamma_sum[s+3] = 0.0;
	        	}

		
   			yt = observations[T-1];
			for(int t = T-1; t > 0; t--){
				const int yt1 = observations[t-1];
				const double ctt = ct[t-1];
				for(int s = 0; s < hiddenStates ; s+=unroll_inc){
					double beta_news0 = 0.0;
					double alphat1Ns0 = alpha[(t-1)*hiddenStates + s];
					double beta_news1 = 0.0;
					double alphat1Ns1 = alpha[(t-1)*hiddenStates + s+1];
					double beta_news2 = 0.0;
					double alphat1Ns2 = alpha[(t-1)*hiddenStates + s+2];
					double beta_news3 = 0.0;
					double alphat1Ns3 = alpha[(t-1)*hiddenStates + s+3];
					for(int j = 0; j < hiddenStates; j+=unroll_inc){
						
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
    
		
		
		
		
		
		
		
		
		
		for(int s = 0; s < hiddenStates; s+=unroll_inc){
			double gamma_sums_inv0 = 1./gamma_sum[s];
			double gamma_sums_inv1 = 1./gamma_sum[s+1];
			double gamma_sums_inv2 = 1./gamma_sum[s+2];
			double gamma_sums_inv3 = 1./gamma_sum[s+3];
			for(int j = 0; j < hiddenStates; j+=unroll_inc){
				transitionMatrix[s*hiddenStates+j] = a_new[s*hiddenStates+j]*gamma_sums_inv0;
				transitionMatrix[s*hiddenStates+j+1] = a_new[s*hiddenStates+j+1]*gamma_sums_inv0;
				transitionMatrix[s*hiddenStates+j+2] = a_new[s*hiddenStates+j+2]*gamma_sums_inv0;
				transitionMatrix[s*hiddenStates+j+3] = a_new[s*hiddenStates+j+3]*gamma_sums_inv0;
	
				transitionMatrix[(s+1)*hiddenStates+j] = a_new[(s+1)*hiddenStates+j]*gamma_sums_inv1;
				transitionMatrix[(s+1)*hiddenStates+j+1] = a_new[(s+1)*hiddenStates+j+1]*gamma_sums_inv1;
				transitionMatrix[(s+1)*hiddenStates+j+2] = a_new[(s+1)*hiddenStates+j+2]*gamma_sums_inv1;
				transitionMatrix[(s+1)*hiddenStates+j+3] = a_new[(s+1)*hiddenStates+j+3]*gamma_sums_inv1;
	
				transitionMatrix[(s+2)*hiddenStates+j] = a_new[(s+2)*hiddenStates+j]*gamma_sums_inv2;
				transitionMatrix[(s+2)*hiddenStates+j+1] = a_new[(s+2)*hiddenStates+j+1]*gamma_sums_inv2;
				transitionMatrix[(s+2)*hiddenStates+j+2] = a_new[(s+2)*hiddenStates+j+2]*gamma_sums_inv2;
				transitionMatrix[(s+2)*hiddenStates+j+3] = a_new[(s+2)*hiddenStates+j+3]*gamma_sums_inv2;
	
				transitionMatrix[(s+3)*hiddenStates+j] = a_new[(s+3)*hiddenStates+j]*gamma_sums_inv3;
				transitionMatrix[(s+3)*hiddenStates+j+1] = a_new[(s+3)*hiddenStates+j+1]*gamma_sums_inv3;
				transitionMatrix[(s+3)*hiddenStates+j+2] = a_new[(s+3)*hiddenStates+j+2]*gamma_sums_inv3;
				transitionMatrix[(s+3)*hiddenStates+j+3] = a_new[(s+3)*hiddenStates+j+3]*gamma_sums_inv3;
	
			}
		}


		
		yt = observations[T-1];
		//add remaining parts of the sum of gamma 
		for(int s = 0; s < hiddenStates; s+=unroll_inc){
			double gamma_Ts0 = gamma_T[s];
			//if you use real gamma you have to divide by ct[t-1]
			double gamma_tot0 = gamma_Ts0 + gamma_sum[s] /* /ct[T-1] */;
			gamma_T[s] = 1./gamma_tot0;
	        	b_new[yt*hiddenStates+s]+=gamma_Ts0;
   
			double gamma_Ts1 = gamma_T[s+1];
			//if you use real gamma you have to divide by ct[t-1]
			double gamma_tot1 = gamma_Ts1 + gamma_sum[s+1] /* /ct[T-1] */;
			gamma_T[s+1] = 1./gamma_tot1;
	        	b_new[yt*hiddenStates+s+1]+=gamma_Ts1;
	   
			double gamma_Ts2 = gamma_T[s+2];
			//if you use real gamma you have to divide by ct[t-1]
			double gamma_tot2 = gamma_Ts2 + gamma_sum[s+2] /* /ct[T-1] */;
				gamma_T[s+2] = 1./gamma_tot2;
	        	b_new[yt*hiddenStates+s+2]+=gamma_Ts2;
   
			double gamma_Ts3 = gamma_T[s+3];
			//if you use real gamma you have to divide by ct[t-1]
			double gamma_tot3 = gamma_Ts3 + gamma_sum[s+3] /* /ct[T-1] */;
			gamma_T[s+3] = 1./gamma_tot3;
	        	b_new[yt*hiddenStates+s+3]+=gamma_Ts3;
   
		}
		
		//compute new emission matrix
		for(int v = 0; v < differentObservables; v+=unroll_inc){
			for(int s = 0; s < hiddenStates; s+=unroll_inc){
			
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
	free(beta);
	free(beta_new);
	free(alpha);
	free(ab);
			

	return 0; 
} 
