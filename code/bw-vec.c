#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <float.h>

#include "tsc_x86.h"
#include "io.h"
#include "tested.h"
#include "util.h"
#include <immintrin.h>

double EPSILON = 1e-4;
#define DELTA 1e-2
#define BUFSIZE 1<<26

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
	int y0 = y[0];

	//compute alpha(0)
	for(int s = 0; s < N; s++){
		double alphas = p[s] * b[y0*N + s];
		ct0 += alphas;
		alpha[s] = alphas;
	}
	
	ct0 = 1.0 / ct0;

	for(int s = 0; s < N; s++){
		alpha[s] *= ct0;
	}

	ct[0] = ct0;

	//compute alpha(t)
	for(int t = 1; t < T-1; t++){
		double ctt = 0.0;	
		const int yt = y[t];	

		for(int s = 0; s<N; s++){
			double alphatNs = 0;

			for(int j = 0; j < N; j++){
				alphatNs += alpha[(t-1)*N + j] * a[s*N + j];
			}

			alphatNs *= b[yt*N + s];
			ctt += alphatNs;
			alpha[t*N + s] = alphatNs;
		}
 
		ctt = 1.0 / ctt;
		
		for(int s = 0; s<N; s++){
			alpha[t*N+s] *= ctt;
		}

		ct[t] = ctt;
	}
		
	double ctt = 0.0;	
	int yt = y[T-1];
	
	for(int s = 0; s<N; s++){
		double alphatNs = 0;
		for(int j = 0; j < N; j++){
			alphatNs += alpha[(T-2)*N + j] * a[s*N + j];
		}

		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[(T-1)*N + s] = alphatNs;
	}

	ctt = 1.0 / ctt;
		
	for(int s = 0; s<N; s++){
		double alphaT1Ns = alpha[(T-1) * N + s]*ctt;
		alpha[(T-1)*N+s] = alphaT1Ns;
		gamma_T[s] = alphaT1Ns;
	}

	ct[T-1] = ctt;

	//FUSED BACKWARD and UPDATE STEP

	for(int s = 0; s < N; s++){
		beta[s] = ctt;
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

			double ps =alphat1Ns*beta_news;  
			p[s] = ps;
			beta_new[s] = beta_news*ctt;
			gamma_sum[s]+= ps;
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


void baum_welch(double* const a, double* const b, double* const p, const int* const y, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int N, const int K, const int T){

	double* beta = (double*) malloc(N  * sizeof(double));
	double* beta_new = (double*) malloc(N * sizeof(double));
	double* alpha = (double*) malloc(N * T * sizeof(double));
	double* ab = (double*) malloc(N * N * K * sizeof(double));

	int yt = y[T-1];

	//add remaining parts of the sum of gamma 
	for(int s = 0; s < N; s++){
		double gamma_Ts = gamma_T[s];
		double gamma_sums = gamma_sum[s];
		double gamma_tot = gamma_Ts + gamma_sums;
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

	//Transpose a_new

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
	int y0 = y[0];

	//compute alpha(0)
	for(int s = 0; s < N; s++){
		double alphas = p[s] * b[y0*N + s];
		ctt += alphas;
		alpha[s] = alphas;
	}
	
	ctt = 1.0 / ctt;

	for(int s = 0; s < N; s++){
		alpha[s] *= ctt;
	}

	ct[0] = ctt;
	ctt = 0.0;	
	yt = y[1];	

	//Compute alpha(1) and scale transitionMatrix
	for(int s = 0; s<N-1; s++){
		double alphatNs = 0;

		for(int j = 0; j < N; j++){
			double asNj =  a_new[s*N + j] * gamma_sum[j];
			a_new[s*N+j] = 0.0;
			a[s*N + j] = asNj;
			alphatNs += alpha[0*N + j] * asNj;
		}

		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[1*N + s] = alphatNs;
	}
	
	double alphatNs = 0;
	for(int j = 0; j < N; j++){
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
	ctt = 1.0 / ctt;
	
	for(int s = 0; s<N; s++){
		alpha[1*N+s] *= ctt;
	}

	ct[1] = ctt;

	//compute alpha(t)
	for(int t = 2; t < T-1; t++){
		ctt = 0.0;	
		yt = y[t];
	
		for(int s = 0; s<N; s++){
			double alphatNs = 0;

			for(int j = 0; j < N; j++){
				alphatNs += alpha[(t-1)*N + j] *a[s*N + j];
			}

			alphatNs *= b[yt*N + s];
			ctt += alphatNs;
			alpha[t*N + s] = alphatNs;
		}

		ctt = 1.0 / ctt;
		
		for(int s = 0; s<N; s++){
			alpha[t*N+s] *= ctt;
		}

		ct[t] = ctt;
	}
		
	ctt = 0.0;	
	yt = y[T-1];	

	//compute alpha(T-1)
	for(int s = 0; s<N; s++){
		double alphatNs = 0;

		for(int j = 0; j < N; j++){
			alphatNs += alpha[(T-2)*N + j] * a[s*N + j];
		}

		alphatNs *= b[yt*N + s];
		ctt += alphatNs;
		alpha[(T-1)*N + s] = alphatNs;
	}

	ctt = 1.0 / ctt;
		
	for(int s = 0; s<N; s++){
		double alphaT1Ns = alpha[(T-1) * N + s]*ctt;
		alpha[(T-1)*N+s] = alphaT1Ns;
		gamma_T[s] = alphaT1Ns;
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
		beta[s] = ctt;
	}
	
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

			double ps =alphat1Ns*beta_news;  
			p[s] = ps;
			beta_new[s] = beta_news*ctt;
			gamma_sum[s]+= ps;
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
		double gamma_tot = gamma_Ts + gamma_sum[s];
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

void heatup(double* const transitionMatrix,double* const stateProb,double* const emissionMatrix,const int* const observations,const int hiddenStates,const int differentObservables,const int T){

	double* ct = (double*) malloc( T* sizeof(double));
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

	myInt64 cycles;
   	myInt64 start;
    	int minima=10;
    	int variableSteps=100-cbrt(hiddenStates*differentObservables*T)/3;
    	int maxSteps=minima < variableSteps ? variableSteps : minima;
    	minima=1;    
    	variableSteps=10-log10(hiddenStates*differentObservables*T);
    	int maxRuns=minima < variableSteps ? variableSteps : minima;
	double runs[maxRuns]; 

	srand(seed);

	//ground truth
	double* groundTransitionMatrix = (double*) _mm_malloc(hiddenStates*hiddenStates*sizeof(double),32);
	double* groundEmissionMatrix = (double*) _mm_malloc(hiddenStates*differentObservables*sizeof(double),32);
	makeMatrix(hiddenStates, hiddenStates, groundTransitionMatrix);
	makeMatrix(hiddenStates, differentObservables, groundEmissionMatrix);
	int groundInitialState = rand()%hiddenStates;
	int* observations = (int*) _mm_malloc ( T * sizeof(int),32);
	makeObservations(hiddenStates, differentObservables, groundInitialState, groundTransitionMatrix,groundEmissionMatrix,T, observations);
	
	double* transitionMatrix = (double*) _mm_malloc(hiddenStates*hiddenStates*sizeof(double),32);
	double* transitionMatrixSafe = (double*) _mm_malloc(hiddenStates*hiddenStates*sizeof(double),32);
	double* transitionMatrixTesting=(double*) _mm_malloc(hiddenStates*hiddenStates*sizeof(double),32);

	double* emissionMatrix = (double*) _mm_malloc(hiddenStates*differentObservables*sizeof(double),32);
	double* emissionMatrixSafe = (double*) _mm_malloc(hiddenStates*differentObservables*sizeof(double),32);
	double* emissionMatrixTesting=(double*) _mm_malloc(hiddenStates*differentObservables*sizeof(double),32);

	double* stateProb  = (double*) _mm_malloc(hiddenStates * sizeof(double),32);
	double* stateProbSafe  = (double*) _mm_malloc(hiddenStates * sizeof(double),32);
	double* stateProbTesting  = (double*) _mm_malloc(hiddenStates * sizeof(double),32);

	double* gamma_T = (double*) _mm_malloc( hiddenStates * sizeof(double),32);
	double* gamma_sum = (double*) _mm_malloc( hiddenStates * sizeof(double),32);
	double* a_new = (double*) _mm_malloc(hiddenStates * hiddenStates * sizeof(double),32);
	double* b_new = (double*) _mm_malloc(differentObservables*hiddenStates * sizeof(double),32);
	double* ct = (double*) _mm_malloc((T+4)*sizeof(double),32);
	double* beta = (double*) _mm_malloc(hiddenStates  * sizeof(double),32);
	double* beta_new = (double*) _mm_malloc(hiddenStates * sizeof(double),32);
	double* alpha = (double*) _mm_malloc(hiddenStates * T * sizeof(double),32);
	double* ab = (double*) _mm_malloc(hiddenStates * hiddenStates * differentObservables * sizeof(double),32);
	double* reduction = (double*) _mm_malloc(4  * sizeof(double),32);
	
	//random init transition matrix, emission matrix and state probabilities.
	makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
	makeMatrix(hiddenStates, differentObservables, emissionMatrix);
	makeProbabilities(stateProb,hiddenStates);

	transpose(emissionMatrix, hiddenStates, differentObservables);

	//copy for resetting to initial state.
	memcpy(transitionMatrixSafe, transitionMatrix, hiddenStates*hiddenStates*sizeof(double));
   	memcpy(emissionMatrixSafe, emissionMatrix, hiddenStates*differentObservables*sizeof(double));
    	memcpy(stateProbSafe, stateProb, hiddenStates * sizeof(double));

	//heat up cache
	//heatup(transitionMatrix,stateProb,emissionMatrix,observations,hiddenStates,differentObservables,T);
	
	//matrix for flushing cache
	volatile unsigned char* buf = malloc(BUFSIZE*sizeof(char));
	
   	double disparance;
   	int steps = 0;

	for (int run=0; run<maxRuns; run++){

		//reset to init
		memcpy(transitionMatrix, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
   		memcpy(emissionMatrix, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
        	memcpy(stateProb, stateProbSafe, hiddenStates * sizeof(double));
		
        	double logLikelihood=-DBL_MAX;

		//only needed for testing with R
		//write_init(transitionMatrix, emissionMatrix, observations, stateProb, hiddenStates, differentObservables, T);
        
        	steps=1;
       	
		_flush_cache(buf,BUFSIZE);
		start = start_tsc();
        
        	__m256d one = _mm256_set1_pd(1.0);
		
		//Tranpose transition matrix              
		for(int by = 0; by < hiddenStates; by+=4){
	
			//Diagonal block
			__m256d diag0 = _mm256_load_pd(transitionMatrix + by*hiddenStates + by);
			__m256d diag1 = _mm256_load_pd(transitionMatrix + (by+1)*hiddenStates + by);
			__m256d diag2 = _mm256_load_pd(transitionMatrix + (by+2)*hiddenStates + by);
			__m256d diag3 = _mm256_load_pd(transitionMatrix + (by+3)*hiddenStates + by);
		
			__m256d tmp0 = _mm256_shuffle_pd(diag0,diag1, 0x0);
			__m256d tmp1 = _mm256_shuffle_pd(diag2,diag3, 0x0);
			__m256d tmp2 = _mm256_shuffle_pd(diag0,diag1, 0xF);
			__m256d tmp3 = _mm256_shuffle_pd(diag2,diag3, 0xF);
                    	
			__m256d row0 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
			__m256d row1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
			__m256d row2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
			__m256d row3 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);
			
			_mm256_store_pd(transitionMatrix + by*hiddenStates + by,row0);
			_mm256_store_pd(transitionMatrix + (by+1)*hiddenStates + by,row1);
			_mm256_store_pd(transitionMatrix + (by+2)*hiddenStates + by,row2);
			_mm256_store_pd(transitionMatrix + (by+3)*hiddenStates + by,row3);
	
			//Offdiagonal blocks
			for(int bx = by + 4; bx < hiddenStates; bx+= 4){
								
				__m256d upper0 = _mm256_load_pd(transitionMatrix + by*hiddenStates + bx);
				__m256d upper1 = _mm256_load_pd(transitionMatrix + (by+1)*hiddenStates + bx);
				__m256d upper2 = _mm256_load_pd(transitionMatrix + (by+2)*hiddenStates + bx);
				__m256d upper3 = _mm256_load_pd(transitionMatrix + (by+3)*hiddenStates + bx);
									
				__m256d lower0 = _mm256_load_pd(transitionMatrix + bx * hiddenStates + by);
				__m256d lower1 = _mm256_load_pd(transitionMatrix + (bx+1)*hiddenStates + by);
				__m256d lower2 = _mm256_load_pd(transitionMatrix + (bx+2)*hiddenStates + by);
				__m256d lower3 = _mm256_load_pd(transitionMatrix + (bx+3)*hiddenStates + by);
				
				__m256d utmp0 = _mm256_shuffle_pd(upper0,upper1, 0x0);
				__m256d utmp1 = _mm256_shuffle_pd(upper2,upper3, 0x0);
				__m256d utmp2 = _mm256_shuffle_pd(upper0,upper1, 0xF);
				__m256d utmp3 = _mm256_shuffle_pd(upper2,upper3, 0xF);
					
				__m256d ltmp0 = _mm256_shuffle_pd(lower0,lower1, 0x0);
				__m256d ltmp1 = _mm256_shuffle_pd(lower2,lower3, 0x0);
				__m256d ltmp2 = _mm256_shuffle_pd(lower0,lower1, 0xF);
				__m256d ltmp3 = _mm256_shuffle_pd(lower2,lower3, 0xF);
        				            
				__m256d urow0 = _mm256_permute2f128_pd(utmp0, utmp1, 0x20);
				__m256d urow1 = _mm256_permute2f128_pd(utmp2, utmp3, 0x20);
				__m256d urow2 = _mm256_permute2f128_pd(utmp0, utmp1, 0x31);
				__m256d urow3 = _mm256_permute2f128_pd(utmp2, utmp3, 0x31);
	        			            
				__m256d lrow0 = _mm256_permute2f128_pd(ltmp0, ltmp1, 0x20);
				__m256d lrow1 = _mm256_permute2f128_pd(ltmp2, ltmp3, 0x20);
				__m256d lrow2 = _mm256_permute2f128_pd(ltmp0, ltmp1, 0x31);
				__m256d lrow3 = _mm256_permute2f128_pd(ltmp2, ltmp3, 0x31);
						
				_mm256_store_pd(transitionMatrix + by*hiddenStates + bx,lrow0);
				_mm256_store_pd(transitionMatrix + (by+1)*hiddenStates + bx,lrow1);
				_mm256_store_pd(transitionMatrix + (by+2)*hiddenStates + bx,lrow2);
				_mm256_store_pd(transitionMatrix + (by+3)*hiddenStates + bx,lrow3);
					
				_mm256_store_pd(transitionMatrix + bx*hiddenStates + by,urow0);
				_mm256_store_pd(transitionMatrix + (bx+1)*hiddenStates + by,urow1);
				_mm256_store_pd(transitionMatrix + (bx+2)*hiddenStates + by,urow2);
				_mm256_store_pd(transitionMatrix + (bx+3)*hiddenStates + by,urow3);	
			}	
		}

		int y0 = observations[0];
		__m256d ct0_vec = _mm256_setzero_pd();

		//compute alpha(0)
		for(int s = 0; s < hiddenStates; s+=4){

			__m256d stateProb_vec = _mm256_load_pd(stateProb +s);
			__m256d emission_vec = _mm256_load_pd(emissionMatrix +y0*hiddenStates +s);
			__m256d alphas_vec = _mm256_mul_pd(stateProb_vec, emission_vec);
			ct0_vec = _mm256_fmadd_pd(stateProb_vec,emission_vec, ct0_vec);
			_mm256_store_pd(alpha+s,alphas_vec);

        	}	
	        	
	        //Reduction of ct_vec
	        __m256d perm = _mm256_permute2f128_pd(ct0_vec,ct0_vec,0b00000011);
	
		__m256d shuffle1 = _mm256_shuffle_pd(ct0_vec, perm, 0b0101);
		__m256d shuffle2 = _mm256_shuffle_pd(perm, ct0_vec, 0b0101);
		
		__m256d ct0_vec_add = _mm256_add_pd(ct0_vec, perm);
		__m256d ct0_temp = _mm256_add_pd(shuffle1, shuffle2);
		__m256d ct0_vec_tot = _mm256_add_pd(ct0_vec_add, ct0_temp);
		__m256d ct0_vec_div = _mm256_div_pd(one ,ct0_vec_tot);
			
	      	_mm256_storeu_pd(ct,ct0_vec_div);
	        
	        for(int s = 0; s < hiddenStates; s+=4){
			__m256d alphas=_mm256_load_pd(alpha+s);
			__m256d alphas_mul=_mm256_mul_pd(alphas,ct0_vec_div);
			_mm256_store_pd(alpha+s,alphas_mul);
	        }

		for(int t = 1; t < T-1; t++){	
			__m256d ctt_vec = _mm256_setzero_pd();
			const int yt = observations[t];
	
			for(int s = 0; s<hiddenStates; s+=4){

				__m256d alphatNs0 = _mm256_setzero_pd();
				__m256d alphatNs1 = _mm256_setzero_pd();
				__m256d alphatNs2 = _mm256_setzero_pd();
				__m256d alphatNs3 = _mm256_setzero_pd();
			
				for(int j = 0; j < hiddenStates; j+=4){
					__m256d alphaFactor=_mm256_load_pd(alpha+(t-1)*hiddenStates+j);
				
					__m256d transition0=_mm256_load_pd(transitionMatrix+(s)*hiddenStates+j);
					__m256d transition1=_mm256_load_pd(transitionMatrix+(s+1)*hiddenStates+j);
					__m256d transition2=_mm256_load_pd(transitionMatrix+(s+2)*hiddenStates+j);
					__m256d transition3=_mm256_load_pd(transitionMatrix+(s+3)*hiddenStates+j);
			
					alphatNs0 =_mm256_fmadd_pd(alphaFactor,transition0,alphatNs0);
					alphatNs1 =_mm256_fmadd_pd(alphaFactor,transition1,alphatNs1);
					alphatNs2 =_mm256_fmadd_pd(alphaFactor,transition2,alphatNs2);
					alphatNs3 =_mm256_fmadd_pd(alphaFactor,transition3,alphatNs3);
				}
								
				__m256d emission = _mm256_load_pd(emissionMatrix + yt*hiddenStates + s);
			
				__m256d alpha01 = _mm256_hadd_pd(alphatNs0, alphatNs1);
				__m256d alpha23 = _mm256_hadd_pd(alphatNs2, alphatNs3);
								
				__m256d permute01 = _mm256_permute2f128_pd(alpha01, alpha23, 0b00110000);
				__m256d permute23 = _mm256_permute2f128_pd(alpha01, alpha23, 0b00100001);
			
				__m256d alpha_tot = _mm256_add_pd(permute01, permute23);

				__m256d alpha_tot_mul = _mm256_mul_pd(alpha_tot,emission);

				ctt_vec = _mm256_add_pd(alpha_tot_mul,ctt_vec);
					
				_mm256_store_pd(alpha + t*hiddenStates + s,alpha_tot_mul);
			}
			
			__m256d perm = _mm256_permute2f128_pd(ctt_vec,ctt_vec,0b00000011);

			__m256d shuffle1 = _mm256_shuffle_pd(ctt_vec, perm, 0b0101);
			__m256d shuffle2 = _mm256_shuffle_pd(perm, ctt_vec, 0b0101);
		
			__m256d ctt_vec_add = _mm256_add_pd(ctt_vec, perm);
			__m256d ctt_temp = _mm256_add_pd(shuffle1, shuffle2);
			__m256d ctt_vec_tot = _mm256_add_pd(ctt_vec_add, ctt_temp);
			__m256d ctt_vec_div = _mm256_div_pd(one,ctt_vec_tot);
		
      			_mm256_storeu_pd(ct + t,ctt_vec_div); 

			for(int s = 0; s<hiddenStates; s+=4){
				__m256d alphas=_mm256_load_pd(alpha+t*hiddenStates+s);
				__m256d alphas_mul=_mm256_mul_pd(alphas,ctt_vec_div);
				_mm256_store_pd(alpha+t*hiddenStates+s,alphas_mul);
			}
		}

		int yt = observations[T-1];	
		__m256d ctt_vec = _mm256_setzero_pd();

		for(int s = 0; s<hiddenStates; s+=4){

			__m256d alphatNs0_vec = _mm256_setzero_pd();
			__m256d alphatNs1_vec = _mm256_setzero_pd();
			__m256d alphatNs2_vec = _mm256_setzero_pd();
			__m256d alphatNs3_vec = _mm256_setzero_pd();
			 
			for(int j = 0; j < hiddenStates; j+=4){
				__m256d alphaFactor=_mm256_load_pd(alpha+(T-2)*hiddenStates+j);
					
				__m256d transition0=_mm256_load_pd(transitionMatrix+(s)*hiddenStates+j);
				__m256d transition1=_mm256_load_pd(transitionMatrix+(s+1)*hiddenStates+j);
				__m256d transition2=_mm256_load_pd(transitionMatrix+(s+2)*hiddenStates+j);
				__m256d transition3=_mm256_load_pd(transitionMatrix+(s+3)*hiddenStates+j);
					
				alphatNs0_vec =_mm256_fmadd_pd(alphaFactor,transition0,alphatNs0_vec);
				alphatNs1_vec =_mm256_fmadd_pd(alphaFactor,transition1,alphatNs1_vec);
				alphatNs2_vec =_mm256_fmadd_pd(alphaFactor,transition2,alphatNs2_vec);
				alphatNs3_vec =_mm256_fmadd_pd(alphaFactor,transition3,alphatNs3_vec);
			}
			
			__m256d emission = _mm256_load_pd(emissionMatrix + yt*hiddenStates + s);
				
			__m256d alpha01 = _mm256_hadd_pd(alphatNs0_vec, alphatNs1_vec);
			__m256d alpha23 = _mm256_hadd_pd(alphatNs2_vec, alphatNs3_vec);
						
			__m256d permute01 = _mm256_permute2f128_pd(alpha01, alpha23, 0b00110000);
			__m256d permute23 = _mm256_permute2f128_pd(alpha01, alpha23, 0b00100001);
								
			__m256d alpha_tot = _mm256_add_pd(permute01, permute23);
			__m256d alpha_tot_mul = _mm256_mul_pd(alpha_tot,emission);
			ctt_vec = _mm256_add_pd(alpha_tot_mul,ctt_vec);
					
			_mm256_store_pd(alpha + (T-1)*hiddenStates + s,alpha_tot_mul);
		}
			
		__m256d perm1 = _mm256_permute2f128_pd(ctt_vec,ctt_vec,0b00000011);
	
		__m256d shuffle11 = _mm256_shuffle_pd(ctt_vec, perm1, 0b0101);
		__m256d shuffle21 = _mm256_shuffle_pd(perm1, ctt_vec, 0b0101);
		
		__m256d ctt_vec_add = _mm256_add_pd(ctt_vec, perm1);
		__m256d ctt_temp = _mm256_add_pd(shuffle11, shuffle21);
		__m256d ctt_vec_tot = _mm256_add_pd(ctt_vec_add, ctt_temp);
		__m256d ctt_vec_div = _mm256_div_pd(one,ctt_vec_tot);
	
      		_mm256_storeu_pd(ct + (T-1),ctt_vec_div); 

		for(int s = 0; s<hiddenStates; s+=4){
			__m256d alphaT1Ns=_mm256_load_pd(alpha+(T-1)*hiddenStates+s);
			__m256d alphaT1Ns_mul=_mm256_mul_pd(alphaT1Ns,ctt_vec_div);
			_mm256_store_pd(alpha+(T-1)*hiddenStates+s,alphaT1Ns_mul);
			_mm256_store_pd(gamma_T+s,alphaT1Ns_mul);
		}

		//FUSED BACKWARD and UPDATE STEP

		__m256d zero = _mm256_setzero_pd();

		for(int s = 0; s < hiddenStates; s+=4){
			_mm256_store_pd(beta + s, ctt_vec_div);
		}
		
		for(int s = 0; s < hiddenStates; s+=4){
			_mm256_store_pd(gamma_sum+ s, zero);
		}
				
		for(int s = 0; s < hiddenStates; s+=4){
			for(int j = 0; j < hiddenStates; j+=4){
				_mm256_store_pd(a_new + s * hiddenStates + j, zero);
				_mm256_store_pd(a_new + (s + 1)* hiddenStates + j, zero);
				_mm256_store_pd(a_new + (s + 2)*hiddenStates + j, zero);
				_mm256_store_pd(a_new + (s + 3)* hiddenStates + j, zero);
			}
		}

		for(int v = 0;  v < differentObservables; v+=4){
			for(int s = 0; s < hiddenStates; s+=4){
				_mm256_store_pd(b_new + v * hiddenStates + s, zero);
				_mm256_store_pd(b_new + (v + 1)* hiddenStates + s, zero);
				_mm256_store_pd(b_new + (v + 2)*hiddenStates + s, zero);
				_mm256_store_pd(b_new + (v + 3)* hiddenStates + s, zero);
			}
		}

		//Transpose transitionMatrix
		for(int by = 0; by < hiddenStates; by+=4){
			
			//Diagonal block
			__m256d diag0 = _mm256_load_pd(transitionMatrix + by*hiddenStates + by);
			__m256d diag1 = _mm256_load_pd(transitionMatrix + (by+1)*hiddenStates + by);
			__m256d diag2 = _mm256_load_pd(transitionMatrix + (by+2)*hiddenStates + by);
			__m256d diag3 = _mm256_load_pd(transitionMatrix + (by+3)*hiddenStates + by);
	
			__m256d tmp0 = _mm256_shuffle_pd(diag0,diag1, 0x0);
			__m256d tmp1 = _mm256_shuffle_pd(diag2,diag3, 0x0);
			__m256d tmp2 = _mm256_shuffle_pd(diag0,diag1, 0xF);
			__m256d tmp3 = _mm256_shuffle_pd(diag2,diag3, 0xF);
                    
			__m256d row0 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
			__m256d row1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
			__m256d row2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
			__m256d row3 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);
			
			_mm256_store_pd(transitionMatrix + by*hiddenStates + by,row0);
			_mm256_store_pd(transitionMatrix + (by+1)*hiddenStates + by,row1);
			_mm256_store_pd(transitionMatrix + (by+2)*hiddenStates + by,row2);
			_mm256_store_pd(transitionMatrix + (by+3)*hiddenStates + by,row3);
	
			//Offdiagonal blocks
			for(int bx = by + 4; bx < hiddenStates; bx+= 4){
										
				__m256d upper0 = _mm256_load_pd(transitionMatrix + by*hiddenStates + bx);
				__m256d upper1 = _mm256_load_pd(transitionMatrix + (by+1)*hiddenStates + bx);
				__m256d upper2 = _mm256_load_pd(transitionMatrix + (by+2)*hiddenStates + bx);
				__m256d upper3 = _mm256_load_pd(transitionMatrix + (by+3)*hiddenStates + bx);
				
				__m256d lower0 = _mm256_load_pd(transitionMatrix + bx * hiddenStates + by);
				__m256d lower1 = _mm256_load_pd(transitionMatrix + (bx+1)*hiddenStates + by);
				__m256d lower2 = _mm256_load_pd(transitionMatrix + (bx+2)*hiddenStates + by);
				__m256d lower3 = _mm256_load_pd(transitionMatrix + (bx+3)*hiddenStates + by);
			
				__m256d utmp0 = _mm256_shuffle_pd(upper0,upper1, 0x0);
				__m256d utmp1 = _mm256_shuffle_pd(upper2,upper3, 0x0);
				__m256d utmp2 = _mm256_shuffle_pd(upper0,upper1, 0xF);
				__m256d utmp3 = _mm256_shuffle_pd(upper2,upper3, 0xF);
        			            
				__m256d ltmp0 = _mm256_shuffle_pd(lower0,lower1, 0x0);
				__m256d ltmp1 = _mm256_shuffle_pd(lower2,lower3, 0x0);
				__m256d ltmp2 = _mm256_shuffle_pd(lower0,lower1, 0xF);
				__m256d ltmp3 = _mm256_shuffle_pd(lower2,lower3, 0xF);
				
				__m256d urow0 = _mm256_permute2f128_pd(utmp0, utmp1, 0x20);
				__m256d urow1 = _mm256_permute2f128_pd(utmp2, utmp3, 0x20);
				__m256d urow2 = _mm256_permute2f128_pd(utmp0, utmp1, 0x31);
				__m256d urow3 = _mm256_permute2f128_pd(utmp2, utmp3, 0x31);
    
				__m256d lrow0 = _mm256_permute2f128_pd(ltmp0, ltmp1, 0x20);
				__m256d lrow1 = _mm256_permute2f128_pd(ltmp2, ltmp3, 0x20);
				__m256d lrow2 = _mm256_permute2f128_pd(ltmp0, ltmp1, 0x31);
				__m256d lrow3 = _mm256_permute2f128_pd(ltmp2, ltmp3, 0x31);
						
				_mm256_store_pd(transitionMatrix + bx*hiddenStates + by,urow0);
				_mm256_store_pd(transitionMatrix + (bx+1)*hiddenStates + by,urow1);
				_mm256_store_pd(transitionMatrix + (bx+2)*hiddenStates + by,urow2);
				_mm256_store_pd(transitionMatrix + (bx+3)*hiddenStates + by,urow3);	
				
				_mm256_store_pd(transitionMatrix + by*hiddenStates + bx,lrow0);
				_mm256_store_pd(transitionMatrix + (by+1)*hiddenStates + bx,lrow1);
				_mm256_store_pd(transitionMatrix + (by+2)*hiddenStates + bx,lrow2);
				_mm256_store_pd(transitionMatrix + (by+3)*hiddenStates + bx,lrow3);
			}	
		}

		for(int v = 0; v < differentObservables; v++){
			for(int s = 0; s < hiddenStates; s+=4){
				for(int j = 0; j < hiddenStates; j+=4){	
					__m256d transition0 = _mm256_load_pd(transitionMatrix+ s * hiddenStates+j);
					__m256d transition1 = _mm256_load_pd(transitionMatrix+ (s+1) * hiddenStates+j);
					__m256d transition2 = _mm256_load_pd(transitionMatrix+ (s+2) * hiddenStates+j);
					__m256d transition3 = _mm256_load_pd(transitionMatrix+ (s+3) * hiddenStates+j);
					
					__m256d emission0 = _mm256_load_pd(emissionMatrix + v * hiddenStates+j);
					
					_mm256_store_pd(ab +(v*hiddenStates + s) * hiddenStates + j, _mm256_mul_pd(transition0,emission0));
					_mm256_store_pd(ab +(v*hiddenStates + s+1) * hiddenStates + j, _mm256_mul_pd(transition1,emission0));
					_mm256_store_pd(ab +(v*hiddenStates + s+2) * hiddenStates + j, _mm256_mul_pd(transition2,emission0));
					_mm256_store_pd(ab +(v*hiddenStates + s+3) * hiddenStates + j, _mm256_mul_pd(transition3,emission0));
					
					
				}
			}
		}
		
   		yt = observations[T-1];
		
		for(int t = T-1; t > 0; t--){
			__m256d ctt_vec = _mm256_set1_pd(ct[t-1]);
			const int yt1 = observations[t-1];

			for(int s = 0; s < hiddenStates ; s+=4){
				__m256d alphat1Ns0_vec = _mm256_set1_pd(alpha[(t-1)*hiddenStates + s]);
				__m256d alphat1Ns1_vec = _mm256_set1_pd(alpha[(t-1)*hiddenStates + s+1]);
				__m256d alphat1Ns2_vec = _mm256_set1_pd(alpha[(t-1)*hiddenStates + s+2]);
				__m256d alphat1Ns3_vec = _mm256_set1_pd(alpha[(t-1)*hiddenStates + s+3]);
											
				__m256d beta_news0 = _mm256_setzero_pd();
				__m256d beta_news1 = _mm256_setzero_pd();
				__m256d beta_news2 = _mm256_setzero_pd();
				__m256d beta_news3 = _mm256_setzero_pd();
				
				__m256d alphatNs = _mm256_load_pd(alpha + (t-1)*hiddenStates + s);

				for(int j = 0; j < hiddenStates; j+=4){									
					__m256d beta_vec = _mm256_load_pd(beta+j);
											
					__m256d abs0 = _mm256_load_pd(ab + (yt*hiddenStates + s)*hiddenStates + j);
					__m256d abs1 = _mm256_load_pd(ab + (yt*hiddenStates + s+1)*hiddenStates + j);
					__m256d abs2 = _mm256_load_pd(ab + (yt*hiddenStates + s+2)*hiddenStates + j);
					__m256d abs3 = _mm256_load_pd(ab + (yt*hiddenStates + s+3)*hiddenStates + j);
					
					__m256d temp = _mm256_mul_pd(abs0,beta_vec);
					__m256d temp1 = _mm256_mul_pd(abs1,beta_vec);
					__m256d temp2 = _mm256_mul_pd(abs2,beta_vec);
					__m256d temp3 = _mm256_mul_pd(abs3,beta_vec);
					
					__m256d a_new_vec = _mm256_load_pd(a_new + s*hiddenStates+j);
					__m256d a_new_vec1 = _mm256_load_pd(a_new + (s+1)*hiddenStates+j);
					__m256d a_new_vec2 = _mm256_load_pd(a_new + (s+2)*hiddenStates+j);
					__m256d a_new_vec3 = _mm256_load_pd(a_new + (s+3)*hiddenStates+j);
					
					__m256d a_new_vec_fma = _mm256_fmadd_pd(alphat1Ns0_vec, temp,a_new_vec);
					__m256d a_new_vec1_fma = _mm256_fmadd_pd(alphat1Ns1_vec, temp1,a_new_vec1);
					__m256d a_new_vec2_fma = _mm256_fmadd_pd(alphat1Ns2_vec, temp2,a_new_vec2);
					__m256d a_new_vec3_fma = _mm256_fmadd_pd(alphat1Ns3_vec, temp3,a_new_vec3);
					
					_mm256_store_pd(a_new + s*hiddenStates+j,a_new_vec_fma);
					_mm256_store_pd(a_new + (s+1)*hiddenStates+j, a_new_vec1_fma);
					_mm256_store_pd(a_new + (s+2)*hiddenStates+j,a_new_vec2_fma);
					_mm256_store_pd(a_new + (s+3)*hiddenStates+j,a_new_vec3_fma);
					
					beta_news0 = _mm256_add_pd(beta_news0,temp);
					beta_news1 = _mm256_add_pd(beta_news1,temp1);
					beta_news2 = _mm256_add_pd(beta_news2,temp2);
					beta_news3 = _mm256_add_pd(beta_news3,temp3);
				}
							
				__m256d gamma_sum_vec = _mm256_load_pd(gamma_sum + s);
				__m256d b_new_vec = _mm256_load_pd(b_new +yt1*hiddenStates+ s);
					
				__m256d beta01 = _mm256_hadd_pd(beta_news0, beta_news1);
				__m256d beta23 = _mm256_hadd_pd(beta_news2, beta_news3);
							
				__m256d permute01 = _mm256_permute2f128_pd(beta01, beta23, 0b00110000);
				__m256d permute23 = _mm256_permute2f128_pd(beta01, beta23, 0b00100001);
								
				__m256d beta_news = _mm256_add_pd(permute01, permute23);
					
				__m256d gamma_sum_vec_fma = _mm256_fmadd_pd(alphatNs, beta_news, gamma_sum_vec);
				__m256d b_new_vec_fma = _mm256_fmadd_pd(alphatNs, beta_news,b_new_vec);
				__m256d ps = _mm256_mul_pd(alphatNs, beta_news);
				__m256d beta_news_mul = _mm256_mul_pd(beta_news, ctt_vec);
					
				_mm256_store_pd(stateProb + s, ps);
				_mm256_store_pd(beta_new + s, beta_news_mul);
				_mm256_store_pd(gamma_sum+s, gamma_sum_vec_fma);
				_mm256_store_pd(b_new +yt1*hiddenStates+ s, b_new_vec_fma);
			}

			double * temp = beta_new;
			beta_new = beta;
			beta = temp;
        		yt=yt1;
		}
        
		do{
			yt = observations[T-1];

			//add remaining parts of the sum of gamma 			
			for(int s = 0; s < hiddenStates; s+=4){
				__m256d gamma_Ts=_mm256_load_pd(gamma_T+s);
				__m256d gamma_sums=_mm256_load_pd(gamma_sum+s);
				__m256d b=_mm256_load_pd(b_new+yt*hiddenStates+s);
				
				__m256d gamma_tot=_mm256_add_pd(gamma_Ts,gamma_sums);
				__m256d b_add=_mm256_add_pd(b,gamma_Ts);
				__m256d gamma_tot_div=_mm256_div_pd(one,gamma_tot);
				__m256d gamma_sums_div=_mm256_div_pd(one,gamma_sums);
				
				_mm256_store_pd(gamma_T+s,gamma_tot_div);
				_mm256_store_pd(gamma_sum+s,gamma_sums_div);
				_mm256_store_pd(b_new+yt*hiddenStates+s,b_add);	   
			}

			//compute new emission matrix
			__m256d zero = _mm256_setzero_pd();
			
			for(int v = 0; v < differentObservables; v+=4){
				for(int s = 0; s < hiddenStates; s+=4){
					__m256d gamma_Tv = _mm256_load_pd(gamma_T + s);
				
					__m256d b_newv0 = _mm256_load_pd(b_new + v * hiddenStates + s);
					__m256d b_newv1 = _mm256_load_pd(b_new + (v+1) * hiddenStates + s);
					__m256d b_newv2 = _mm256_load_pd(b_new + (v+2) * hiddenStates + s);
					__m256d b_newv3 = _mm256_load_pd(b_new + (v+3) * hiddenStates + s);
				
					__m256d b_temp0 = _mm256_mul_pd(b_newv0,gamma_Tv);
					__m256d b_temp1 = _mm256_mul_pd(b_newv1,gamma_Tv);
					__m256d b_temp2 = _mm256_mul_pd(b_newv2,gamma_Tv);
					__m256d b_temp3 = _mm256_mul_pd(b_newv3,gamma_Tv);
				
					_mm256_store_pd(emissionMatrix + v *hiddenStates + s, b_temp0);
					_mm256_store_pd(emissionMatrix + (v+1) *hiddenStates + s, b_temp1);
					_mm256_store_pd(emissionMatrix + (v+2) *hiddenStates + s, b_temp2);
					_mm256_store_pd(emissionMatrix + (v+3) *hiddenStates + s, b_temp3);
		
					_mm256_store_pd(b_new+v*hiddenStates+s,zero);
					_mm256_store_pd(b_new+(v+1)*hiddenStates+s,zero);
					_mm256_store_pd(b_new+(v+2)*hiddenStates+s,zero);
					_mm256_store_pd(b_new+(v+3)*hiddenStates+s,zero);	
				}
			}

			//FORWARD

			//Transpose a_new
			    
			for(int by = 0; by < hiddenStates; by+=4){
	
				//Diagonal block
				__m256d diag0 = _mm256_load_pd(a_new + by*hiddenStates + by);
				__m256d diag1 = _mm256_load_pd(a_new + (by+1)*hiddenStates + by);
				__m256d diag2 = _mm256_load_pd(a_new + (by+2)*hiddenStates + by);
				__m256d diag3 = _mm256_load_pd(a_new + (by+3)*hiddenStates + by);
		
				__m256d tmp0 = _mm256_shuffle_pd(diag0,diag1, 0x0);
				__m256d tmp1 = _mm256_shuffle_pd(diag2,diag3, 0x0);
				__m256d tmp2 = _mm256_shuffle_pd(diag0,diag1, 0xF);
				__m256d tmp3 = _mm256_shuffle_pd(diag2,diag3, 0xF);
                    	
				__m256d row0 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
				__m256d row1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
				__m256d row2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
				__m256d row3 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);
			
				_mm256_store_pd(a_new + by*hiddenStates + by,row0);
				_mm256_store_pd(a_new + (by+1)*hiddenStates + by,row1);
				_mm256_store_pd(a_new + (by+2)*hiddenStates + by,row2);
				_mm256_store_pd(a_new + (by+3)*hiddenStates + by,row3);
	
				
				//Offdiagonal blocks
				for(int bx = by + 4; bx < hiddenStates; bx+= 4){
										
					__m256d upper0 = _mm256_load_pd(a_new + by*hiddenStates + bx);
					__m256d upper1 = _mm256_load_pd(a_new + (by+1)*hiddenStates + bx);
					__m256d upper2 = _mm256_load_pd(a_new + (by+2)*hiddenStates + bx);
					__m256d upper3 = _mm256_load_pd(a_new + (by+3)*hiddenStates + bx);
										
					__m256d lower0 = _mm256_load_pd(a_new + bx * hiddenStates + by);
					__m256d lower1 = _mm256_load_pd(a_new + (bx+1)*hiddenStates + by);
					__m256d lower2 = _mm256_load_pd(a_new + (bx+2)*hiddenStates + by);
					__m256d lower3 = _mm256_load_pd(a_new + (bx+3)*hiddenStates + by);
				
					__m256d utmp0 = _mm256_shuffle_pd(upper0,upper1, 0x0);
					__m256d utmp1 = _mm256_shuffle_pd(upper2,upper3, 0x0);
					__m256d utmp2 = _mm256_shuffle_pd(upper0,upper1, 0xF);
					__m256d utmp3 = _mm256_shuffle_pd(upper2,upper3, 0xF);
					
					__m256d ltmp0 = _mm256_shuffle_pd(lower0,lower1, 0x0);
					__m256d ltmp1 = _mm256_shuffle_pd(lower2,lower3, 0x0);
					__m256d ltmp2 = _mm256_shuffle_pd(lower0,lower1, 0xF);
					__m256d ltmp3 = _mm256_shuffle_pd(lower2,lower3, 0xF);
        				            
					__m256d urow0 = _mm256_permute2f128_pd(utmp0, utmp1, 0x20);
					__m256d urow1 = _mm256_permute2f128_pd(utmp2, utmp3, 0x20);
					__m256d urow2 = _mm256_permute2f128_pd(utmp0, utmp1, 0x31);
					__m256d urow3 = _mm256_permute2f128_pd(utmp2, utmp3, 0x31);
	        			            
					__m256d lrow0 = _mm256_permute2f128_pd(ltmp0, ltmp1, 0x20);
					__m256d lrow1 = _mm256_permute2f128_pd(ltmp2, ltmp3, 0x20);
					__m256d lrow2 = _mm256_permute2f128_pd(ltmp0, ltmp1, 0x31);
					__m256d lrow3 = _mm256_permute2f128_pd(ltmp2, ltmp3, 0x31);
						
					_mm256_store_pd(a_new + by*hiddenStates + bx,lrow0);
					_mm256_store_pd(a_new + (by+1)*hiddenStates + bx,lrow1);
					_mm256_store_pd(a_new + (by+2)*hiddenStates + bx,lrow2);
					_mm256_store_pd(a_new + (by+3)*hiddenStates + bx,lrow3);
						
					_mm256_store_pd(a_new + bx*hiddenStates + by,urow0);
					_mm256_store_pd(a_new + (bx+1)*hiddenStates + by,urow1);
					_mm256_store_pd(a_new + (bx+2)*hiddenStates + by,urow2);
					_mm256_store_pd(a_new + (bx+3)*hiddenStates + by,urow3);	
			
				}	
			}

			int y0 = observations[0];
		  	__m256d ct0_vec = _mm256_setzero_pd();

			//compute alpha(0)
		        for(int s = 0; s < hiddenStates; s+=4){
					__m256d stateProb_vec = _mm256_load_pd(stateProb +s);
					__m256d emission_vec = _mm256_load_pd(emissionMatrix +y0*hiddenStates +s);
					__m256d alphas_vec = _mm256_mul_pd(stateProb_vec, emission_vec);
					ct0_vec = _mm256_fmadd_pd(stateProb_vec,emission_vec, ct0_vec);
					_mm256_store_pd(alpha+s,alphas_vec);
	        	}	
	        	
	        		        	
	        	__m256d perm = _mm256_permute2f128_pd(ct0_vec,ct0_vec,0b00000011);
	
			__m256d shuffle1 = _mm256_shuffle_pd(ct0_vec, perm, 0b0101);
			__m256d shuffle2 = _mm256_shuffle_pd(perm, ct0_vec, 0b0101);
		
			__m256d ct0_vec_add = _mm256_add_pd(ct0_vec, perm);
			__m256d ct0_temp = _mm256_add_pd(shuffle1, shuffle2);
			__m256d ct0_vec_tot = _mm256_add_pd(ct0_vec_add, ct0_temp);
			__m256d ct0_vec_div = _mm256_div_pd(one,ct0_vec_tot);
			
	      		_mm256_storeu_pd(ct,ct0_vec_div);
	   
	        	for(int s = 0; s < hiddenStates; s+=4){
				__m256d alphas=_mm256_load_pd(alpha+s);
				__m256d alphas_mul=_mm256_mul_pd(alphas,ct0_vec_div);
				_mm256_store_pd(alpha+s,alphas_mul);

	        	}
	
			yt = observations[1];	
			__m256d ctt_vec = _mm256_setzero_pd();

			//Compute alpha(1) and scale transitionMatrix
			for(int s = 0; s<hiddenStates; s+=4){	
				__m256d alphatNs0 = _mm256_setzero_pd();
				__m256d alphatNs1 = _mm256_setzero_pd();
				__m256d alphatNs2 = _mm256_setzero_pd();
				__m256d alphatNs3 = _mm256_setzero_pd();
				
				for(int j = 0; j < hiddenStates; j+=4){
					__m256d gammaSum=_mm256_load_pd(gamma_sum+j);
					__m256d aNew0=_mm256_load_pd(a_new+s*hiddenStates+j);
					__m256d aNew1=_mm256_load_pd(a_new+(s+1)*hiddenStates+j);
					__m256d aNew2=_mm256_load_pd(a_new+(s+2)*hiddenStates+j);
					__m256d aNew3=_mm256_load_pd(a_new+(s+3)*hiddenStates+j);
					
					__m256d as0=_mm256_mul_pd(aNew0,gammaSum);
					__m256d as1=_mm256_mul_pd(aNew1,gammaSum);
					__m256d as2=_mm256_mul_pd(aNew2,gammaSum);
					__m256d as3=_mm256_mul_pd(aNew3,gammaSum);

					__m256d zeroes = _mm256_setzero_pd();
					_mm256_store_pd(a_new+s*hiddenStates+j,zeroes);
					_mm256_store_pd(a_new+(s+1)*hiddenStates+j,zeroes);
					_mm256_store_pd(a_new+(s+2)*hiddenStates+j,zeroes);
					_mm256_store_pd(a_new+(s+3)*hiddenStates+j,zeroes);

					_mm256_store_pd(transitionMatrix+(s)*hiddenStates+j,as0);
					_mm256_store_pd(transitionMatrix+(s+1)*hiddenStates+j,as1);
					_mm256_store_pd(transitionMatrix+(s+2)*hiddenStates+j,as2);
					_mm256_store_pd(transitionMatrix+(s+3)*hiddenStates+j,as3);
					
					__m256d alphaFactor = _mm256_load_pd(alpha + j);
					
					alphatNs0 =_mm256_fmadd_pd(alphaFactor,as0,alphatNs0);
					alphatNs1 =_mm256_fmadd_pd(alphaFactor,as1,alphatNs1);
					alphatNs2 =_mm256_fmadd_pd(alphaFactor,as2,alphatNs2);
					alphatNs3 =_mm256_fmadd_pd(alphaFactor,as3,alphatNs3);
				}
				
				__m256d emission = _mm256_load_pd(emissionMatrix + yt*hiddenStates + s);
				
				__m256d alpha01 = _mm256_hadd_pd(alphatNs0, alphatNs1);
				__m256d alpha23 = _mm256_hadd_pd(alphatNs2, alphatNs3);
								
				__m256d permute01 = _mm256_permute2f128_pd(alpha01, alpha23, 0b00110000);
				__m256d permute23 = _mm256_permute2f128_pd(alpha01, alpha23, 0b00100001);
								
				__m256d alpha_tot = _mm256_add_pd(permute01, permute23);
				__m256d alpha_tot_mul = _mm256_mul_pd(alpha_tot,emission);

				ctt_vec = _mm256_add_pd(alpha_tot_mul,ctt_vec);

				_mm256_store_pd(alpha + hiddenStates + s,alpha_tot_mul);
			}
					
			__m256d perm1 = _mm256_permute2f128_pd(ctt_vec,ctt_vec,0b00000011);
	
			__m256d shuffle11 = _mm256_shuffle_pd(ctt_vec, perm1, 0b0101);
			__m256d shuffle21 = _mm256_shuffle_pd(perm1, ctt_vec, 0b0101);
		
			__m256d ctt_vec_add = _mm256_add_pd(ctt_vec, perm1);
			__m256d ctt_temp = _mm256_add_pd(shuffle11, shuffle21);
			__m256d ctt_vec_tot = _mm256_add_pd(ctt_vec_add, ctt_temp);
			__m256d ctt_vec_div = _mm256_div_pd(one,ctt_vec_tot);
		
	      		_mm256_storeu_pd(ct + 1,ctt_vec_div);  

			//scale alpha(t)
		        for(int s = 0; s<hiddenStates; s+=4){
				__m256d alphas=_mm256_load_pd(alpha+hiddenStates+s);
				__m256d alphas_mul = _mm256_mul_pd(alphas,ctt_vec_div);
				_mm256_store_pd(alpha+hiddenStates+s,alphas_mul);
	        	}

			for(int t = 2; t < T-1; t++){	
				__m256d ctt_vec = _mm256_setzero_pd();
				const int yt = observations[t];	

				for(int s = 0; s<hiddenStates; s+=4){
					__m256d alphatNs0 = _mm256_setzero_pd();
					__m256d alphatNs1 = _mm256_setzero_pd();
					__m256d alphatNs2 = _mm256_setzero_pd();
					__m256d alphatNs3 = _mm256_setzero_pd();
				
					for(int j = 0; j < hiddenStates; j+=4){
						__m256d alphaFactor=_mm256_load_pd(alpha+(t-1)*hiddenStates+j);
					
						__m256d transition0=_mm256_load_pd(transitionMatrix+(s)*hiddenStates+j);
						__m256d transition1=_mm256_load_pd(transitionMatrix+(s+1)*hiddenStates+j);
						__m256d transition2=_mm256_load_pd(transitionMatrix+(s+2)*hiddenStates+j);
						__m256d transition3=_mm256_load_pd(transitionMatrix+(s+3)*hiddenStates+j);

						alphatNs0 =_mm256_fmadd_pd(alphaFactor,transition0,alphatNs0);
						alphatNs1 =_mm256_fmadd_pd(alphaFactor,transition1,alphatNs1);
						alphatNs2 =_mm256_fmadd_pd(alphaFactor,transition2,alphatNs2);
						alphatNs3 =_mm256_fmadd_pd(alphaFactor,transition3,alphatNs3);
					}
							
					__m256d emission = _mm256_load_pd(emissionMatrix + yt*hiddenStates + s);
				
					__m256d alpha01 = _mm256_hadd_pd(alphatNs0, alphatNs1);
					__m256d alpha23 = _mm256_hadd_pd(alphatNs2, alphatNs3);
								
					__m256d permute01 = _mm256_permute2f128_pd(alpha01, alpha23, 0b00110000);
					__m256d permute23 = _mm256_permute2f128_pd(alpha01, alpha23, 0b00100001);
								
					__m256d alpha_tot = _mm256_add_pd(permute01, permute23);
					__m256d alpha_tot_mul = _mm256_mul_pd(alpha_tot,emission);

					ctt_vec = _mm256_add_pd(alpha_tot_mul,ctt_vec);
					

					_mm256_store_pd(alpha + t*hiddenStates + s,alpha_tot_mul);

				}
			
				__m256d perm = _mm256_permute2f128_pd(ctt_vec,ctt_vec,0b00000011);
	
				__m256d shuffle1 = _mm256_shuffle_pd(ctt_vec, perm, 0b0101);
				__m256d shuffle2 = _mm256_shuffle_pd(perm, ctt_vec, 0b0101);
		
				__m256d ctt_vec_add = _mm256_add_pd(ctt_vec, perm);
				__m256d ctt_temp = _mm256_add_pd(shuffle1, shuffle2);
				__m256d ctt_vec_tot = _mm256_add_pd(ctt_vec_add, ctt_temp);
				__m256d ctt_vec_div = _mm256_div_pd(one,ctt_vec_tot);
		
	      			_mm256_storeu_pd(ct + t,ctt_vec_div);        
	
				for(int s = 0; s<hiddenStates; s+=4){
					__m256d alphas=_mm256_load_pd(alpha+t*hiddenStates+s);
					__m256d alphas_mul=_mm256_mul_pd(alphas,ctt_vec_div);
					_mm256_store_pd(alpha+t*hiddenStates+s,alphas_mul);
				}
			}
			
			yt = observations[T-1];	
			__m256d ctT_vec = _mm256_setzero_pd();

			for(int s = 0; s<hiddenStates; s+=4){
	
				__m256d alphatNs0_vec = _mm256_setzero_pd();
				__m256d alphatNs1_vec = _mm256_setzero_pd();
				__m256d alphatNs2_vec = _mm256_setzero_pd();
				__m256d alphatNs3_vec = _mm256_setzero_pd();
				 
				for(int j = 0; j < hiddenStates; j+=4){
					__m256d alphaFactor=_mm256_load_pd(alpha+(T-2)*hiddenStates+j);
					
					__m256d transition0=_mm256_load_pd(transitionMatrix+(s)*hiddenStates+j);
					__m256d transition1=_mm256_load_pd(transitionMatrix+(s+1)*hiddenStates+j);
					__m256d transition2=_mm256_load_pd(transitionMatrix+(s+2)*hiddenStates+j);
					__m256d transition3=_mm256_load_pd(transitionMatrix+(s+3)*hiddenStates+j);

					alphatNs0_vec =_mm256_fmadd_pd(alphaFactor,transition0,alphatNs0_vec);
					alphatNs1_vec =_mm256_fmadd_pd(alphaFactor,transition1,alphatNs1_vec);
					alphatNs2_vec =_mm256_fmadd_pd(alphaFactor,transition2,alphatNs2_vec);
					alphatNs3_vec =_mm256_fmadd_pd(alphaFactor,transition3,alphatNs3_vec);
				}
								
				__m256d emission = _mm256_load_pd(emissionMatrix + yt*hiddenStates + s);
				
				__m256d alpha01 = _mm256_hadd_pd(alphatNs0_vec, alphatNs1_vec);
				__m256d alpha23 = _mm256_hadd_pd(alphatNs2_vec, alphatNs3_vec);
							
				__m256d permute01 = _mm256_permute2f128_pd(alpha01, alpha23, 0b00110000);
				__m256d permute23 = _mm256_permute2f128_pd(alpha01, alpha23, 0b00100001);
								
				__m256d alpha_tot = _mm256_add_pd(permute01, permute23);
				__m256d alpha_tot_mul = _mm256_mul_pd(alpha_tot,emission);

				ctT_vec = _mm256_add_pd(alpha_tot_mul,ctT_vec);
					
				_mm256_store_pd(alpha + (T-1)*hiddenStates + s,alpha_tot_mul);
			}
						
			__m256d perm2 = _mm256_permute2f128_pd(ctT_vec,ctT_vec,0b00000011);
	
			__m256d shuffle12 = _mm256_shuffle_pd(ctT_vec, perm2, 0b0101);
			__m256d shuffle22 = _mm256_shuffle_pd(perm2, ctT_vec, 0b0101);
		
			__m256d ctT_vec_add = _mm256_add_pd(ctT_vec, perm2);
			__m256d ctT_temp = _mm256_add_pd(shuffle12, shuffle22);
			__m256d ctT_vec_tot = _mm256_add_pd(ctT_vec_add, ctT_temp);
			__m256d ctT_vec_div = _mm256_div_pd(one,ctT_vec_tot);
		
	      		_mm256_storeu_pd(ct + (T-1),ctT_vec_div); 
	      			        
			for(int s = 0; s<hiddenStates; s+=4){
				__m256d alphaT1Ns=_mm256_load_pd(alpha+(T-1)*hiddenStates+s);
				__m256d alphaT1Ns_mul=_mm256_mul_pd(alphaT1Ns,ctT_vec_div);
				_mm256_store_pd(alpha+(T-1)*hiddenStates+s,alphaT1Ns_mul);
				_mm256_store_pd(gamma_T+s,alphaT1Ns_mul);

			}

			//Transpose transitionMatrix
			for(int by = 0; by < hiddenStates; by+=4){
	
				//Diagonal block
				__m256d diag0 = _mm256_load_pd(transitionMatrix + by*hiddenStates + by);
				__m256d diag1 = _mm256_load_pd(transitionMatrix + (by+1)*hiddenStates + by);
				__m256d diag2 = _mm256_load_pd(transitionMatrix + (by+2)*hiddenStates + by);
				__m256d diag3 = _mm256_load_pd(transitionMatrix + (by+3)*hiddenStates + by);
		
				__m256d tmp0 = _mm256_shuffle_pd(diag0,diag1, 0x0);
				__m256d tmp1 = _mm256_shuffle_pd(diag2,diag3, 0x0);
				__m256d tmp2 = _mm256_shuffle_pd(diag0,diag1, 0xF);
				__m256d tmp3 = _mm256_shuffle_pd(diag2,diag3, 0xF);
                    	
				__m256d row0 = _mm256_permute2f128_pd(tmp0, tmp1, 0x20);
				__m256d row1 = _mm256_permute2f128_pd(tmp2, tmp3, 0x20);
				__m256d row2 = _mm256_permute2f128_pd(tmp0, tmp1, 0x31);
				__m256d row3 = _mm256_permute2f128_pd(tmp2, tmp3, 0x31);
			
				_mm256_store_pd(transitionMatrix + by*hiddenStates + by,row0);
				_mm256_store_pd(transitionMatrix + (by+1)*hiddenStates + by,row1);
				_mm256_store_pd(transitionMatrix + (by+2)*hiddenStates + by,row2);
				_mm256_store_pd(transitionMatrix + (by+3)*hiddenStates + by,row3);
	
				//Offdiagonal blocks
				for(int bx = by + 4; bx < hiddenStates; bx+= 4){
										
					__m256d upper0 = _mm256_load_pd(transitionMatrix + by*hiddenStates + bx);
					__m256d upper1 = _mm256_load_pd(transitionMatrix + (by+1)*hiddenStates + bx);
					__m256d upper2 = _mm256_load_pd(transitionMatrix + (by+2)*hiddenStates + bx);
					__m256d upper3 = _mm256_load_pd(transitionMatrix + (by+3)*hiddenStates + bx);
										
					__m256d lower0 = _mm256_load_pd(transitionMatrix + bx * hiddenStates + by);
					__m256d lower1 = _mm256_load_pd(transitionMatrix + (bx+1)*hiddenStates + by);
					__m256d lower2 = _mm256_load_pd(transitionMatrix + (bx+2)*hiddenStates + by);
					__m256d lower3 = _mm256_load_pd(transitionMatrix + (bx+3)*hiddenStates + by);
				
					__m256d utmp0 = _mm256_shuffle_pd(upper0,upper1, 0x0);
					__m256d utmp1 = _mm256_shuffle_pd(upper2,upper3, 0x0);
					__m256d utmp2 = _mm256_shuffle_pd(upper0,upper1, 0xF);
					__m256d utmp3 = _mm256_shuffle_pd(upper2,upper3, 0xF);
					
					__m256d ltmp0 = _mm256_shuffle_pd(lower0,lower1, 0x0);
					__m256d ltmp1 = _mm256_shuffle_pd(lower2,lower3, 0x0);
					__m256d ltmp2 = _mm256_shuffle_pd(lower0,lower1, 0xF);
					__m256d ltmp3 = _mm256_shuffle_pd(lower2,lower3, 0xF);
        				            
					__m256d urow0 = _mm256_permute2f128_pd(utmp0, utmp1, 0x20);
					__m256d urow1 = _mm256_permute2f128_pd(utmp2, utmp3, 0x20);
					__m256d urow2 = _mm256_permute2f128_pd(utmp0, utmp1, 0x31);
					__m256d urow3 = _mm256_permute2f128_pd(utmp2, utmp3, 0x31);
	        			            
					__m256d lrow0 = _mm256_permute2f128_pd(ltmp0, ltmp1, 0x20);
					__m256d lrow1 = _mm256_permute2f128_pd(ltmp2, ltmp3, 0x20);
					__m256d lrow2 = _mm256_permute2f128_pd(ltmp0, ltmp1, 0x31);
					__m256d lrow3 = _mm256_permute2f128_pd(ltmp2, ltmp3, 0x31);
						
					_mm256_store_pd(transitionMatrix + by*hiddenStates + bx,lrow0);
					_mm256_store_pd(transitionMatrix + (by+1)*hiddenStates + bx,lrow1);
					_mm256_store_pd(transitionMatrix + (by+2)*hiddenStates + bx,lrow2);
					_mm256_store_pd(transitionMatrix + (by+3)*hiddenStates + bx,lrow3);
						
					_mm256_store_pd(transitionMatrix + bx*hiddenStates + by,urow0);
					_mm256_store_pd(transitionMatrix + (bx+1)*hiddenStates + by,urow1);
					_mm256_store_pd(transitionMatrix + (bx+2)*hiddenStates + by,urow2);
					_mm256_store_pd(transitionMatrix + (bx+3)*hiddenStates + by,urow3);	
				}	
			}
			
			for(int s = 0; s < hiddenStates; s+=4){			        
			        _mm256_store_pd(beta+s, ctT_vec_div);
	       	 	}
	        
	       		for(int s = 0; s < hiddenStates; s+=4){
			        _mm256_store_pd(gamma_sum+s, _mm256_setzero_pd());
	        	}

			for(int v = 0; v < differentObservables; v++){
				for(int s = 0; s < hiddenStates; s+=4){
					for(int j = 0; j < hiddenStates; j+=4){	
					
						__m256d transition0 = _mm256_load_pd(transitionMatrix+ s * hiddenStates+j);
						__m256d transition1 = _mm256_load_pd(transitionMatrix+ (s+1) * hiddenStates+j);
						__m256d transition2 = _mm256_load_pd(transitionMatrix+ (s+2) * hiddenStates+j);
						__m256d transition3 = _mm256_load_pd(transitionMatrix+ (s+3) * hiddenStates+j);
					
						__m256d emission0 = _mm256_load_pd(emissionMatrix + v * hiddenStates+j);
						
						_mm256_store_pd(ab +(v*hiddenStates + s) * hiddenStates + j, _mm256_mul_pd(transition0,emission0));
						_mm256_store_pd(ab +(v*hiddenStates + s+1) * hiddenStates + j, _mm256_mul_pd(transition1,emission0));
						_mm256_store_pd(ab +(v*hiddenStates + s+2) * hiddenStates + j, _mm256_mul_pd(transition2,emission0));
						_mm256_store_pd(ab +(v*hiddenStates + s+3) * hiddenStates + j, _mm256_mul_pd(transition3,emission0));
					}
				}
			}
		
   			yt = observations[T-1];

			for(int t = T-1; t > 0; t--){
				__m256d ctt_vec = _mm256_set1_pd(ct[t-1]);
				const int yt1 = observations[t-1];

				for(int s = 0; s < hiddenStates ; s+=4){
					
					__m256d alphat1Ns0_vec = _mm256_set1_pd(alpha[(t-1)*hiddenStates + s]);
					__m256d alphat1Ns1_vec = _mm256_set1_pd(alpha[(t-1)*hiddenStates + s+1]);
					__m256d alphat1Ns2_vec = _mm256_set1_pd(alpha[(t-1)*hiddenStates + s+2]);
					__m256d alphat1Ns3_vec = _mm256_set1_pd(alpha[(t-1)*hiddenStates + s+3]);
												
					__m256d beta_news0 = _mm256_setzero_pd();
					__m256d beta_news1 = _mm256_setzero_pd();
					__m256d beta_news2 = _mm256_setzero_pd();
					__m256d beta_news3 = _mm256_setzero_pd();
					
					__m256d alphatNs = _mm256_load_pd(alpha + (t-1)*hiddenStates + s);

					for(int j = 0; j < hiddenStates; j+=4){
										
						__m256d beta_vec = _mm256_load_pd(beta+j);
												
						__m256d abs0 = _mm256_load_pd(ab + (yt*hiddenStates + s)*hiddenStates + j);
						__m256d abs1 = _mm256_load_pd(ab + (yt*hiddenStates + s+1)*hiddenStates + j);
						__m256d abs2 = _mm256_load_pd(ab + (yt*hiddenStates + s+2)*hiddenStates + j);
						__m256d abs3 = _mm256_load_pd(ab + (yt*hiddenStates + s+3)*hiddenStates + j);
	
						__m256d temp = _mm256_mul_pd(abs0,beta_vec);
						__m256d temp1 = _mm256_mul_pd(abs1,beta_vec);
						__m256d temp2 = _mm256_mul_pd(abs2,beta_vec);
						__m256d temp3 = _mm256_mul_pd(abs3,beta_vec);
						
						__m256d a_new_vec = _mm256_load_pd(a_new + s*hiddenStates+j);
						__m256d a_new_vec1 = _mm256_load_pd(a_new + (s+1)*hiddenStates+j);
						__m256d a_new_vec2 = _mm256_load_pd(a_new + (s+2)*hiddenStates+j);
						__m256d a_new_vec3 = _mm256_load_pd(a_new + (s+3)*hiddenStates+j);
						
						__m256d a_new_vec_fma = _mm256_fmadd_pd(alphat1Ns0_vec, temp,a_new_vec);
						__m256d a_new_vec1_fma = _mm256_fmadd_pd(alphat1Ns1_vec, temp1,a_new_vec1);
						__m256d a_new_vec2_fma = _mm256_fmadd_pd(alphat1Ns2_vec, temp2,a_new_vec2);
						__m256d a_new_vec3_fma = _mm256_fmadd_pd(alphat1Ns3_vec, temp3,a_new_vec3);
						
						_mm256_store_pd(a_new + s*hiddenStates+j,a_new_vec_fma);
						_mm256_store_pd(a_new + (s+1)*hiddenStates+j, a_new_vec1_fma);
						_mm256_store_pd(a_new + (s+2)*hiddenStates+j,a_new_vec2_fma);
						_mm256_store_pd(a_new + (s+3)*hiddenStates+j,a_new_vec3_fma);
						
						beta_news0 = _mm256_add_pd(beta_news0,temp);
						beta_news1 = _mm256_add_pd(beta_news1,temp1);
						beta_news2 = _mm256_add_pd(beta_news2,temp2);
						beta_news3 = _mm256_add_pd(beta_news3,temp3);
					}
											
					__m256d gamma_sum_vec = _mm256_load_pd(gamma_sum + s);
					__m256d b_new_vec = _mm256_load_pd(b_new +yt1*hiddenStates+ s);
					
					__m256d beta01 = _mm256_hadd_pd(beta_news0, beta_news1);
					__m256d beta23 = _mm256_hadd_pd(beta_news2, beta_news3);
							
					__m256d permute01 = _mm256_permute2f128_pd(beta01, beta23, 0b00110000);
					__m256d permute23 = _mm256_permute2f128_pd(beta01, beta23, 0b00100001);
								
					__m256d beta_news = _mm256_add_pd(permute01, permute23);
					
					__m256d gamma_sum_vec_fma = _mm256_fmadd_pd(alphatNs, beta_news, gamma_sum_vec);
					__m256d b_new_vec_fma = _mm256_fmadd_pd(alphatNs, beta_news,b_new_vec);
					__m256d ps = _mm256_mul_pd(alphatNs, beta_news);
					__m256d beta_news_mul = _mm256_mul_pd(beta_news, ctt_vec);
					
					_mm256_store_pd(stateProb + s, ps);
					_mm256_store_pd(beta_new + s, beta_news_mul);
					_mm256_store_pd(gamma_sum+s, gamma_sum_vec_fma);
					_mm256_store_pd(b_new +yt1*hiddenStates+ s, b_new_vec_fma);	
		
				}

				double * temp = beta_new;
				beta_new = beta;
				beta = temp;
        			yt=yt1;
			}
        
        		steps+=1;
        		
		        double oldLogLikelihood=logLikelihood;
		        double newLogLikelihood = 0.0;
			
			#ifdef __INTEL_COMPILER

			__m256d logLikelihood_vec = _mm256_setzero_pd();
			
		        for(int time = 0; time < T; time+=4){
			        __m256d ct_vec = _mm256_load_pd(ct + time);
			        __m256d log2 = _mm256_log_pd(ct_vec);
			        logLikelihood_vec = _mm256_sub_pd(logLikelihood_vec , log2);
		        }
		        
		        _mm256_store_pd(reduction,logLikelihood_vec);
			newLogLikelihood = reduction[0] +reduction[1] +reduction[2] +reduction[3];
			
			#elif __GNUC__
			
		        for(int time = 0; time < T; time++){
			        newLogLikelihood -= log2(ct[time]);
			 }
			#endif

		        logLikelihood=newLogLikelihood;
		        disparance=newLogLikelihood-oldLogLikelihood;
	
		}while (disparance>EPSILON && steps<maxSteps);
    
		yt = observations[T-1];

		//add remaining parts of the sum of gamma 
		for(int s = 0; s < hiddenStates; s+=4){
	        	
	        	__m256d gamma_Ts = _mm256_load_pd(gamma_T + s);
	        	__m256d gamma_sums = _mm256_load_pd(gamma_sum + s);
	        	__m256d b_new_vec = _mm256_load_pd(b_new + yt*hiddenStates + s);
	        	
	        	__m256d gamma_tot = _mm256_add_pd(gamma_Ts, gamma_sums);
	        	__m256d gamma_T_inv =  _mm256_div_pd(one,gamma_tot);
			__m256d gamma_sums_inv = _mm256_div_pd(one,gamma_sums);
			
			_mm256_store_pd(gamma_T+s,gamma_T_inv);
			_mm256_store_pd(gamma_sum + s,gamma_sums_inv);
	        	_mm256_store_pd(b_new + yt*hiddenStates + s, _mm256_add_pd(b_new_vec, gamma_Ts));
		}
		
		
		for(int s = 0; s < hiddenStates; s+=4){
			
			__m256d gamma_inv0 = _mm256_set1_pd(gamma_sum[s]);
			__m256d gamma_inv1 = _mm256_set1_pd(gamma_sum[s+1]);		
			__m256d gamma_inv2 = _mm256_set1_pd(gamma_sum[s+2]);		
			__m256d gamma_inv3 = _mm256_set1_pd(gamma_sum[s+3]);
					
			for(int j = 0; j < hiddenStates; j+=4){
			
				__m256d a_news = _mm256_load_pd(a_new + s *hiddenStates+j);
				__m256d a_news1 = _mm256_load_pd(a_new + (s+1) *hiddenStates+j);
				__m256d a_news2 = _mm256_load_pd(a_new + (s+2) *hiddenStates+j);
				__m256d a_news3 = _mm256_load_pd(a_new + (s+3) *hiddenStates+j);
				
				__m256d temp0 = _mm256_mul_pd(a_news, gamma_inv0);
				__m256d temp1 = _mm256_mul_pd(a_news1, gamma_inv1);
				__m256d temp2 = _mm256_mul_pd(a_news2, gamma_inv2);
				__m256d temp3 = _mm256_mul_pd(a_news3, gamma_inv3);
				
				_mm256_store_pd(transitionMatrix + s*hiddenStates+j, temp0);
				_mm256_store_pd(transitionMatrix + (s+1)*hiddenStates+j, temp1);
				_mm256_store_pd(transitionMatrix + (s+2)*hiddenStates+j, temp2);
				_mm256_store_pd(transitionMatrix + (s+3)*hiddenStates+j, temp3);

			}
		}
		
		//compute new emission matrix
		for(int v = 0; v < differentObservables; v+=4){
			for(int s = 0; s < hiddenStates; s+=4){
			
				__m256d gamma_Tv = _mm256_load_pd(gamma_T + s);
				
				__m256d b_newv0 = _mm256_load_pd(b_new + v * hiddenStates + s);
				__m256d b_newv1 = _mm256_load_pd(b_new + (v+1) * hiddenStates + s);
				__m256d b_newv2 = _mm256_load_pd(b_new + (v+2) * hiddenStates + s);
				__m256d b_newv3 = _mm256_load_pd(b_new + (v+3) * hiddenStates + s);
				
				__m256d b_temp0 = _mm256_mul_pd(b_newv0,gamma_Tv);
				__m256d b_temp1 = _mm256_mul_pd(b_newv1,gamma_Tv);
				__m256d b_temp2 = _mm256_mul_pd(b_newv2,gamma_Tv);
				__m256d b_temp3 = _mm256_mul_pd(b_newv3,gamma_Tv);
				
				_mm256_store_pd(emissionMatrix + v *hiddenStates + s, b_temp0);
				_mm256_store_pd(emissionMatrix + (v+1) *hiddenStates + s, b_temp1);
				_mm256_store_pd(emissionMatrix + (v+2) *hiddenStates + s, b_temp2);
				_mm256_store_pd(emissionMatrix + (v+3) *hiddenStates + s, b_temp3);
		
			}
		}
		
		cycles = stop_tsc(start);
       		cycles = cycles/steps;
		runs[run]=cycles;

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
	

	//printf("steps = %i\n", steps);
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
        
    	_mm_free(groundTransitionMatrix);
	_mm_free(groundEmissionMatrix);
	_mm_free(observations);
	_mm_free(transitionMatrix);
	_mm_free(emissionMatrix);
	_mm_free(stateProb);
   	_mm_free(ct);
	_mm_free(gamma_T);
	_mm_free(gamma_sum);
	_mm_free(a_new);
	_mm_free(b_new);
  	_mm_free(transitionMatrixSafe);
	_mm_free(emissionMatrixSafe);
   	_mm_free(stateProbSafe);
	_mm_free(transitionMatrixTesting);
	_mm_free(emissionMatrixTesting);
	_mm_free(stateProbTesting);
	_mm_free(beta);
	_mm_free(beta_new);
	_mm_free(alpha);
	_mm_free(ab);
	_mm_free(reduction);
	free((void*)buf);
			
	return 0; 
} 
