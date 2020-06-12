#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <float.h>

#include "tsc_x86.h"
#include "io.h"
#include "tested.h"
#include "util.h"

double EPSILON = 1e-4;
#define DELTA 1e-2
#define BUFSIZE 1<<26
#define maxSteps 100

myInt64 bw(double* transitionMatrix, double* stateProb, double* emissionMatrix, double* const alpha, double* beta,double* const gamma, double* const xi,   const int * const observations, double* const ct,double* const inv_ct, const int hiddenStates, const int differentObservables, const int T){

        double logLikelihood=-DBL_MAX;
        double disparance;
        int steps=0;
	myInt64 start = start_tsc();

	do{
            	
            	//FORWARD

	        double ct0 = 0.0;
	        int y0 = observations[0];

	        //compute alpha(0)
	        for(int s = 0; s < hiddenStates; s++){
		        double alphas = stateProb[s] * emissionMatrix[s*differentObservables + y0];
		        ct0 += alphas;
		        alpha[s] = alphas;
	        }
	        
	        ct0 = 1.0 / ct0;

	        for(int s = 0; s < hiddenStates; s++){
		        alpha[s] *= ct0;
	        }

	        ct[0] = ct0;

	        for(int t = 1; t < T; t++){
		        double ctt = 0.0;	
		        const int yt = observations[t];

		        for(int s = 0; s<hiddenStates; s++){
			        double alphatNs = 0;

			        for(int j = 0; j < hiddenStates; j++){
				        alphatNs += alpha[(t-1)*hiddenStates + j] * transitionMatrix[j*hiddenStates + s];
			        }

			        alphatNs *= emissionMatrix[s*differentObservables + yt];
			        ctt += alphatNs;
			        alpha[t*hiddenStates + s] = alphatNs;
		        }
 
		        ctt = 1.0 / ctt;
		        
		        for(int s = 0; s<hiddenStates; s++){
			        alpha[t*hiddenStates+s] *= ctt;
		        }

		        ct[t] = ctt;
	        }	

	        //BACKWARD
	        double ctT1 = ct[T-1];
	
	        for(int s = 0; s < hiddenStates; s++){
		        beta[(T-1)*hiddenStates + s] = ctT1;
	        }

	        for(int t = T-1; t > 0; t--){
		        const int yt =observations[t];
		        const double ctt1 = ct[t-1];

		        for(int s = 0; s < hiddenStates; s++){
			        double betat1Ns = 0;

			        for(int j = 0; j < hiddenStates; j++){
				        betat1Ns += beta[t*hiddenStates + j ] * transitionMatrix[s*hiddenStates + j] * emissionMatrix[j*differentObservables + yt];
			        }

			        beta[(t-1)*hiddenStates + s] = ctt1*betat1Ns;
		        }
	        }

            	//UPDATE
	        double xi_sum, gamma_sum_numerator, gamma_sum_denominator;

	        for(int t = 0; t < T; t++){
		        for(int s = 0; s < hiddenStates; s++){
			        gamma[t*hiddenStates + s] = alpha[t*hiddenStates + s] * beta[t*hiddenStates + s];
		        }
	        }

	        for(int t = 1; t < T; t++){
		        const int yt = observations[t];

		        for(int s = 0; s < hiddenStates; s++){
			        const double alphat1Ns = alpha[(t-1)*hiddenStates + s];

			        for(int j = 0; j < hiddenStates; j++){
				        xi[((t-1) * hiddenStates + s) * hiddenStates + j] = alphat1Ns * transitionMatrix[s*hiddenStates + j] * beta[t*hiddenStates + j] * emissionMatrix[j*differentObservables + yt]; 
			        }
		        }
	        }

	        const double ct0div = 1/ct[0];
	        inv_ct[0] = ct0div;

	        for(int s = 0; s < hiddenStates; s++){
            		stateProb[s] = gamma[s]*ct0div;
            
		        for(int j = 0; j < hiddenStates; j++){
			        xi_sum = 0.;
			        gamma_sum_denominator = 0.;

			        for(int t = 1; t < T; t++){
				        xi_sum += xi[((t-1) * hiddenStates + s) * hiddenStates + j];	
				        double inv_ctt1 = 1 / ct[t-1];
				        inv_ct[t-1] = inv_ctt1;
				        gamma_sum_denominator += gamma[(t-1)*hiddenStates + s]*inv_ctt1;
			        }

			        // new transition matrix
			        transitionMatrix[s*hiddenStates + j] = xi_sum / gamma_sum_denominator;
		        }

		        double inv_ctt1 = 1/ct[T-1];
		        inv_ct[T-1]=inv_ctt1;
		        gamma_sum_denominator += gamma[(T-1)*hiddenStates + s]*inv_ctt1;
		        const double gamma_sum_denominator_div = 1/gamma_sum_denominator;

		        for(int v = 0; v < differentObservables; v++){
			        gamma_sum_numerator = 0.;

			        for(int t = 0; t < T; t++){
				        if(observations[t] == v){
					        gamma_sum_numerator += gamma[t*hiddenStates + s]*inv_ct[t];
				        }
			        }

			        // new emmision matrix
			        emissionMatrix[s*differentObservables + v] = gamma_sum_numerator * gamma_sum_denominator_div;
		        }
	        }
            	steps+=1;

	        double oldLogLikelihood=logLikelihood;
	        double newLogLikelihood = 0.0;

	        for(int time = 0; time < T; time++){
		        newLogLikelihood -= log2(ct[time]);
	        }
	        
	        logLikelihood=newLogLikelihood;
	        disparance=newLogLikelihood-oldLogLikelihood;

	}while (disparance>EPSILON && steps<maxSteps);

	myInt64 cycles = stop_tsc(start);
	cycles = cycles/steps;
        return cycles;
}
void forward(const double* const a, const double* const p, const double* const b, double* const alpha,  const int * const y, double* const ct, const int N, const int K, const int T){

	double ct0 = 0.0;
	int y0 = y[0];

	//compute alpha(0)
	for(int s = 0; s < N; s++){
		double alphas = p[s] * b[s*K + y0];
		ct0 += alphas;
		alpha[s] = alphas;
	}
	
	ct0 = 1.0 / ct0;

	for(int s = 0; s < N; s++){
		alpha[s] *= ct0;
	}

	ct[0] = ct0;

	for(int t = 1; t < T; t++){
		double ctt = 0.0;	
		const int yt = y[t];

		for(int s = 0; s<N; s++){
			double alphatNs = 0;

			for(int j = 0; j < N; j++){
				alphatNs += alpha[(t-1)*N + j] * a[j*N + s];
			}

			alphatNs *= b[s*K + yt];
			ctt += alphatNs;
			alpha[t*N + s] = alphatNs;
		}

		ctt = 1.0 / ctt;
		
		for(int s = 0; s<N; s++){
			alpha[t*N+s] *= ctt;
		}

		ct[t] = ctt;
	}

	return;
}

void backward(const double* const a, const double* const b, double* const beta, const int * const y, const double * const ct, const int N, const int K, const int T ){
	
	double ctT1 = ct[T-1];

	for(int s = 0; s < N; s++){
		beta[(T-1)*N + s] =ctT1;
	}

	for(int t = T-1; t > 0; t--){
		const int yt =y[t];
		const double ctt1 = ct[t-1];

		for(int s = 0; s < N; s++){
			double betat1Ns = 0;

			for(int j = 0; j < N; j++){
				betat1Ns += beta[t*N + j ] * a[s*N + j] * b[j*K + yt];
			}

			beta[(t-1)*N + s] = ctt1*betat1Ns;
		}
	}
	return;
}

void update(double* const a, double* const p, double* const b, const double* const alpha, const double* const beta, double* const gamma, double* const xi, const int* const y, const double* const ct,double* const inv_ct,const int N, const int K, const int T){


	double xi_sum, gamma_sum_numerator, gamma_sum_denominator;

	for(int t = 0; t < T; t++){
		for(int s = 0; s < N; s++){
			gamma[t*N + s] = alpha[t*N + s] * beta[t*N + s];
		}
	}

	for(int t = 1; t < T; t++){
		const int yt = y[t];

		for(int s = 0; s < N; s++){
			const double alphat1Ns = alpha[(t-1)*N + s];

			for(int j = 0; j < N; j++){
				xi[((t-1) * N + s) * N + j] = alphat1Ns * a[s*N + j] * beta[t*N + j] * b[j*K + yt]; 
			}
		}
	}

	const double ct0div = 1/ct[0];
	inv_ct[0] = ct0div;

	for(int s = 0; s < N; s++){
    		p[s] = gamma[s]*ct0div;
    
		for(int j = 0; j < N; j++){
			xi_sum = 0.;
			gamma_sum_denominator = 0.;
			for(int t = 1; t < T; t++){
				xi_sum += xi[((t-1) * N + s) * N + j];	
				double inv_ctt1 = 1 / ct[t-1];
				inv_ct[t-1] = inv_ctt1;
				gamma_sum_denominator += gamma[(t-1)*N + s]*inv_ctt1;
			}

			// new transition matrix
			a[s*N + j] = xi_sum / gamma_sum_denominator;
		}

		double inv_ctt1 = 1/ct[T-1];
		inv_ct[T-1]=inv_ctt1;
		gamma_sum_denominator += gamma[(T-1)*N + s]*inv_ctt1;
		const double gamma_sum_denominator_div = 1/gamma_sum_denominator;

		for(int v = 0; v < K; v++){
			gamma_sum_numerator = 0.;

			for(int t = 0; t < T; t++){
				if(y[t] == v){
					gamma_sum_numerator += gamma[t*N + s]*inv_ct[t];
				}
			}

			// new emmision matrix
			b[s*K + v] = gamma_sum_numerator * gamma_sum_denominator_div;
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
	double* inv_ct = (double*) malloc(T * sizeof(double));
	
	for(int j=0;j<10;j++){
		forward(transitionMatrix, piVector, emissionMatrix, alpha, observations, ct, hiddenStates, differentObservables, T);	
		backward(transitionMatrix, emissionMatrix, beta, observations, ct, hiddenStates, differentObservables, T);
		update(transitionMatrix, piVector, emissionMatrix, alpha, beta, gamma, xi, observations, ct, inv_ct, hiddenStates, differentObservables, T);
	}	

	free(alpha);
	free(beta);
	free(gamma);
	free(xi);
   	free(ct);
   	free(inv_ct);
	
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
	
	const int maxRuns=10;
	const int cachegrind_runs = 1;

	double runs[maxRuns]; 
	srand(seed);

	//ground truth
	double* groundTransitionMatrix = (double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* groundEmissionMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));
	makeMatrix(hiddenStates, hiddenStates, groundTransitionMatrix);
	makeMatrix(hiddenStates, differentObservables, groundEmissionMatrix);
	int groundInitialState = rand()%hiddenStates;
	int* observations = (int*) malloc ( T * sizeof(int));
	makeObservations(hiddenStates, differentObservables, groundInitialState, groundTransitionMatrix,groundEmissionMatrix,T, observations);


	double* transitionMatrix = (double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* transitionMatrixSafe = (double*) malloc(hiddenStates*hiddenStates*sizeof(double));
	double* transitionMatrixTesting=(double*) malloc(hiddenStates*hiddenStates*sizeof(double));

	double* emissionMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));
	double* emissionMatrixSafe = (double*) malloc(hiddenStates*differentObservables*sizeof(double));
	double* emissionMatrixTesting=(double*) malloc(hiddenStates*differentObservables*sizeof(double));

	double* stateProb  = (double*) malloc(hiddenStates * sizeof(double));
	double* stateProbSafe  = (double*) malloc(hiddenStates * sizeof(double));
	double* stateProbTesting  = (double*) malloc(hiddenStates * sizeof(double));

	double* alpha = (double*) malloc(hiddenStates * T * sizeof(double));
	double* beta = (double*) malloc(hiddenStates * T * sizeof(double));
	double* gamma = (double*) malloc(hiddenStates * T * sizeof(double));
	double* xi = (double*) malloc(hiddenStates * hiddenStates * (T-1) * sizeof(double)); 
	double* ct = (double*) malloc(T*sizeof(double));
	double* inv_ct = (double*) malloc(T*sizeof(double));
	
	//random init transition matrix, emission matrix and state probabilities.
	makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
	makeMatrix(hiddenStates, differentObservables, emissionMatrix);
	makeProbabilities(stateProb,hiddenStates);

	//copy for resetting to initial state.
	memcpy(transitionMatrixSafe, transitionMatrix, hiddenStates*hiddenStates*sizeof(double));
   	memcpy(emissionMatrixSafe, emissionMatrix, hiddenStates*differentObservables*sizeof(double));
    	memcpy(stateProbSafe, stateProb, hiddenStates * sizeof(double));

	//heat up cache
	heatup(transitionMatrix,stateProb,emissionMatrix,observations,hiddenStates,differentObservables,T);
	    	
	//matrix for flushing cache
	//volatile unsigned char* buf = malloc(BUFSIZE*sizeof(char));
    		
	for (int run=0; run<cachegrind_runs; run++){

		//reset to init
       		memcpy(transitionMatrix, transitionMatrixSafe, hiddenStates*hiddenStates*sizeof(double));
	   	memcpy(emissionMatrix, emissionMatrixSafe, hiddenStates*differentObservables*sizeof(double));
        	memcpy(stateProb, stateProbSafe, hiddenStates * sizeof(double));	
       	
       		//_flush_cache(buf,BUFSIZE);
       	
		myInt64 cycles=bw(transitionMatrix, stateProb,  emissionMatrix, alpha, beta, gamma, xi, observations, ct, inv_ct, hiddenStates, differentObservables, T);
	
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
	free(transitionMatrixTesting);
	free(emissionMatrixTesting);
	free(stateProbTesting);
	//free((void*)buf);

	return 0; 
} 
