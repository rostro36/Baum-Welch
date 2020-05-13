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

void makeMatrix(const int dim1,const int dim2, double* const matrix){

	for (int row=0;row<dim1;row++){
		//make probabilites for one row
		makeProbabilities(matrix + row*dim2,dim2);
	}
}

void forward(const double* const a, const double* const p, const double* const b, double* const alpha,  const int * const y, double* const ct, const int N, const int K, const int T){

	double ct0 = 0.0;
	//compute alpha(0) and scaling factor for t = 0
	int y0 = y[0];
	for(int s = 0; s < N; s++){
		double alphas = p[s] * b[s*K + y0];
		ct0 += alphas;
		alpha[s] = alphas;
	}
	
	ct0 = 1.0 / ct0;

	//scale alpha(0)
	for(int s = 0; s < N; s++){
		alpha[s] *= ct0;
	}
	ct[0] = ct0;

	for(int t = 1; t < T; t++){
		double ctt = 0.0;	
		const int yt = y[t];	
		//ct[t]=0.0;
		for(int s = 0; s<N; s++){// s=new_state
			double alphatNs = 0;
			//alpha[s*T + t] = 0;
			for(int j = 0; j < N; j++){//j=old_states
				alphatNs += alpha[(t-1)*N + j] * a[j*N + s];
			}

			alphatNs *= b[s*K + yt];
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

	return;
}

void backward(const double* const a, const double* const b, double* const beta, const int * const y, const double * const ct, const int N, const int K, const int T ){
	
	double ctT1 = ct[T-1];	
	for(int s = 0; s < N; s++){
		beta[(T-1)*N + s] = /* 1* */ctT1;
	}

	for(int t = T-1; t > 0; t--){
		const int yt =y[t];
		const double ctt1 = ct[t-1];
		for(int s = 0; s < N; s++){//s=older state
			double betat1Ns = 0;
       			//beta[(t-1)*N+s] = 0.;
			for(int j = 0; j < N; j++){//j=newer state
				betat1Ns += beta[t*N + j ] * a[s*N + j] * b[j*K + yt];
			}
			beta[(t-1)*N + s] = ctt1*betat1Ns;
		}
	}
	return;
}

void update(double* const a, double* const p, double* const b, const double* const alpha, const double* const beta, double* const gamma, double* const xi, const int* const y, const double* const ct,const int N, const int K, const int T){


	double xi_sum, gamma_sum_numerator, gamma_sum_denominator;

	for(int t = 0; t < T; t++){
		for(int s = 0; s < N; s++){ // s old state
			gamma[t*N + s] = alpha[t*N + s] * beta[t*N + s];
		}
	}

	for(int t = 1; t < T; t++){
		const int yt = y[t];
		for(int s = 0; s < N; s++){
			const double alphat1Ns = alpha[(t-1)*N + s] ;
			for(int j = 0; j < N; j++){ // j new state
				xi[((t-1) * N + s) * N + j] = alphat1Ns * a[s*N + j] * beta[t*N + j] * b[j*K + yt]; 
			}
		}
	}

	//to save T divisions (in if y[t] == v block)
	double* inv_ct = (double*) malloc(T * sizeof(double));

	const double ct0div = 1/ct[0];
	inv_ct[0] = ct0div;

	for(int s = 0; s < N; s++){
		// new pi
		//XXX the next line is not in the r library hmm.
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

		gamma_sum_denominator += gamma[(T-1)*N + s]/ct[T-1];
		const double gamma_sum_denominator_div = 1/gamma_sum_denominator ;
		for(int v = 0; v < K; v++){
			gamma_sum_numerator = 0.;
			for(int t = 0; t < T; t++){//why 1 indented => better?
				if(y[t] == v){
					gamma_sum_numerator += gamma[t*N + s]/ct[t];
				}
			}
			// new emmision matrix
			b[s*K + v] = gamma_sum_numerator * gamma_sum_denominator_div;
		}
	}

	return;
}

int finished(const double* const alpha,const double* const beta, const double* const ct, double* const l,const int N,const int T){

	//log likelihood
	double oldLogLikelihood=*l;

	double newLogLikelihood = 0.0;
	//evidence with alpha only:

	for(int time = 0; time < T; time++){
		newLogLikelihood -= log10(ct[time]);
	}
	
	*l=newLogLikelihood;

	return (newLogLikelihood-oldLogLikelihood)<EPSILON;
	
}


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
	double* emissionMatrix = (double*) malloc(hiddenStates*differentObservables*sizeof(double));
	double* emissionMatrixSafe = (double*) malloc(hiddenStates*differentObservables*sizeof(double));

	//init state distribution
	double* stateProb  = (double*) malloc(hiddenStates * sizeof(double));
	double* stateProbSafe  = (double*) malloc(hiddenStates * sizeof(double));

	double* alpha = (double*) malloc( T *hiddenStates * sizeof(double));
	double* beta = (double*) malloc(T *hiddenStates *  sizeof(double));
	double* gamma = (double*) malloc(T * hiddenStates * sizeof(double));
	double* xi = (double*) malloc((T-1) * hiddenStates * hiddenStates *  sizeof(double)); 


	double* ct = (double*) malloc(T*sizeof(double));
    
    //to save T divisions (in if y[t] == v block)
    double* inv_ct = (double*) malloc(T * sizeof(double));	


	//heatup needs some data.
	//makeMatrix(hiddenStates, hiddenStates, transitionMatrix);
	//makeMatrix(hiddenStates, differentObservables, emissionMatrix);
	//makeProbabilities(stateProb,hiddenStates);
	//heatup(transitionMatrix,stateProb,emissionMatrix,observations,hiddenStates,differentObservables,T);
	
        int steps=0;
        int finishing;
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
		makeObservations(hiddenStates, differentObservables, groundInitialState, groundTransitionMatrix,groundEmissionMatrix,T, observations); 
		//XXX Ist es notwendig nach jedem run neue observations zu machen?

		//only needed for testing
		//write_init(transitionMatrix, emissionMatrix, observations, stateProb, hiddenStates, differentObservables, T);
        
		//XXX start after makeMatrix
        	steps=0;
		start = start_tsc();

	
		do{
	        double ct0 = 0.0;
	        //compute alpha(0) and scaling factor for t = 0
	        int y0 = observations[0];
	        for(int s = 0; s < hiddenStates; s++){
		        double alphas = stateProb[s] * emissionMatrix[s*differentObservables + y0];
		        ct0 += alphas;
		        alpha[s] = alphas;
	        }
	        
	        ct0 = 1.0 / ct0;

	        //scale alpha(0)
	        for(int s = 0; s < hiddenStates; s++){
		        alpha[s] *= ct0;
	        }
	        ct[0] = ct0;

	        for(int t = 1; t < T; t++){
		        double ctt = 0.0;	
		        const int yt = observations[t];	
		        for(int s = 0; s<hiddenStates; s++){// s=new_state
			        double alphatNs = 0;
			        for(int j = 0; j < hiddenStates; j++){//j=old_states
				        alphatNs += alpha[(t-1)*hiddenStates + j] * transitionMatrix[j*hiddenStates + s];
			        }

			        alphatNs *= emissionMatrix[s*differentObservables + yt];
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
            double ctT1 = ct[T-1];	
            for(int s = 0; s < hiddenStates; s++){
                beta[(T-1)*hiddenStates + s] = /* 1* */ctT1;
            }

            for(int t = T-1; t > 0; t--){
                const int yt =observations[t];
                const double ctt1 = ct[t-1];
                for(int s = 0; s < hiddenStates; s++){//s=older state
	                double betat1Ns = 0;
	                for(int j = 0; j < hiddenStates; j++){//j=newer state
		                betat1Ns += beta[t*hiddenStates + j ] * transitionMatrix[s*hiddenStates + j] * emissionMatrix[j*differentObservables + yt];
	                }
	                beta[(t-1)*hiddenStates + s] = ctt1*betat1Ns;
                }
            }



            double xi_sum, gamma_sum_numerator, gamma_sum_denominator;

            for(int t = 0; t < T; t++){
                for(int s = 0; s < hiddenStates; s++){ // s old state
                    gamma[t*hiddenStates + s] = alpha[t*hiddenStates + s] * beta[t*hiddenStates + s];
                }
            }

            for(int t = 1; t < T; t++){
                const int yt = observations[t];
                for(int s = 0; s < hiddenStates; s++){
                    const double alphat1Ns = alpha[(t-1)*hiddenStates + s] ;
                    for(int j = 0; j < hiddenStates; j++){ // j new state
	                    xi[((t-1) * hiddenStates + s) * hiddenStates + j] = alphat1Ns * transitionMatrix[s*hiddenStates + j] * beta[t*hiddenStates + j] * emissionMatrix[j*differentObservables + yt]; 
                    }
                }
            }

            const double ct0div = 1/ct[0];
            inv_ct[0] = ct0div;

            for(int s = 0; s < hiddenStates; s++){
                // new pi
                //XXX the next line is not in the r library hmm.
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

                gamma_sum_denominator += gamma[(T-1)*hiddenStates + s]/ct[T-1];
                const double gamma_sum_denominator_div = 1/gamma_sum_denominator ;
                for(int v = 0; v < differentObservables; v++){
                    gamma_sum_numerator = 0.;
                    for(int t = 0; t < T; t++){//why 1 indented => better?
	                    if(observations[t] == v){
		                    gamma_sum_numerator += gamma[t*hiddenStates + s]/ct[t];
	                    }
                    }
                    // new emmision matrix
                    emissionMatrix[s*differentObservables + v] = gamma_sum_numerator * gamma_sum_denominator_div;
                }
            }
            //log likelihood
	        double oldLogLikelihood=logLikelihood;

	        double newLogLikelihood = 0.0;
	        //evidence with alpha only:

	        for(int time = 0; time < T; time++){
		        newLogLikelihood -= log10(ct[time]);
	        }
	        
	        logLikelihood=newLogLikelihood;
            finishing = (newLogLikelihood-oldLogLikelihood)<EPSILON;    		
            steps+=1;
		}while (steps<maxSteps && !finishing);

		cycles = stop_tsc(start);
        cycles = cycles/steps;


        tested_implementation(hiddenStates, differentObservables, T, transitionMatrixSafe, emissionMatrixSafe, stateProbSafe, observations);
		if (similar(transitionMatrixSafe,transitionMatrix,hiddenStates,hiddenStates) && similar(emissionMatrixSafe,emissionMatrix,differentObservables,hiddenStates)){
			runs[run]=cycles;
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
    free(inv_ct);
    free(transitionMatrixSafe);
	free(emissionMatrixSafe);
    free(stateProbSafe);
	return 0; 
} 
