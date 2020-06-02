#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <math.h>
#include <float.h> //for DOUBL_MAX
#include "tsc_x86.h"

#include "io.h"
#include "tested.h"

#include <immintrin.h>

#define EPSILON 1e-4
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
myInt64 bw(double* const transitionMatrix, double* const emissionMatrix, double* const stateProb, const int* const observations, double * const gamma_sum, double* const gamma_T,double* const a_new,double* const b_new, double* const ct, const int hiddenStates, const int differentObservables, const int T,double* beta, double* beta_new, double* alpha, double* ab,double* beta_workingset,double* reduction){

   		double disparance;
   	
        	double logLikelihood=-DBL_MAX; //Took down here.

		//only needed for testing with R
		//write_init(transitionMatrix, emissionMatrix, observations, stateProb, hiddenStates, differentObservables, T);
        
        	int steps=1;
		myInt64 start = start_tsc();
        
                      	
                __m256d one = _mm256_set1_pd(1.0);
   		
   		for(int by = 0; by < hiddenStates; by+=4){
	
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
	
	

		
		//compute alpha(0) and scaling factor for t = 0
		int y0 = observations[0];
		__m256d ct0_vec = _mm256_setzero_pd();
		for(int s = 0; s < hiddenStates; s+=4){
					
			/*
			__m256d alphas=_mm256_load_pd(stateProb+s);
			__m256d emissionMatr=_mm256_load_pd(emissionMatrix+y0*hiddenStates+s);
			alphas=_mm256_mul_pd(alphas,emissionMatr);
			_mm256_store_pd(alpha+s,alphas);
			ctt+=alpha[s]+alpha[s+1]+alpha[s+2]+alpha[s+3];
				
			*/
			__m256d stateProb_vec = _mm256_load_pd(stateProb +s);
			__m256d emission_vec = _mm256_load_pd(emissionMatrix +y0*hiddenStates +s);
			__m256d alphas_vec = _mm256_mul_pd(stateProb_vec, emission_vec);
			
			//ct0_vec = _mm256_add_pd(ct0_vec, alphas_vec);
			
			ct0_vec = _mm256_fmadd_pd(stateProb_vec,emission_vec, ct0_vec);
			_mm256_store_pd(alpha+s,alphas_vec);
					
			/* 	
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
		  	      alpha[s+3] = alphas3; */
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
	        
	        
	        //ctt = 1.0 / ctt;
		//__m256d scalingFactor= _mm256_set1_pd(ctt);
	        //scale alpha(0)
	        for(int s = 0; s < hiddenStates; s+=4){
			__m256d alphas=_mm256_load_pd(alpha+s);
			__m256d alphas_mul=_mm256_mul_pd(alphas,ct0_vec_div);
			_mm256_store_pd(alpha+s,alphas_mul);
		      	/* alpha[s] *= ctt;
		      	alpha[s+1] *= ctt;
		      	alpha[s+2] *= ctt;
		      	alpha[s+3] *= ctt; */
	        }
	        
	        
	
	

		for(int t = 1; t < T-1; t++){
			//double ctt = 0.0;	
			__m256d ctt_vec = _mm256_setzero_pd();
			const int yt = observations[t];	
			for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				/*
				double alphatNs0 = 0;
				double alphatNs1 = 0;
				double alphatNs2 = 0;
				double alphatNs3 = 0;
				*/
				
				__m256d alphatNs0 = _mm256_setzero_pd();
				__m256d alphatNs1 = _mm256_setzero_pd();
				__m256d alphatNs2 = _mm256_setzero_pd();
				__m256d alphatNs3 = _mm256_setzero_pd();
			
			
				for(int j = 0; j < hiddenStates; j+=4){//j=old_states
					__m256d alphaFactor=_mm256_load_pd(alpha+(t-1)*hiddenStates+j);
				
					__m256d transition0=_mm256_load_pd(transitionMatrix+(s)*hiddenStates+j);
					__m256d transition1=_mm256_load_pd(transitionMatrix+(s+1)*hiddenStates+j);
					__m256d transition2=_mm256_load_pd(transitionMatrix+(s+2)*hiddenStates+j);
					__m256d transition3=_mm256_load_pd(transitionMatrix+(s+3)*hiddenStates+j);
				
				
					/*
					__m256d alphat0=_mm256_mul_pd(alphaFactor,transition0);
					__m256d alphat1=_mm256_mul_pd(alphaFactor,transition1);
					__m256d alphat2=_mm256_mul_pd(alphaFactor,transition2);
					__m256d alphat3=_mm256_mul_pd(alphaFactor,transition3);
											
					alphat0=_mm256_hadd_pd(alphat0,alphat0);
					alphat1=_mm256_hadd_pd(alphat1,alphat1);
					alphat2=_mm256_hadd_pd(alphat2,alphat2);
					alphat3=_mm256_hadd_pd(alphat3,alphat3);
					
					alphatNs0+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat0,1))+_mm256_cvtsd_f64(alphat0);
					alphatNs1+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat1,1))+_mm256_cvtsd_f64(alphat1);
					alphatNs2+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat2,1))+_mm256_cvtsd_f64(alphat2);
					alphatNs3+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat3,1))+_mm256_cvtsd_f64(alphat3);
					*/
					
					/* double alphaFactor0 = alpha[(t-1)*hiddenStates + j];
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
					alphatNs3 += alphaFactor3 * transitionMatrix[(s+3)*hiddenStates + j+3]; */
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
				
				/*
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
				*/
			}
			
			
			__m256d perm = _mm256_permute2f128_pd(ctt_vec,ctt_vec,0b00000011);

			__m256d shuffle1 = _mm256_shuffle_pd(ctt_vec, perm, 0b0101);

			__m256d shuffle2 = _mm256_shuffle_pd(perm, ctt_vec, 0b0101);
		
			__m256d ctt_vec_add = _mm256_add_pd(ctt_vec, perm);

			__m256d ctt_temp = _mm256_add_pd(shuffle1, shuffle2);

			__m256d ctt_vec_tot = _mm256_add_pd(ctt_vec_add, ctt_temp);

			__m256d ctt_vec_div = _mm256_div_pd(one,ctt_vec_tot);
		
      			_mm256_storeu_pd(ct + t,ctt_vec_div); 
			
			
			
			//scaling factor for t 
			//ctt = 1.0 / ctt;
			//scalingFactor=_mm256_set1_pd(ctt);
			//scale alpha(t)
			for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				__m256d alphas=_mm256_load_pd(alpha+t*hiddenStates+s);
				__m256d alphas_mul=_mm256_mul_pd(alphas,ctt_vec_div);
				_mm256_store_pd(alpha+t*hiddenStates+s,alphas_mul);
				/* alpha[t*hiddenStates+s] *= ctt;
				alpha[t*hiddenStates+s+1] *= ctt;
				alpha[t*hiddenStates+s+2] *= ctt;
				alpha[t*hiddenStates+s+3] *= ctt; */
			}
			//ct[t] = ctt;
		}



		
		
		
		
		
		
		
		
		
		
		
		
			//compute alpha(T-1)

			int yt = observations[T-1];	
			__m256d ctt_vec = _mm256_setzero_pd();
			for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				/*
				double alphatNs0 = 0;
				double alphatNs1 = 0;
				double alphatNs2 = 0;
				double alphatNs3 = 0;
				*/
				__m256d alphatNs0_vec = _mm256_setzero_pd();
				__m256d alphatNs1_vec = _mm256_setzero_pd();
				__m256d alphatNs2_vec = _mm256_setzero_pd();
				__m256d alphatNs3_vec = _mm256_setzero_pd();
				 
				for(int j = 0; j < hiddenStates; j+=4){//j=old_states
					__m256d alphaFactor=_mm256_load_pd(alpha+(T-2)*hiddenStates+j);
					
					__m256d transition0=_mm256_load_pd(transitionMatrix+(s)*hiddenStates+j);
					__m256d transition1=_mm256_load_pd(transitionMatrix+(s+1)*hiddenStates+j);
					__m256d transition2=_mm256_load_pd(transitionMatrix+(s+2)*hiddenStates+j);
					__m256d transition3=_mm256_load_pd(transitionMatrix+(s+3)*hiddenStates+j);
					
					/*
					__m256d alphat0=_mm256_mul_pd(alphaFactor,transition0);
					__m256d alphat1=_mm256_mul_pd(alphaFactor,transition1);
					__m256d alphat2=_mm256_mul_pd(alphaFactor,transition2);
					__m256d alphat3=_mm256_mul_pd(alphaFactor,transition3);
					
					alphat0=_mm256_hadd_pd(alphat0,alphat0);
					alphat1=_mm256_hadd_pd(alphat1,alphat1);
					alphat2=_mm256_hadd_pd(alphat2,alphat2);
					alphat3=_mm256_hadd_pd(alphat3,alphat3);
					
					alphatNs0+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat0,1))+_mm256_cvtsd_f64(alphat0);
					alphatNs1+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat1,1))+_mm256_cvtsd_f64(alphat1);
					alphatNs2+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat2,1))+_mm256_cvtsd_f64(alphat2);
					alphatNs3+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat3,1))+_mm256_cvtsd_f64(alphat3);
					*/

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
				
			
				/*
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
				*/
			}
			
						
			//scaling factor for T-1
			
			__m256d perm1 = _mm256_permute2f128_pd(ctt_vec,ctt_vec,0b00000011);
	
			__m256d shuffle11 = _mm256_shuffle_pd(ctt_vec, perm1, 0b0101);

			__m256d shuffle21 = _mm256_shuffle_pd(perm1, ctt_vec, 0b0101);
		
			__m256d ctt_vec_add = _mm256_add_pd(ctt_vec, perm1);

			__m256d ctt_temp = _mm256_add_pd(shuffle11, shuffle21);

			__m256d ctt_vec_tot = _mm256_add_pd(ctt_vec_add, ctt_temp);

			__m256d ctt_vec_div = _mm256_div_pd(one,ctt_vec_tot);
		
	      		_mm256_storeu_pd(ct + (T-1),ctt_vec_div); 
	      			
	      			
			
			//scale alpha(t)
			for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				__m256d alphaT1Ns=_mm256_load_pd(alpha+(T-1)*hiddenStates+s);
				__m256d alphaT1Ns_mul=_mm256_mul_pd(alphaT1Ns,ctt_vec_div);
				_mm256_store_pd(alpha+(T-1)*hiddenStates+s,alphaT1Ns_mul);
				_mm256_store_pd(gamma_T+s,alphaT1Ns_mul);
				
				/* //s
				double alphaT1Ns0 = alpha[(T-1) * hiddenStates + s]*ctt;
				alpha[(T-1)*hiddenStates + s] = alphaT1Ns0;
				gamma_T[s] = alphaT1Ns0;
				//s+1
				double alphaT1Ns1 = alpha[(T-1) * hiddenStates + s+1]*ctt;
				alpha[(T-1)*hiddenStates + s+1] = alphaT1Ns1;
				gamma_T[s+1] = alphaT1Ns1;
				//s+2
				double alphaT1Ns2 = alpha[(T-1) * hiddenStates + s+2]*ctt;
				alpha[(T-1)*hiddenStates + s+2] = alphaT1Ns2;
				gamma_T[s+2] = alphaT1Ns2;
				//s+3
				double alphaT1Ns3 = alpha[(T-1) * hiddenStates + s+3]*ctt;
				alpha[(T-1)*hiddenStates + s+3] = alphaT1Ns3;
				gamma_T[s+3] = alphaT1Ns3; 
				 */
			}
			//ct[T-1] = ctt;

			
			
		

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
	
			//const double ctt = ct[t-1];
			__m256d ctt_vec = _mm256_set1_pd(ct[t-1]);//_mm256_set_pd(ctt,ctt,ctt,ctt);
			const int yt1 = observations[t-1];
			for(int s = 0; s < hiddenStates ; s+=4){
			
				/*
				double alphat1Ns0 = alpha[(t-1)*hiddenStates + s];
				double alphat1Ns1 = alpha[(t-1)*hiddenStates + s+1];
				double alphat1Ns2 = alpha[(t-1)*hiddenStates + s+2];
				double alphat1Ns3 = alpha[(t-1)*hiddenStates + s+3];
				*/
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
					
				//XXX Better Reduction
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
        	


	

		//for(int i = 0; i < 10; i++){
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
				/* double gamma_Ts0 = gamma_T[s];
				//if you use real gamma you have to divide by ct[t-1]
				double gamma_sums0 = gamma_sum[s];
				double gamma_tot0 = gamma_Ts0 + gamma_sums0 ;
				gamma_T[s] = 1./gamma_tot0;
				gamma_sum[s] = 1./gamma_sums0;
		        	b_new[yt*hiddenStates+s]+=gamma_Ts0;
	   
				double gamma_Ts1 = gamma_T[s+1];
				//if you use real gamma you have to divide by ct[t-1]
				double gamma_sums1 = gamma_sum[s+1];
				double gamma_tot1 = gamma_Ts1 + gamma_sums1 ;
				gamma_T[s+1] = 1./gamma_tot1;
				gamma_sum[s+1] = 1./gamma_sums1;
		        	b_new[yt*hiddenStates+s+1]+=gamma_Ts1;
	   
				double gamma_Ts2 = gamma_T[s+2];
				//if you use real gamma you have to divide by ct[t-1]
				double gamma_sums2 = gamma_sum[s+2];
				double gamma_tot2 = gamma_Ts2 + gamma_sums2 ;
				gamma_T[s+2] = 1./gamma_tot2;
				gamma_sum[s+2] = 1./gamma_sums2;
		        	b_new[yt*hiddenStates+s+2]+=gamma_Ts2;
	   
				double gamma_Ts3 = gamma_T[s+3];
				//if you use real gamma you have to divide by ct[t-1]
				double gamma_sums3 = gamma_sum[s+3];
				double gamma_tot3 = gamma_Ts3 + gamma_sums3 ;
				gamma_T[s+3] = 1./gamma_tot3;
				gamma_sum[s+3] = 1./gamma_sums3;
		        	b_new[yt*hiddenStates+s+3]+=gamma_Ts3; */
	   
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
					/* emissionMatrix[v*hiddenStates + s] = b_new[v*hiddenStates + s] * gamma_T0;
					b_new[v*hiddenStates + s] = 0.0;
					
					emissionMatrix[v*hiddenStates + s+1] = b_new[v*hiddenStates + s+1] * gamma_T1;
					b_new[v*hiddenStates + s+1] = 0.0;
					
					emissionMatrix[v*hiddenStates + s+2] = b_new[v*hiddenStates + s+2] * gamma_T2;
					b_new[v*hiddenStates + s+2] = 0.0;
					
					emissionMatrix[v*hiddenStates + s+3] = b_new[v*hiddenStates + s+3] * gamma_T3;
					b_new[v*hiddenStates + s+3] = 0.0; */
					
					
					/*
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
		
					*/
					
		
				}
			}

			//FORWARD

			//Transpose a_new. Note that it is not necessary to transpose matrix a.
			//Transpose transitionMatrix
			    
			for(int by = 0; by < hiddenStates; by+=4){
	
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
			
			
			
			
			
			
			
			
			
			
			
			
			
			//double ctt = 0.0;

			//compute alpha(0) and scaling factor for t = 0
			int y0 = observations[0];
		  	__m256d ct0_vec = _mm256_setzero_pd();
		        for(int s = 0; s < hiddenStates; s+=4){
					
					/*
					__m256d alphas=_mm256_load_pd(stateProb+s);
					__m256d emissionMatr=_mm256_load_pd(emissionMatrix+y0*hiddenStates+s);
					alphas=_mm256_mul_pd(alphas,emissionMatr);
					_mm256_store_pd(alpha+s,alphas);
					ctt+=alpha[s]+alpha[s+1]+alpha[s+2]+alpha[s+3];
					
					*/
					__m256d stateProb_vec = _mm256_load_pd(stateProb +s);
					__m256d emission_vec = _mm256_load_pd(emissionMatrix +y0*hiddenStates +s);
					__m256d alphas_vec = _mm256_mul_pd(stateProb_vec, emission_vec);
			
					//ct0_vec = _mm256_add_pd(ct0_vec, alphas_vec);
			
					ct0_vec = _mm256_fmadd_pd(stateProb_vec,emission_vec, ct0_vec);
					_mm256_store_pd(alpha+s,alphas_vec);
					
				/* 	
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
		  	      alpha[s+3] = alphas3; */
	        	}	
	        	
	        		        	
	        	__m256d perm = _mm256_permute2f128_pd(ct0_vec,ct0_vec,0b00000011);
	
			__m256d shuffle1 = _mm256_shuffle_pd(ct0_vec, perm, 0b0101);

			__m256d shuffle2 = _mm256_shuffle_pd(perm, ct0_vec, 0b0101);
		
			__m256d ct0_vec_add = _mm256_add_pd(ct0_vec, perm);
			__m256d ct0_temp = _mm256_add_pd(shuffle1, shuffle2);
			__m256d ct0_vec_tot = _mm256_add_pd(ct0_vec_add, ct0_temp);
			__m256d ct0_vec_div = _mm256_div_pd(one,ct0_vec_tot);
			
	      		_mm256_storeu_pd(ct,ct0_vec_div);
	        
	        
	        	//ctt = 1.0 / ctt;
			//__m256d scalingFactor= _mm256_set1_pd(ctt);
	        	//scale alpha(0)
	        	for(int s = 0; s < hiddenStates; s+=4){
					__m256d alphas=_mm256_load_pd(alpha+s);
					__m256d alphas_mul=_mm256_mul_pd(alphas,ct0_vec_div);
					_mm256_store_pd(alpha+s,alphas_mul);
		        	/* alpha[s] *= ctt;
		        	alpha[s+1] *= ctt;
		        	alpha[s+2] *= ctt;
		        	alpha[s+3] *= ctt; */
	        	}
	
			//ct[0] = ctt;
			
			//Compute alpha(1) and scale transitionMatrix
			//ctt = 0.0;	
			yt = observations[1];	
			__m256d ctt_vec = _mm256_setzero_pd();
			for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				
				/*
				double alphatNs0 = 0;
				double alphatNs1 = 0;
				double alphatNs2 = 0;
				double alphatNs3 = 0;
				*/	
				__m256d alphatNs0 = _mm256_setzero_pd();
				__m256d alphatNs1 = _mm256_setzero_pd();
				__m256d alphatNs2 = _mm256_setzero_pd();
				__m256d alphatNs3 = _mm256_setzero_pd();
				
			
				for(int j = 0; j < hiddenStates; j+=4){//j=old_states
					__m256d gammaSum=_mm256_load_pd(gamma_sum+j);
					__m256d aNew0=_mm256_load_pd(a_new+s*hiddenStates+j);
					__m256d aNew1=_mm256_load_pd(a_new+(s+1)*hiddenStates+j);
					__m256d aNew2=_mm256_load_pd(a_new+(s+2)*hiddenStates+j);
					__m256d aNew3=_mm256_load_pd(a_new+(s+3)*hiddenStates+j);
					
					__m256d as0=_mm256_mul_pd(aNew0,gammaSum);
					__m256d as1=_mm256_mul_pd(aNew1,gammaSum);
					__m256d as2=_mm256_mul_pd(aNew2,gammaSum);
					__m256d as3=_mm256_mul_pd(aNew3,gammaSum);
					/* double alphaFactor0 = alpha[j];
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
					double as3Nj3 = a_new[(s+3)*hiddenStates+j+3] * gamma_sum3;  */
					__m256d zeroes = _mm256_setzero_pd();
					_mm256_store_pd(a_new+s*hiddenStates+j,zeroes);
					_mm256_store_pd(a_new+(s+1)*hiddenStates+j,zeroes);
					_mm256_store_pd(a_new+(s+2)*hiddenStates+j,zeroes);
					_mm256_store_pd(a_new+(s+3)*hiddenStates+j,zeroes);
					/* a_new[s*hiddenStates+j] = 0.0;
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
					a_new[(s+3)*hiddenStates+j+3] = 0.0; */
					
					
					
					_mm256_store_pd(transitionMatrix+(s)*hiddenStates+j,as0);
					_mm256_store_pd(transitionMatrix+(s+1)*hiddenStates+j,as1);
					_mm256_store_pd(transitionMatrix+(s+2)*hiddenStates+j,as2);
					_mm256_store_pd(transitionMatrix+(s+3)*hiddenStates+j,as3);
					/* transitionMatrix[s*hiddenStates + j] = as0Nj0;
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
					transitionMatrix[(s+3)*hiddenStates + j+3] = as3Nj3; */
					
					/*
					as0=_mm256_hadd_pd(as0,as0);
					as1=_mm256_hadd_pd(as1,as1);
					as2=_mm256_hadd_pd(as2,as2);
					as3=_mm256_hadd_pd(as3,as3);
					
					alphatNs0+=_mm_cvtsd_f64(_mm256_extractf128_pd(as0,1))+_mm256_cvtsd_f64(as0);
					alphatNs1+=_mm_cvtsd_f64(_mm256_extractf128_pd(as1,1))+_mm256_cvtsd_f64(as1);
					alphatNs2+=_mm_cvtsd_f64(_mm256_extractf128_pd(as2,1))+_mm256_cvtsd_f64(as2);
					alphatNs3+=_mm_cvtsd_f64(_mm256_extractf128_pd(as3,1))+_mm256_cvtsd_f64(as3);
					
					*/

					
					
					
					__m256d alphaFactor = _mm256_load_pd(alpha + j);
					
					alphatNs0 =_mm256_fmadd_pd(alphaFactor,as0,alphatNs0);
					alphatNs1 =_mm256_fmadd_pd(alphaFactor,as1,alphatNs1);
					alphatNs2 =_mm256_fmadd_pd(alphaFactor,as2,alphatNs2);
					alphatNs3 =_mm256_fmadd_pd(alphaFactor,as3,alphatNs3);

					
					/* alphatNs0 += alphaFactor0 * as0Nj0;
					alphatNs0 += alphaFactor1 * as0Nj1;
					alphatNs0 += alphaFactor2 * as0Nj2;
					alphatNs0 += alphaFactor3 * as0Nj3;
					
					alphatNs1 += alphaFactor0 * as1Nj0;
					alphatNs1 += alphaFactor1 * as1Nj1;
					alphatNs1 += alphaFactor2 * as1Nj2;
					alphatNs1 += alphaFactor3 * as1Nj3;
					
					alphatNs2 += alphaFactor0 * as2Nj0;
					alphatNs2 += alphaFactor1 * as2Nj1;
					alphatNs2 += alphaFactor2 * as2Nj1;
					alphatNs2 += alphaFactor3 * as2Nj3;
				
					alphatNs3 += alphaFactor0 * as3Nj0;
					alphatNs3 += alphaFactor1 * as3Nj1;
					alphatNs3 += alphaFactor2 * as3Nj2;
					alphatNs3 += alphaFactor3 * as3Nj3; */
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
				
				
				/*
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
				*/
				
			}
			

			
					
			__m256d perm1 = _mm256_permute2f128_pd(ctt_vec,ctt_vec,0b00000011);
	
			__m256d shuffle11 = _mm256_shuffle_pd(ctt_vec, perm1, 0b0101);

			__m256d shuffle21 = _mm256_shuffle_pd(perm1, ctt_vec, 0b0101);
		
			__m256d ctt_vec_add = _mm256_add_pd(ctt_vec, perm1);

			__m256d ctt_temp = _mm256_add_pd(shuffle11, shuffle21);

			__m256d ctt_vec_tot = _mm256_add_pd(ctt_vec_add, ctt_temp);

			__m256d ctt_vec_div = _mm256_div_pd(one,ctt_vec_tot);
		
	      		_mm256_storeu_pd(ct + 1,ctt_vec_div);  
	      		
	        
	      		
			//scaling factor for t 
			//ctt = 1.0 / ctt;


			//scale alpha(t)
		        for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				__m256d alphas=_mm256_load_pd(alpha+hiddenStates+s);
				__m256d alphas_mul = _mm256_mul_pd(alphas,ctt_vec_div);
				_mm256_store_pd(alpha+hiddenStates+s,alphas_mul);
		    	    /* alpha[1*hiddenStates + s] *= ctt;
		    	    alpha[1*hiddenStates + s+1] *= ctt;
		    	    alpha[1*hiddenStates + s+2] *= ctt;
		    	    alpha[1*hiddenStates + s+3] *= ctt; */
	        	}
			//ct[1] = ctt;















			for(int t = 2; t < T-1; t++){
				//double ctt = 0.0;	
				__m256d ctt_vec = _mm256_setzero_pd();
				const int yt = observations[t];	
				for(int s = 0; s<hiddenStates; s+=4){// s=new_state
					/*
					double alphatNs0 = 0;
					double alphatNs1 = 0;
					double alphatNs2 = 0;
					double alphatNs3 = 0;
					*/
					
					__m256d alphatNs0 = _mm256_setzero_pd();
					__m256d alphatNs1 = _mm256_setzero_pd();
					__m256d alphatNs2 = _mm256_setzero_pd();
					__m256d alphatNs3 = _mm256_setzero_pd();
				
				
					for(int j = 0; j < hiddenStates; j+=4){//j=old_states
						__m256d alphaFactor=_mm256_load_pd(alpha+(t-1)*hiddenStates+j);
					
						__m256d transition0=_mm256_load_pd(transitionMatrix+(s)*hiddenStates+j);
						__m256d transition1=_mm256_load_pd(transitionMatrix+(s+1)*hiddenStates+j);
						__m256d transition2=_mm256_load_pd(transitionMatrix+(s+2)*hiddenStates+j);
						__m256d transition3=_mm256_load_pd(transitionMatrix+(s+3)*hiddenStates+j);
					
					
						/*
						__m256d alphat0=_mm256_mul_pd(alphaFactor,transition0);
						__m256d alphat1=_mm256_mul_pd(alphaFactor,transition1);
						__m256d alphat2=_mm256_mul_pd(alphaFactor,transition2);
						__m256d alphat3=_mm256_mul_pd(alphaFactor,transition3);
	
										
						alphat0=_mm256_hadd_pd(alphat0,alphat0);
						alphat1=_mm256_hadd_pd(alphat1,alphat1);
						alphat2=_mm256_hadd_pd(alphat2,alphat2);
						alphat3=_mm256_hadd_pd(alphat3,alphat3);
						
						alphatNs0+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat0,1))+_mm256_cvtsd_f64(alphat0);
						alphatNs1+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat1,1))+_mm256_cvtsd_f64(alphat1);
						alphatNs2+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat2,1))+_mm256_cvtsd_f64(alphat2);
						alphatNs3+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat3,1))+_mm256_cvtsd_f64(alphat3);
						*/
						
						/* double alphaFactor0 = alpha[(t-1)*hiddenStates + j];
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
						alphatNs3 += alphaFactor3 * transitionMatrix[(s+3)*hiddenStates + j+3]; */

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
				
					/*
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
					*/
				}
			
			
				__m256d perm = _mm256_permute2f128_pd(ctt_vec,ctt_vec,0b00000011);
	
				__m256d shuffle1 = _mm256_shuffle_pd(ctt_vec, perm, 0b0101);

				__m256d shuffle2 = _mm256_shuffle_pd(perm, ctt_vec, 0b0101);
		
				__m256d ctt_vec_add = _mm256_add_pd(ctt_vec, perm);

				__m256d ctt_temp = _mm256_add_pd(shuffle1, shuffle2);

				__m256d ctt_vec_tot = _mm256_add_pd(ctt_vec_add, ctt_temp);

				__m256d ctt_vec_div = _mm256_div_pd(one,ctt_vec_tot);
		
	      			_mm256_storeu_pd(ct + t,ctt_vec_div);        
			
			
				//scaling factor for t 
				//ctt = 1.0 / ctt;
				//scalingFactor=_mm256_set1_pd(ctt);
				//scale alpha(t)
				for(int s = 0; s<hiddenStates; s+=4){// s=new_state
					__m256d alphas=_mm256_load_pd(alpha+t*hiddenStates+s);
					__m256d alphas_mul=_mm256_mul_pd(alphas,ctt_vec_div);
					_mm256_store_pd(alpha+t*hiddenStates+s,alphas_mul);
					/* alpha[t*hiddenStates+s] *= ctt;
					alpha[t*hiddenStates+s+1] *= ctt;
					alpha[t*hiddenStates+s+2] *= ctt;
					alpha[t*hiddenStates+s+3] *= ctt; */
				}
				//ct[t] = ctt;

			}
			
			//compute alpha(T-1)

			yt = observations[T-1];	
			__m256d ctT_vec = _mm256_setzero_pd();
			for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				/*
				double alphatNs0 = 0;
				double alphatNs1 = 0;
				double alphatNs2 = 0;
				double alphatNs3 = 0;
				*/
				__m256d alphatNs0_vec = _mm256_setzero_pd();
				__m256d alphatNs1_vec = _mm256_setzero_pd();
				__m256d alphatNs2_vec = _mm256_setzero_pd();
				__m256d alphatNs3_vec = _mm256_setzero_pd();
				 
				for(int j = 0; j < hiddenStates; j+=4){//j=old_states
					__m256d alphaFactor=_mm256_load_pd(alpha+(T-2)*hiddenStates+j);
					
					__m256d transition0=_mm256_load_pd(transitionMatrix+(s)*hiddenStates+j);
					__m256d transition1=_mm256_load_pd(transitionMatrix+(s+1)*hiddenStates+j);
					__m256d transition2=_mm256_load_pd(transitionMatrix+(s+2)*hiddenStates+j);
					__m256d transition3=_mm256_load_pd(transitionMatrix+(s+3)*hiddenStates+j);
					
					/*
					__m256d alphat0=_mm256_mul_pd(alphaFactor,transition0);
					__m256d alphat1=_mm256_mul_pd(alphaFactor,transition1);
					__m256d alphat2=_mm256_mul_pd(alphaFactor,transition2);
					__m256d alphat3=_mm256_mul_pd(alphaFactor,transition3);
					
					alphat0=_mm256_hadd_pd(alphat0,alphat0);
					alphat1=_mm256_hadd_pd(alphat1,alphat1);
					alphat2=_mm256_hadd_pd(alphat2,alphat2);
					alphat3=_mm256_hadd_pd(alphat3,alphat3);
					
					alphatNs0+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat0,1))+_mm256_cvtsd_f64(alphat0);
					alphatNs1+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat1,1))+_mm256_cvtsd_f64(alphat1);
					alphatNs2+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat2,1))+_mm256_cvtsd_f64(alphat2);
					alphatNs3+=_mm_cvtsd_f64(_mm256_extractf128_pd(alphat3,1))+_mm256_cvtsd_f64(alphat3);
					*/

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
				
				
				/*
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
				*/
			}
			
						
			//scaling factor for T-1
			
			__m256d perm2 = _mm256_permute2f128_pd(ctT_vec,ctT_vec,0b00000011);
	
			__m256d shuffle12 = _mm256_shuffle_pd(ctT_vec, perm2, 0b0101);

			__m256d shuffle22 = _mm256_shuffle_pd(perm2, ctT_vec, 0b0101);
		
			__m256d ctT_vec_add = _mm256_add_pd(ctT_vec, perm2);

			__m256d ctT_temp = _mm256_add_pd(shuffle12, shuffle22);

			__m256d ctT_vec_tot = _mm256_add_pd(ctT_vec_add, ctT_temp);

			__m256d ctT_vec_div = _mm256_div_pd(one,ctT_vec_tot);
		
	      		_mm256_storeu_pd(ct + (T-1),ctT_vec_div); 
	      			        	    
			
			//scale alpha(t)
			for(int s = 0; s<hiddenStates; s+=4){// s=new_state
				__m256d alphaT1Ns=_mm256_load_pd(alpha+(T-1)*hiddenStates+s);
				__m256d alphaT1Ns_mul=_mm256_mul_pd(alphaT1Ns,ctT_vec_div);
				_mm256_store_pd(alpha+(T-1)*hiddenStates+s,alphaT1Ns_mul);
				_mm256_store_pd(gamma_T+s,alphaT1Ns_mul);
				
				/* //s
				double alphaT1Ns0 = alpha[(T-1) * hiddenStates + s]*ctt;
				alpha[(T-1)*hiddenStates + s] = alphaT1Ns0;
				gamma_T[s] = alphaT1Ns0;
				//s+1
				double alphaT1Ns1 = alpha[(T-1) * hiddenStates + s+1]*ctt;
				alpha[(T-1)*hiddenStates + s+1] = alphaT1Ns1;
				gamma_T[s+1] = alphaT1Ns1;
				//s+2
				double alphaT1Ns2 = alpha[(T-1) * hiddenStates + s+2]*ctt;
				alpha[(T-1)*hiddenStates + s+2] = alphaT1Ns2;
				gamma_T[s+2] = alphaT1Ns2;
				//s+3
				double alphaT1Ns3 = alpha[(T-1) * hiddenStates + s+3]*ctt;
				alpha[(T-1)*hiddenStates + s+3] = alphaT1Ns3;
				gamma_T[s+3] = alphaT1Ns3; 
				 */
			}
			//ct[T-1] = ctt;

			
			
			
			
			
			
			
						
			
			
			//Transpose transitionMatrix
			for(int by = 0; by < hiddenStates; by+=4){
	
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
		
				//const double ctt = ct[t-1];
				__m256d ctt_vec = _mm256_set1_pd(ct[t-1]);//_mm256_set_pd(ctt,ctt,ctt,ctt);
				const int yt1 = observations[t-1];
				for(int s = 0; s < hiddenStates ; s+=4){

					/*
					double alphat1Ns0 = alpha[(t-1)*hiddenStates + s];
					double alphat1Ns1 = alpha[(t-1)*hiddenStates + s+1];
					double alphat1Ns2 = alpha[(t-1)*hiddenStates + s+2];
					double alphat1Ns3 = alpha[(t-1)*hiddenStates + s+3];
					*/
					
					
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
					
					/*
					//XXX PROBABLY BETTER REDUCTION
					_mm256_store_pd(reduction,beta_news0);
					beta_workingset[0] = reduction[0] + reduction[1] + reduction[2] + reduction[3] ;
					
					_mm256_store_pd(reduction,beta_news1);
					beta_workingset[1] = reduction[0] + reduction[1] + reduction[2] + reduction[3] ;
					
					_mm256_store_pd(reduction,beta_news2);
					beta_workingset[2] = reduction[0] + reduction[1] + reduction[2] + reduction[3] ;
					
					_mm256_store_pd(reduction,beta_news3);
					beta_workingset[3] = reduction[0] + reduction[1] + reduction[2] + reduction[3] ;
					*/
					
					//__m256d beta_news = _mm256_load_pd(beta_workingset);
					
						
					__m256d gamma_sum_vec = _mm256_load_pd(gamma_sum + s);
					__m256d b_new_vec = _mm256_load_pd(b_new +yt1*hiddenStates+ s);
					
					//XXX Better Reduction
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
        		
        	        //log likelihood
		        double oldLogLikelihood=logLikelihood;
	
		        double newLogLikelihood = 0.0;
		        //evidence with alpha only:
			
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
		

	myInt64 cycles = stop_tsc(start);
        cycles = cycles/steps;
        return cycles;
	




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
	const int cachegrind_runs = 1;
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
	double* groundTransitionMatrix = (double*) _mm_malloc(hiddenStates*hiddenStates*sizeof(double),32);
	double* groundEmissionMatrix = (double*) _mm_malloc(hiddenStates*differentObservables*sizeof(double),32);

	//set ground truth to some random values
	makeMatrix(hiddenStates, hiddenStates, groundTransitionMatrix);
	makeMatrix(hiddenStates, differentObservables, groundEmissionMatrix);
	int groundInitialState = rand()%hiddenStates;
	
	//the observations we made
	int* observations = (int*) _mm_malloc ( T * sizeof(int),32);
	makeObservations(hiddenStates, differentObservables, groundInitialState, groundTransitionMatrix,groundEmissionMatrix,T, observations);//??? added
	
	//the matrices which should approximate the ground truth
	double* transitionMatrix = (double*) _mm_malloc(hiddenStates*hiddenStates*sizeof(double),32);
	double* transitionMatrixSafe = (double*) _mm_malloc(hiddenStates*hiddenStates*sizeof(double),32);
	double* transitionMatrixTesting=(double*) _mm_malloc(hiddenStates*hiddenStates*sizeof(double),32);
	double* emissionMatrix = (double*) _mm_malloc(hiddenStates*differentObservables*sizeof(double),32);
	double* emissionMatrixSafe = (double*) _mm_malloc(hiddenStates*differentObservables*sizeof(double),32);
	double* emissionMatrixTesting=(double*) _mm_malloc(hiddenStates*differentObservables*sizeof(double),32);

	//init state distribution
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
	
	
	double* beta_workingset = (double*) _mm_malloc(4  * sizeof(double),32);
	double* reduction = (double*) _mm_malloc(4  * sizeof(double),32);
	
	
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
		
	        myInt64 cycles=bw(transitionMatrix, emissionMatrix, stateProb, observations, gamma_sum, gamma_T,a_new,b_new, ct, hiddenStates, differentObservables, T,beta, beta_new, alpha, ab,beta_workingset, reduction);

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

	transpose(emissionMatrix,differentObservables,hiddenStates);
	//emissionMatrix is not in state major order
	transpose(emissionMatrixTesting, differentObservables,hiddenStates);
        tested_implementation(hiddenStates, differentObservables, T, transitionMatrixTesting, emissionMatrixTesting, stateProbTesting, observations);
		

	if (!similar(transitionMatrixTesting,transitionMatrix,hiddenStates,hiddenStates) && similar(emissionMatrixTesting,emissionMatrix,differentObservables,hiddenStates)){
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
	_mm_free(beta_workingset);
	_mm_free(reduction);
			

	return 0; 
} 
