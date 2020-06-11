#ifndef UTIL_FILE_
#define UTIL_FILE_

inline void _flush_cache(volatile unsigned char* buf,const int BUFSIZE){
    for(unsigned int i = 0; i < BUFSIZE; ++i){
        buf[i] += i;
    }
}

void transpose(double* a, const int rows, const int cols);

int compare_doubles (const void *a, const void *b);

int chooseOf(const int choices, const double* const probArray);

void makeObservations(const int hiddenStates, const int differentObservables, const int groundInitialState, const double* const groundTransitionMatrix, const double* const groundEmissionMatrix, const int T, int* const observations);

void makeProbabilities(double* const probabilities, const int options);

void makeMatrix(const int dim1,const int dim2, double* const matrix);

int finished(const double* const ct, double* const l,const int N,const int T,const int EPSILON);

int similar(const double * const a, const double * const b , const int N, const int M, const int DELTA);

#endif
