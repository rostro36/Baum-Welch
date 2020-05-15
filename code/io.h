#ifndef IO_FILE_
#define IO_FILE_


void write_matrix_file(const double * const a, const int R, const int C, char* filename);
void write_vector_file(const double* const v, const int R, char* filename);
void write_matrix_file_int(const int * const a, const int R, const int C, char* filename);
void write_vector_file_int(const int* const v, const int R, char* filename);
void read_matrix_file(double * const a, const int R, const int C, char* filename);
void read_vector_file(double* const v, const int R, char* filename);
void read_matrix_file_int(int * const a, const int R, const int C, char* filename);
void read_vector_file_int(int* const v, const int R, char* filename);
void print_matrix(const double * const  a, const int R, const int C);
void print_vector(const double * const a, const int L);
void print_vector_int(const int * const a, const int L);
void write_all(const double * const groundTransitionMatrix,
		const double * const groundEmissionMatrix,
		const double * const transitionMatrix,
		const double * const emissionMatrix,
		const int * const observations,
		const double * const stateProb,
		const double * const alpha,
		const double * const beta,
		const double * const gamma,
		const double * const xi,
		const int hiddenStates,
		const int differentObservables,
		const int T);	
void write_init(const double * const transitionMatrix,
		const double * const emissionMatrix,
		const int * const observations,
		const double * const stateProb,
		const int hiddenStates,
		const int differentObservables,
		const int T);
void write_result(const double * const transitionMatrix,
		const double * const emissionMatrix,
		const int * const observations,
		const double * const stateProb,
		const int const steps,
		const int hiddenStates,
		const int differentObservables,
		const int T);
#endif
