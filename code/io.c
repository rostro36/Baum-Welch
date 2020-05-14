#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>

#include "io.h"

void write_matrix_file(const double * const a, const int R, const int C, char* filename){

	FILE* fp = fopen(filename, "w+");
	for(int i = 0; i < R; i++){
		for(int j = 0; j <  C-1; j++){
			fprintf(fp, "%lf, ", a[i*C + j]);
		}
		fprintf(fp,"%lf \n", a[i*C + C-1]);
	}
	
	fclose(fp);

}

void write_vector_file(const double* const v, const int R, char* filename){
	write_matrix_file(v,R,1,filename);
}

void write_matrix_file_int(const int * const a, const int R, const int C, char* filename){

	FILE* fp = fopen(filename, "w+");
	for(int i = 0; i < R; i++){
		for(int j = 0; j <  C-1; j++){
			fprintf(fp, "%i, ", a[i*C + j]);
		}
		fprintf(fp,"%i \n", a[i*C + C-1]);
	}
	
	fclose(fp);

}

void write_vector_file_int(const int* const v, const int R, char* filename){
	write_matrix_file_int(v,R,1,filename);
}

void read_matrix_file(double * const a, const int R, const int C, char* filename){

	FILE* fp = fopen(filename, "r");
	char buffer[1024];

	for(int i = 0; i < R; i++){
		char * line = fgets(buffer,sizeof(buffer),fp);
		char * record = strtok(line," ,");
		for(int j = 0; j <  C; j++){
			a[i*C+j] = strtod(record,NULL);
			record = strtok(NULL," ,");
		}
	}

}

void read_vector_file(double* const v, const int R, char* filename){
	read_matrix_file(v,R,1,filename);
}

void read_matrix_file_int(int * const a, const int R, const int C, char* filename){

	FILE* fp = fopen(filename, "r");
	char buffer[1024];

	for(int i = 0; i < R; i++){
		char * line = fgets(buffer,sizeof(buffer),fp);
		char * record = strtok(line," ,");
		for(int j = 0; j <  C; j++){
			a[i*C+j] = strtod(record,NULL);
			record = strtok(NULL," ,");
		}
	}

}

void read_vector_file_int(int* const v, const int R, char* filename){
	read_matrix_file_int(v,R,1,filename);
}



//printing the data of an double array
//R is number of rows
//C is number of columns
void print_matrix(const double * const  a, const int R, const int C){
	for(int i =0; i < R; i++){
		for(int j = 0; j < C-1; j++){
			printf("%lf, ",a[i*C + j]);
		}
		printf("%lf \n", a[i*C + C-1]);
	}
	printf("\n");
}

//printing the data of a double vector
//L is lenght of vector
void print_vector(const double * const a, const int L){
	print_matrix(a,L,1);
}

//printing the data of a integer vector
//L is lenght of vector
void print_vector_int(const int * const a, const int L){
	for(int j = 0; j < L-1; j++){
		printf("%i, ",a[j]);
	}
	printf("%i \n", a[L-1]);
	
	printf("\n");
}


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
		const int T){


	char gtname[100]="test_matrices/groundTransitionMatrix.csv";
	write_matrix_file(groundTransitionMatrix,hiddenStates,hiddenStates,gtname);	

	char gename[100]="test_matrices/groundEmissionMatrix.csv";
	write_matrix_file(groundEmissionMatrix,hiddenStates,differentObservables,gename);	

	char tname[100]="test_matrices/transitionMatrix.csv";
	write_matrix_file(transitionMatrix,hiddenStates,hiddenStates,tname);	

	char ename[100]="test_matrices/emissionMatrix.csv";
	write_matrix_file(emissionMatrix,hiddenStates,differentObservables,ename);	

	char oname[100]="test_matrices/observations.csv";
	write_vector_file_int(observations,T,oname);	

	char pname[100]="test_matrices/stateProb.csv";
	write_vector_file(stateProb,hiddenStates,pname);	

	char aname[100]="test_matrices/alpha.csv";
	write_matrix_file(emissionMatrix,hiddenStates,T,aname);	

	char bname[100]="test_matrices/beta.csv";
	write_matrix_file(emissionMatrix,hiddenStates,T,bname);	

	char gname[100]="test_matrices/gamma.csv";
	write_matrix_file(emissionMatrix,hiddenStates,T,gname);	

	char xname[100]="test_matrices/xi.csv";
	write_matrix_file(emissionMatrix,hiddenStates,hiddenStates*(T-1),xname);	

}


void write_init(const double * const transitionMatrix,
		const double * const emissionMatrix,
		const int * const observations,
		const double * const stateProb,
		const int hiddenStates,
		const int differentObservables,
		const int T){


	char tname[100]="init/transitionMatrix.csv";
	write_matrix_file(transitionMatrix,hiddenStates,hiddenStates,tname);	

	char ename[100]="init/emissionMatrix.csv";
	write_matrix_file(emissionMatrix,hiddenStates,differentObservables,ename);	

	char oname[100]="init/observations.csv";
	write_vector_file_int(observations,T,oname);	

	char pname[100]="init/stateProb.csv";
	write_vector_file(stateProb,hiddenStates,pname);	

}

void write_result(const double * const transitionMatrix,
		const double * const emissionMatrix,
		const int * const observations,
		const double * const stateProb,
		const int const steps,
		const int hiddenStates,
		const int differentObservables,
		const int T){


	char tname[100]="result/transitionMatrix.csv";
	write_matrix_file(transitionMatrix,hiddenStates,hiddenStates,tname);	

	char ename[100]="result/emissionMatrix.csv";
	write_matrix_file(emissionMatrix,hiddenStates,differentObservables,ename);	

	char oname[100]="result/observations.csv";
	write_vector_file_int(observations,T,oname);	

	char pname[100]="result/stateProb.csv";
	write_vector_file(stateProb,hiddenStates,pname);

	char sname[100]="result/steps.csv";
	write_vector_file_int(&steps,1,sname);	

}


