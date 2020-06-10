#include <stdio.h> 
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define T 235

void main(){
	int b=T;
	int* a=(int*) malloc(T*sizeof(int));
	srand(time(NULL));
	for(int i=0;i<b;i++){
		a[i]=rand()%27+1;
	}
	FILE *fp;
	fp=fopen("sequence.seq","w");
	fprintf(fp,"T= %d \n", b);
	for(int i=0;i<b;i++){
		fprintf(fp,"%d ", a[i]);
	}	
	fclose(fp);
}