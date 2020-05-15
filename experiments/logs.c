#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
//https://stackoverflow.com/questions/9411823/fast-log2float-x-implementation-c/28730362#28730362
int main(int argc, char *argv[]){
    double variable = atof(argv[1]);
    double maths = log2(variable);

    union { float val; int32_t x; } u = { variable };
    register float log_2 = (float)(((u.x >> 23) & 255) - 128);              
    u.x   &= ~(255 << 23);
    u.x   += 127 << 23;
    log_2 += ((-0.34484843f) * u.val + 2.02466578f) * u.val - 0.67487759f; 

    printf("math %f \n self %f \n", maths, log_2);
    return 0;
}
