#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
int N=10000000;

int main() {
    clock_t a=clock();
    float* arr = (float*)malloc((N+1)*sizeof(float));
    float Sum=0;
#pragma acc kernels
    for (int i = 0; i < N; i++) {
        arr[i] = (float)sin(2 * M_PI * i / N);
        Sum+=arr[i];
    }
    printf("%.25f\n", Sum);
    clock_t b=clock();
    double d=(double)(b-a);
    printf("%.25f", d);
    return 0;
}
