#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <time.h>

int N=10000000;

int main() {
    double* arr = (double*)malloc((N+1)*sizeof(double));
    double Sum=0;
    clock_t a=clock();
#pragma acc kernels
    for (int i = 0; i < N; i++)
        arr[i] = sin(2 * M_PI * i / N);
    clock_t b=clock();
#pragma acc kernels
    for (int i = 0; i < N; i++)
        Sum+=arr[i];
    printf("%.25f\n", Sum);

    double d=(double)(b-a)/CLOCKS_PER_SEC;
    printf("%.25f", d);
    free(arr);
    return 0;
}
