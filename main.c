#include <stdio.h>
#include <malloc.h>
#include <math.h>

int N=10000000;

int main() {
    double Sum=0;
#pragma acc kernels
    for (int i = 0; i < N; i++) {
        Sum+=sin(2 * M_PI * i / N);
    }
    printf("%.25f", Sum);
    return 0;
}
