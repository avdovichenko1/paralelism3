#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc, char *argv[]) {
    
    int max_num_iter = atoi(argv[1]); // количество итераций
    double max_toch = atof(argv[2]); // точность
    int raz = atoi(argv[3]); // размер сетки
    clock_t a=clock();
    
    double *buf;
    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = -1;
    double step1 = 10.0 / (raz - 1);

    double* u = (double*)calloc(raz * raz, sizeof(double));
    double* up = (double*)calloc(raz * raz, sizeof(double));
    double x1 = 10.0;
    double x2 = 20.0;
    double y1 = 20.0;
    double y2 = 30.0;
    u[0] = up[0] = x1;
    u[raz] = up[raz] = x2;
    u[raz * (raz - 1) + 1] = up[raz * (raz - 1) + 1] = y1;
    u[raz * raz] = up[raz * raz] = y2;

    // Move data to device (accelerator)
#pragma acc enter data create(u[0:raz*raz], up[0:raz*raz]) copyin(raz, step1)
#pragma acc kernels
    {
        // Initialize boundary conditions
#pragma acc loop independent
        for (int i = 0; i < raz; i++) {
            u[i * raz] = up[i * raz] = x1 + i * step1;
            u[i] = up[i] = x1 + i * step1;
            u[(raz - 1) * raz + i] = up[(raz - 1) * raz + i] = y1 + i * step1;
            u[i * raz + (raz - 1)] = up[i * raz + (raz - 1)] = x2 + i * step1;
        }
    }

    int itter = 0;
    double error = 1.0;
    // Perform iterations until convergence
    while (itter < 1000000 && error > 1e-6) {
        itter++;
        // Every 100 iterations or the first iteration, calculate error
        if (itter % 100 == 0 || itter == 1) {
            // Perform Jacobi iteration on device
#pragma acc data present(u[0:raz*raz], up[0:raz*raz])
#pragma acc kernels async(1)
            {
#pragma acc loop independent collapse(2)
                for (int i = 1; i < raz - 1; i++) {
                    for (int j = 1; j < raz - 1; j++) {
                        up[i * raz + j] =
                                0.25 * (u[(i + 1) * raz + j] + u[(i - 1) * raz + j] + u[i * raz + j - 1] + u[i * raz + j + 1]);
                    }
                }
            }
            int id = 0;
#pragma acc wait
            // Calculate error and update u
#pragma acc host_data use_device(u, up)
            {
                cublasDaxpy(handle, raz * raz, &alpha, up, 1, u, 1);
                cublasIdamax(handle, raz * raz, u, 1, &id);
            }
#pragma acc update self(u[id-1:1])

#pragma acc update self(u[id-1:1])
            error = fabs(u[id - 1]);
#pragma acc host_data use_device(u, up)
            cublasDcopy(handle, raz * raz, up, 1, u, 1);

        } else {
#pragma acc data present(u[0:raz*raz], up[0:raz*raz])
#pragma acc kernels async(1)
            {
#pragma acc loop independent collapse(2)
                for (int i = 1; i < raz - 1; i++) {
                    for (int j = 1; j < raz - 1; j++) {
                        up[i * raz + j] =
                                0.25 * (u[(i + 1) * raz + j] + u[(i - 1) * raz + j] + u[i * raz + j - 1] + u[i * raz + j + 1]);
                    }
                }
            }
        }
        buf = u;
        u = up;
        up = buf;

        if (itter % 100 == 0 || itter == 1)
#pragma acc wait(1)
            printf("%d %e\n", itter, error);

    }

    printf("%d\n", itter);
    printf("%e", error);
    cublasDestroy(handle);
    return 0;
}
