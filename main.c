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

    double *dop;
    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = -1;
    double step1 = 10.0 / (raz - 1);

    double* arr_pred = (double*)calloc(raz * raz, sizeof(double));
    double* arr_new = (double*)calloc(raz * raz, sizeof(double));

    double x1 = 10.0;
    double x2 = 20.0;
    double y1 = 20.0;
    double y2 = 30.0;
    arr_pred[0] = 10;
    arr_pred[raz] = 20;
    arr_pred[raz * (raz - 1) + 1] = 20;
    arr_pred[raz * raz] = 30;
    
    double error = 1 + max_toch;
    // Move data to device (accelerator)
#pragma acc enter data create(arr_pred[0:raz*raz], arr_new[0:raz*raz]) copyin(raz, step1)
#pragma acc kernels
    {
        // Initialize boundary conditions
#pragma acc loop independent
        for (int i = 0; i < raz; i++) {
            arr_new[i * raz] = arr_pred[i * raz] = x1 + i * step1;
            arr_new[i] = arr_pred[i] = x1 + i * step1;
            arr_new[(raz - 1) * raz + i] = arr_pred[(raz - 1) * raz + i] = y1 + i * step1;
            arr_new[i * raz + (raz - 1)] = arr_pred[i * raz + (raz - 1)] = x2 + i * step1;
        }
    }

    int itter = 0;
    // Perform iterations until convergence
    while (itter < 1000000 && error > 1e-6) {
        itter++;
        // Every 100 iterations or the first iteration, calculate error
        if (itter % 100 == 0 || itter == 1) {
            // Perform Jacobi iteration on device
#pragma acc data present(arr_pred[0:raz*raz], arr_new[0:raz*raz])
#pragma acc kernels async(1)
            {
#pragma acc loop independent collapse(2)
                for (int i = 1; i < raz - 1; i++) {
                    for (int j = 1; j < raz - 1; j++) {
                        arr_pred[i * raz + j] =
                                0.25 * (arr_new[(i + 1) * raz + j] + arr_new[(i - 1) * raz + j] + arr_new[i * raz + j - 1] + arr_new[i * raz + j + 1]);
                    }
                }
            }
            int id = 0;
#pragma acc wait
            // Calculate error and update arr_new
#pragma acc host_data use_device(arr_pred, arr_new)
            {
                cublasDaxpy(handle, raz * raz, &alpha, arr_pred, 1, arr_new, 1);
                cublasIdamax(handle, raz * raz, arr_new, 1, &id);
            }
#pragma acc update self(arr_new[id-1:1])

#pragma acc update self(arr_new[id-1:1])
            error = fabs(arr_new[id - 1]);
#pragma acc host_data use_device(arr_pred, arr_new)
            cublasDcopy(handle, raz * raz, arr_pred, 1, arr_new, 1);

        } else {
#pragma acc data present(arr_pred[0:raz*raz], arr_new[0:raz*raz])
#pragma acc kernels async(1)
            {
#pragma acc loop independent collapse(2)
                for (int i = 1; i < raz - 1; i++) {
                    for (int j = 1; j < raz - 1; j++) {
                        arr_pred[i * raz + j] =
                                0.25 * (arr_new[(i + 1) * raz + j] + arr_new[(i - 1) * raz + j] + arr_new[i * raz + j - 1] + arr_new[i * raz + j + 1]);
                    }
                }
            }
        }
        dop = arr_new;
        arr_new = arr_pred;
        arr_pred = dop;

        if (itter % 100 == 0 || itter == 1)
#pragma acc wait(1)
            printf("%d %e\n", itter, error);

    }

    printf("%d\n", itter);
    printf("%e", error);
    cublasDestroy(handle);
    return 0;
}
