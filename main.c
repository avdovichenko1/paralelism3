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
    

    const double alpha = -1;
    double step1 = 10.0 / (raz - 1);

    double* arr_pred = (double*)calloc(raz * raz, sizeof(double));
    double* arr_new = (double*)calloc(raz * raz, sizeof(double));

    arr_pred[0] = 10;
    arr_pred[raz] = 20;
    arr_pred[raz * (raz - 1) + 1] = 20;
    arr_pred[raz * raz] = 30;

    int num_iter = 0;
    double error = 1 + max_toch;

#pragma acc enter data create(arr_pred[0:raz*raz], arr_new[0:raz*raz]) copyin(raz, step1)
#pragma acc kernels
    {
#pragma acc loop independent
        for (int j = 0; j < raz; j++) {
            arr_pred[j] = 10 + j * step1;
            arr_pred[j * raz] = 10 + j * step1;
            arr_pred[(raz - 1) * raz + j] = 20 + j * step1;
            arr_pred[j * raz + (raz - 1)] = 20 + j * step1;
        }
    }
    cublasHandle_t handle;
    cublasCreate(&handle);

    while (max_num_iter > num_iter && max_toch < error) {
        num_iter++;
        // Every 100 iterations or the first iteration, calculate error
        if (num_iter % 100 == 0 || num_iter == 1) {
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

        if (num_iter % 100 == 0 || num_iter == 1)
#pragma acc wait(1)
            printf("%d %e\n", num_iter, error);

    }

    printf("%d\n", num_iter);
    printf("%e", error);
    cublasDestroy(handle);
    return 0;
}
