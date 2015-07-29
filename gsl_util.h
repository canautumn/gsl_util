//
// gsl_util.h
// Customized GSL utilities
// Created by Can on 7/27/15.
//

#ifndef GSL_UTIL_H
#define GSL_UTIL_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector_double.h>


//
// gsl_util_pinv
// Pseudo-inversion using Jacobi SVD
//
void gsl_util_pinv(const gsl_matrix *A_c, gsl_matrix *pinv) {

    gsl_matrix *A = gsl_matrix_calloc(A_c->size1, A_c->size2);
    gsl_matrix_memcpy(A, A_c);

    gsl_matrix *V = gsl_matrix_calloc(A->size1, A->size2);
    gsl_vector *S = gsl_vector_calloc(A->size1);

    gsl_linalg_SV_decomp_jacobi(A, V, S);

    for (size_t j = 0; j < A->size1; ++j) {
        double factor = gsl_vector_get(S, j);
        if (factor > 1e-10 || factor < -1e-10) factor = 1 / factor; else factor = 0;
        for (size_t i = 0; i < A->size2; ++i)
            gsl_matrix_set(V, i, j, gsl_matrix_get(V, i, j) * factor);
    }

    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, V, A, 0.0, pinv);
    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_vector_free(S);
}


//
// mean_of
// Calculate mean value after applying a lambda function. Useful for mean and variance.
// The data must be represented by a vector. Data1 and 2 must have the same size.
// If the sampled == 1, the mean is a sampled mean (the denominator is the number of samples minus 1).
//
double mean_of(const gsl_vector * data1, const gsl_vector * data2, std::function<double(double, double)> pre_operation, int sampled = 0) {

    if (data1->size != data2->size) {
        std::cerr << "Vector size error" << endl;
        return 0;
    }
    double sum = 0;
    for (size_t ob = 0; ob < data1->size; ++ob) {
        sum += pre_operation(gsl_vector_get(data1, ob), gsl_vector_get(data2, ob));
    }
    sum /= data1->size - sampled;
    return sum;
}


//
// gsl_util_covariance_matrix
// Calculate covariance matrix. The covariance matrix must be a square matrix with the
// length equal to the second dimension of the matrix to be calculated.
//
void gsl_util_covariance_matrix(const gsl_matrix * data, gsl_matrix * covariance) {

    size_t dims = data->size2;
    size_t obs = data->size1;
    if (covariance->size1 != covariance->size2 || covariance->size2 != dims) {
        std::cerr << "Dimension error" << endl;
        return;
    }

    double * means = new double[dims];

    for (size_t dim = 0; dim < dims; ++dim) {
        gsl_vector_const_view data1 = gsl_matrix_const_column(data, dim);
        means[dim] = mean_of(&data1.vector, &data1.vector, [] (double el1, double el2) {
            return el1;
        });
    }

    for (size_t dim1 = 0; dim1 < dims; ++dim1) {
        gsl_vector_const_view data1 = gsl_matrix_const_column(data, dim1);
        double means_dim1 = means[dim1];
        for (size_t dim2 = dim1; dim2 < dims; ++dim2) {
            gsl_vector_const_view data2 = gsl_matrix_const_column(data, dim2);
            double means_dim2 = means[dim2];
            double cov = mean_of(&data1.vector, &data2.vector, [means_dim1, means_dim2] (double el1, double el2) {
               return (el1 - means_dim1) * (el2 - means_dim2);
            }, 1);
            gsl_matrix_set(covariance, dim1, dim2, cov);
            gsl_matrix_set(covariance, dim2, dim1, cov);
        }
    }

    delete means;
}


#endif //GSL_UTIL_H
