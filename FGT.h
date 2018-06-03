#ifndef M_CPD
#define M_CPD

#include <Eigen>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <omp.h>
#include <cmath>
#include"io.h"

typedef Vector3f vec3;


inline void construct_G(const MatrixX3f &Y0, const float beta, MatrixXf &G)
{
	const uint32_t m = Y0.rows();
	for (uint32_t i = 0; i < m; ++i)
	{
		for (uint32_t j = 0; j < m; ++j)
		{
			const Vector3f degree = (Y0.row(i) - Y0.row(j));
			G(i, j) = expf(-0.5f*(1.0 / beta)*degree.dot(degree));
		}
	}
}

//P posterior probabilities
inline void construct_P(const MatrixX3f &X, const MatrixX3f &Y, MatrixXf &P1, MatrixXf &PX, const float sigma)
{
	const uint32_t m = Y.rows();
	const uint32_t n = X.rows();

	MatrixXf K(m, n);
	MatrixXf a(n, 1);
	MatrixXf Kt1(n, 1);

	for (uint32_t j = 0; j < n; ++j)
	{
		for (uint32_t i = 0; i < m; ++i)
		{

			const Vector3f degree = (X.row(i) - Y.row(j));
			K(i, j) = expf(-0.5f*(1.0 / sigma)*degree.dot(degree));
		}
	}
	Kt1 = K.transpose()*VectorXf::Ones(n);
	for (uint32_t i = 0, j = 0; i < n; ++i)
	{
		a(i, j) = 1.0 / Kt1(i, j);
	}
	VectorXf a_v(n, 1);
	for (int i = 0; i < n; i++) a_v(i) = a(i, 0);
	P1 = K*a;
	PX = K*(a_v.asDiagonal()*X);
}

inline void get_W(const MatrixX3f &X, const MatrixX3f &Y0, const MatrixXf &P1, const MatrixXf &PX, const MatrixXf &G, MatrixX3f &W, const float lambda, const float sigma)
{
	const uint32_t m = Y0.rows();
	const uint32_t n = X.rows();

	const MatrixXf diag = P1.asDiagonal();
	const MatrixXf id = (1.0 / (lambda*sigma*sigma))*MatrixXf::Identity(m, m);
	const MatrixXf A = (diag * G + id);
	const MatrixXf A_inv = A.inverse();
	const MatrixXf B = PX - diag * Y0;
	W = A.colPivHouseholderQr().solve(B);
}

// X - template(reference), Y - points to map
template<uint32_t DIM>
void CPD(const MatrixX3f &X, const MatrixX3f &Y0, MatrixX3f &Y, const float lambda, const float beta, const float sigma_initial, const float alpha)
{
	std::string filepath = "F:/ВУЗ/Курсовая.КГ/томография сетка 5/FGT/";
	double xsr = 0, ysr = 0, zsr = 0;
	typedef Matrix<float, -1, DIM> MatrixXDf;

	// initialization
	const int iterations = 100;
	float sigma = sigma_initial;
	const uint32_t m = Y0.rows();
	const uint32_t n = X.rows();
	Y = Y0;
	MatrixXDf W(m, DIM);
	int k = int(std::pow(m, 1.0 / 2.0));
	MatrixXf P1(m, 1);
	MatrixXf PX(m, 3);

	//MatrixXf P(m, n);
	MatrixXf G(m, m);
	construct_G(Y0, beta, G);

	unsigned int start_time = clock();
	unsigned int end_time = clock();
	double t1, t2, dt;
	t1 = omp_get_wtime();
	for (uint32_t i = 0; i < iterations; ++i)
	{
		construct_P(X, Y, P1, PX, sigma);

		get_W(X, Y0, P1, PX, G, W, lambda, sigma);

		Y = Y0 + G*W;

		sigma *= alpha;

	}
	t2 = omp_get_wtime();
	std::printf("  TIME = %f", t2 - t1);
}


#endif

