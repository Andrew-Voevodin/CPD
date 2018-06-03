#ifndef M_CPD
#define M_CPD

#include <Eigen>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <omp.h>
#include"io.h"

typedef Vector3f vec3;


inline void construct_G(const MatrixX3f &Y0, MatrixXf &G, const float beta)
{
	const uint32_t m = Y0.rows();
	for (uint32_t i = 0; i < m; ++i)
	{
		for (uint32_t j = 0; j < m; ++j)
		{
			const Vector3f degree = (Y0.row(i) - Y0.row(j)) / beta;
			G(i, j) = expf(-0.5f*degree.dot(degree));
		}
	}
}

//P posterior probabilities
inline void construct_P(const MatrixX3f &X, const MatrixX3f &Y, MatrixXf &P, const float sigma)
{
	const uint32_t m = Y.rows();
	const uint32_t n = X.rows();

	for (uint32_t j = 0; j < n; ++j)
	{
		float sum = 0.0f;

		for (uint32_t k = 0; k < m; ++k)
		{
			const Vector3f degree = (Y.row(k) - X.row(j))/ sigma;
			sum += expf(-0.5f*degree.dot(degree));
		}

		for (uint32_t i = 0; i < m; ++i)
		{
			const Vector3f degree = (Y.row(i) - X.row(j)) / sigma;
			P(i, j) = expf(-0.5f*degree.dot(degree)) / sum;
		}

	}
}


inline void get_W(const MatrixX3f &X, const MatrixX3f &Y0, const MatrixXf &P, const MatrixXf &G, MatrixX3f &W, const float lambda, const float sigma, double &Q)
{
	const uint32_t m = Y0.rows();
	const uint32_t n = X.rows();

	const MatrixXf diag = (P*VectorXf::Ones(n)).asDiagonal();
	const MatrixXf id = lambda*sigma*sigma*MatrixXf::Identity(m, m);
	const MatrixXf A = (diag * G + id);
	
	const MatrixXf B = P * X - diag * Y0;


	W = A.colPivHouseholderQr().solve(B);

	
}

// X - template(reference), Y - points to map
template<uint32_t DIM>
void CPD(const MatrixX3f &X, const MatrixX3f &Y0, MatrixX3f &Y, const float lambda, const float beta, const float sigma_initial, const float alpha, double& Q)
{
	double xsr = 0, ysr = 0, zsr = 0;
	typedef Matrix<float, -1, DIM> MatrixXDf;

	// initialization
	const int iterations = 100;
	float sigma = sigma_initial;
	const uint32_t m = Y0.rows();
	const uint32_t n = X.rows();
	Y = Y0;
	MatrixXDf W(m, DIM);
	MatrixXf G(m, m);
	int m1 = m / 2;
	int n1 = n / 2;
	int m2 = m - m1;
	int n2 = n - n1;


	MatrixXf P(m, n);
	MatrixXf Pold(m, n);
	VectorXf Q_vector(iterations);

	construct_G(Y0, G, beta);

	unsigned int start_time = clock();
	unsigned int end_time = clock();
	double t1, t2, dt;
	t1 =omp_get_wtime();
	for (uint32_t i = 0; i < iterations; ++i)
	{

		Pold = P;
		construct_P(X, Y, P, sigma);

		get_W(X, Y0, P, G, W, lambda, sigma,Q);

		//update
		Y = Y0 + G*W;

		//Q
		MatrixXf A = G*W;
		for (int i = 0; i<n; i++)
			for (int j = 0; j < m; j++)
			{
				Q = (0.5 / (sigma*sigma)) * Pold(i, j) * (X.row(i) - Y.row(j)).dot(X.row(i) - Y.row(j));
			}
		const VectorXf t1 = VectorXf::Ones(n);
		float np = t1.transpose()*P*VectorXf::Ones(n);
		float max = -100000.0;
		for (int i = 0; i < m; i++)
		{
			float sum = 0.0;
			for (int j = 0; j < 3; j++)
			{
				sum = abs(A(i, j));
			}
			if (max < sum) max = sum;
		}
		Q = Q + np*3.0* log(sigma*sigma) + lambda*0.5*max;
		std::cout << Q<<std::endl;
		Q_vector(i) = Q;

		sigma *= alpha;
		

	}
	t2=omp_get_wtime();
	std::printf("  TIME = %f", t2-t1);
	writeQ("F:/path/Q.txt",Q_vector);

}


#endif

