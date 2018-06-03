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


inline void construct_G(const MatrixX3f &Y0, const float beta, MatrixXf &Q, MatrixXf &L, MatrixXf &G)
{
	const uint32_t m = Y0.rows();
	const uint32_t k = L.rows();
	for (uint32_t i = 0; i < m; ++i)
	{
		for (uint32_t j = 0; j < m; ++j)
		{
			const Vector3f degree = (Y0.row(i) - Y0.row(j));
			G(i, j) = expf(-0.5f*(1.0/beta)*degree.dot(degree));

		}
	}
	EigenSolver<MatrixXf> ei(G);
	Eigen::Matrix<std::complex<float>,-1,1> eisolvM = ei.eigenvalues();
	VectorXf eisolv(m);
	
	for (uint32_t i = 0; i < eisolv.size(); i++)
	{
		eisolv(i) = eisolvM(i, 0).real();
	}

	float *ksolvsort = new float[eisolv.size()];
	float *ksolvsort_num = new float[eisolv.size()];
	for (uint32_t i = 0; i < eisolv.size(); i++)
	{
		ksolvsort[i] = eisolv(i);
		ksolvsort_num[i] = i;
	}

	VectorXf kmaxeisolv(k);

	for (uint32_t i = 0; i < eisolv.size() - 1; i++)
		for(uint32_t j = 0; j < eisolv.size() - 1; j++)
			if (ksolvsort[j + 1] < ksolvsort[j])
			{
				float tmp_num = ksolvsort_num[j + 1];
				ksolvsort_num[j + 1] = ksolvsort_num[j];
				ksolvsort_num[j] = tmp_num;
				float tmp = ksolvsort[j + 1];
				ksolvsort[j + 1] = ksolvsort[j];
				ksolvsort[j] = tmp;
			}
	for (uint32_t i = 0; i < k; i++)
	{
		kmaxeisolv(i) = ksolvsort[eisolv.size()-1-i];
	}
	
	L = kmaxeisolv.asDiagonal();

	Eigen::Matrix<std::complex<float>, -1, -1> eivec_complex = ei.eigenvectors();
	
	MatrixXf eivec(eivec_complex.rows(), eivec_complex.cols());
	
	for (uint32_t i = 0; i < eivec_complex.rows(); i++)
		for (uint32_t j = 0; j < eivec_complex.cols(); j++)
	{
		eivec(i,j) = eivec_complex(i,j).real();
	}
	for (uint32_t i = 0; i < m; i++)
		for (uint32_t j = 0; j <k ; j++)
		{
			Q(i, j) = eivec(i, ksolvsort_num[eisolv.size() - 1-j]);
		}
	delete ksolvsort;
	delete ksolvsort_num;
}

//P posterior probabilities
inline void construct_P(const MatrixX3f &X, const MatrixX3f &Y, MatrixXf &P1, MatrixXf &PX, const float sigma)
{
	const uint32_t m = Y.rows();
	const uint32_t n = X.rows();

	MatrixXf K(m,n);
	MatrixXf a(n, 1);
	MatrixXf Kt1(n,1);

	for (uint32_t j = 0; j < n; ++j)
	{
		for (uint32_t i = 0; i < m; ++i)
		{

			const Vector3f degree = (X.row(i) - Y.row(j));
			K(i, j) = expf(-0.5f*(1.0 / sigma)*degree.dot(degree));

		}
	}
	Kt1 = K.transpose()*VectorXf::Ones(n);
	for (uint32_t i = 0, j=0; i < n; ++i)
	{
		a(i, j) = 1.0 / Kt1(i, j);
	}
	VectorXf a_v(n, 1);
	for (int i = 0; i < n; i++) a_v(i) = a(i, 0);
	P1 = K*a;
	PX = K*(a_v.asDiagonal()*X);
}


inline void get_W(const MatrixX3f &X, const MatrixX3f &Y0, const MatrixXf &P1, const MatrixXf &PX, const MatrixXf &Q, const MatrixXf &L, MatrixX3f &W, const float lambda, const float sigma)
{
	const uint32_t m = Y0.rows();
	const uint32_t n = X.rows();
	const uint32_t k = Q.cols();


	const MatrixXf id = (1.0/(lambda*sigma*sigma))*MatrixXf::Identity(m, m);
	const MatrixXf id1 = (-1.0 / (lambda*sigma*sigma* lambda*sigma*sigma))*MatrixXf::Identity(m, m);
	const MatrixXf id2 = (1.0 / (lambda*sigma*sigma))*MatrixXf::Identity(k, k);


	MatrixXf diagp1 = P1.asDiagonal();
	MatrixXf transq = Q.transpose();
	MatrixXf F = L.inverse() + id2*transq*diagp1*Q;
	MatrixXf F1 = (id*diagp1 - id1*diagp1*Q*F.inverse()*transq*diagp1);
	MatrixXf F2 = (diagp1.inverse()*PX-Y0);
	W = F1*F2;
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
	int k = int(std::pow(m, 1.0/2.0));
	std::printf(" k=%i", k);
	MatrixXf Q(m, k);
	MatrixXf L(k, k);
	MatrixXf P1(m, 1);
	MatrixXf PX(m, 3);

	//MatrixXf P(m, n);
	MatrixXf G(m, m);
	construct_G(Y0, beta, Q, L, G);

	unsigned int start_time = clock();
	unsigned int end_time = clock();
	double t1, t2, dt;
	t1 = omp_get_wtime();
	for (uint32_t i = 0; i < iterations; ++i)
	{

		construct_P(X, Y, P1, PX, sigma);

		get_W(X, Y0, P1, PX, Q, L, W, lambda, sigma);

		Y = Y0 + G*W;

		sigma *= alpha;


	}
	t2 = omp_get_wtime();
	std::printf("  TIME = %f", t2 - t1);
}
