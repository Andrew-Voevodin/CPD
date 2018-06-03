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


//P posterior probabilities

inline void construct_P(const MatrixX3f &X, const MatrixX3f &Y, MatrixXf &P, const MatrixXf &B, const MatrixXf &t, const float sigma2)
{
	const uint32_t m = Y.rows();
	const uint32_t n = X.rows();

	for (uint32_t j = 0; j < n; j++)
	{
		float sum = 0.0f;

		for (uint32_t k = 0; k < m; k++)
		{

			const Vector3f degree = (Y.row(k) - X.row(j));
			sum += expf((-0.5f / sigma2)*degree.dot(degree));

		}


		for (uint32_t i = 0; i < m; i++)
		{
			const Vector3f degree = (Y.row(i) - X.row(j));
			P(i, j) = expf((-0.5f / sigma2)*degree.dot(degree)) / sum;

		}
		//std::printf(" sum=%f ", sum);
	}
}

inline void get_Rst(const MatrixX3f &X, const MatrixX3f &Y0, const MatrixXf &P, MatrixXf &B, MatrixXf &t,float sigma3)
{
	const uint32_t m = Y0.rows();
	const uint32_t n = X.rows();

	VectorXf one_n = VectorXf::Ones(n);
	VectorXf one_m = VectorXf::Ones(m);
	float Np = one_m.transpose()*P*one_n;
	MatrixXf mx = (1.0 / Np)*X.transpose()*P.transpose()*one_m;
	MatrixXf my = (1.0 / Np)*Y0.transpose()*P*one_n;
	MatrixXf X_emp = X - VectorXf::Ones(3)*mx.transpose();
	MatrixXf Y_emp = Y0 - VectorXf::Ones(3)*my.transpose();
	B = (X_emp.transpose()*P.transpose()*Y_emp)*(Y_emp.transpose()*(P*VectorXf::Ones(n)).asDiagonal()*Y_emp).inverse();
	t = mx - B*my;
	MatrixXf PtO = P.transpose()*VectorXf::Ones(m);
	MatrixXf AtR = X_emp.transpose()*P.transpose()*Y_emp*B.transpose();
	MatrixXf XtPX = X_emp.transpose()*PtO.asDiagonal()*X_emp;
	float f = XtPX.trace() - AtR.trace();
	sigma3 = f / (Np*3.0f);
}

// X - template(reference), Y - points to map
template<uint32_t DIM>
void CPD(const MatrixX3f &X, const MatrixX3f &Y0, MatrixX3f &Y, const float lambda, const float beta, const float sigma_initial, const float alpha)
{
	std::string filepath = "F:/path/FGT/";
	double xsr = 0, ysr = 0, zsr = 0;
	typedef Matrix<float, -1, DIM> MatrixXDf;

	// initialization
	const int iterations = 100;
	float sigma = sigma_initial;
	const uint32_t m = Y0.rows();
	const uint32_t n = X.rows();
	Y = Y0;
	MatrixXf B = MatrixXf::Identity(DIM, DIM);
	MatrixXf t = MatrixXf::Zero(DIM, 1);
	float s = 1.0;

	MatrixXf P(m, n);

	float sum = 0.0f;
	for (uint32_t i = 0; i < n; i++) {
		for (uint32_t j = 0; j < m; j++) {
			const Vector3f degree = (X.row(i) - Y0.row(j));
			sum += degree.dot(degree);
		}
	}
	float sigma2 = (1.0f / (DIM*float(m)*float(n)))*sum;
	std::printf(" sigma=%f", sigma2);

	unsigned int start_time = clock();
	unsigned int end_time = clock();
	double t1, t2, dt;
	t1 = omp_get_wtime();
	for (uint32_t i = 0; i < iterations; ++i)
	{

		construct_P(X, Y, P, B, t, sigma2);

		get_Rst(X, Y0, P, B, t, sigma2);

		//update
		Y = Y0*B.transpose() + VectorXf::Ones(m)*t.transpose();

	}
	t2 = omp_get_wtime();
	std::printf("  TIME = %f", t2 - t1);
}


#endif