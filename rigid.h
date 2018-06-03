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
//inline void construct_P(const MatrixX3f &X, const MatrixX3f &Y, MatrixXf &P, const float sigma)
inline void construct_P(const MatrixX3f &X, const MatrixX3f &Y, MatrixXf &P, const MatrixXf &R, const MatrixXf &t, const float s,const float sigma2)
{
	const uint32_t m = Y.rows();
	const uint32_t n = X.rows();

	for (uint32_t j = 0; j < n; j++)
	{
		float sum = 0.0f;
	
		for (uint32_t k = 0; k < m; k++)
		{
		
			const Vector3f degree = (s*(R*Y.row(k).transpose()+t) - X.row(j).transpose()) ;
			sum += expf((-0.5f / sigma2)*degree.dot(degree));
			
		}
		
		
		for (uint32_t i = 0; i < m; i++)
		{
			const Vector3f degree = (s*(R*Y.row(i).transpose()+t) - X.row(j).transpose());
			P(i, j) = expf((-0.5f / sigma2)*degree.dot(degree)) / sum;
		
		}
		std::printf(" j=%i ", j);
	}
}

//inline void get_W(const MatrixX3f &X, const MatrixX3f &Y0, const MatrixXf &P, const MatrixXf &G, MatrixX3f &W, const float lambda, const float sigma)
inline void get_Rst(const MatrixX3f &X, const MatrixX3f &Y0, const MatrixXf &P, MatrixXf &R, MatrixXf &t, float &s, float sigma3)
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
	MatrixXf A = X_emp.transpose()*P.transpose()*Y_emp;
	std::printf(" %f ",Np);
	BDCSVD<Eigen::Matrix<float,-1,-1,0,-1,-1>> A1 = A.bdcSvd();//-fp-model precise опции -fp-model
	float det_UVt = (A1.matrixU()*A1.matrixV().transpose()).determinant();
	VectorXf C = VectorXf::Ones(A1.matrixU().cols());
	C[A1.matrixU().cols()-1] = det_UVt;
	R = A1.matrixU()*C.asDiagonal()*A1.matrixV().transpose();
	s = (A.transpose()*R).trace() / (Y_emp.transpose()*(P*VectorXf::Ones(n)).asDiagonal()*Y_emp).trace();
	t = mx - s*R*my;
	MatrixXf PtO = P.transpose()*VectorXf::Ones(m);
	MatrixXf AtR = A.transpose()*R;
	MatrixXf XtPX = X_emp.transpose()*PtO*X_emp;
	float f = XtPX.trace() - s*AtR.trace();
	sigma3 = f/ (Np*3.0f);
}

// X - template(reference), Y - points to map
template<uint32_t DIM>
void CPD(const MatrixX3f &X, const MatrixX3f &Y0, MatrixX3f &Y, const float lambda, const float beta, const float sigma_initial, const float alpha)
{
	//std::string filepath = "F:/ВУЗ/Курсовая.КГ/томография сетка 4/100iter_0.98alpha_1beta_3sigma_1lambda/";
	std::string filepath = "F:/ВУЗ/Курсовая.КГ/томография сетка 3/Новая папка/FGT3/";
	double xsr = 0, ysr = 0, zsr = 0;
	typedef Matrix<float, -1, DIM> MatrixXDf;

	// initialization
	const int iterations = 100;
	float sigma = sigma_initial;
	const uint32_t m = Y0.rows();
	const uint32_t n = X.rows();
	Y = Y0;
	MatrixXf R= MatrixXf::Identity(DIM, DIM);
	MatrixXf t=MatrixXf::Zero(DIM,1);
	float s=1.0;
	
	MatrixXf P(m, n);

	float sum = 0.0f;
	for (uint32_t i = 0; i < n; i++) {
		for (uint32_t j = 0; j < m; j++) {
			const Vector3f degree = (X.row(i) - Y0.row(j));
			sum += degree.dot(degree);
		}
	}
	float sigma2 = (1.0f / (3.0f*float(m)*float(n)))*sum;
	std::printf(" sigma=%f", sigma2);

	unsigned int start_time = clock();
	unsigned int end_time = clock();
	double t1, t2, dt;
	t1 = omp_get_wtime();
	for (uint32_t i = 0; i < iterations; ++i)
	{
		//std::printf(" iter = %i", i);
		//start_time = clock();

		//expectation
		//construct_P(X, Y, P, sigma);
		construct_P(X, Y, P,R,t,s, sigma2);

		//end_time = clock();
		//std::printf(" time P = %i", end_time - start_time);

		//maximization
		//get_W(X, Y0, P, G, W, lambda, sigma);

		//start_time = clock();

		get_Rst(X, Y0, P, R, t, s, sigma2);

		//end_time = clock();
		//std::printf(" time W = %i", end_time - start_time);

		//update
		Y = s*Y0*R.transpose() + VectorXf::Ones(m)*t.transpose();

		char buffer[20];
		_itoa_s(i, buffer, 10);         // 10 - это система счисления
		std::string s = buffer;
		writePoints(filepath + "t" + s + ".txt", Y, 0, xsr, ysr, zsr);
	}
	t2 = omp_get_wtime();
	std::printf("  TIME = %f", t2 - t1);
}


#endif
