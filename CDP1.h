#ifndef M_CPD
#define M_CPD

#include <Eigen>
#include <cmath>
#include <iostream>

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
			//const float degree = (Y.row(k) - X.row(j)).array().pow(2.0f).sum() / (sigma * sigma);
			const float degree = (Y.row(k) - X.row(j)).array().pow(2.0f).sum() / sigma;
			sum += expf(-0.5f*degree);
		}
		//float w = 0.05f;
		//sum += (w/(1-w))*powf(2.0f*3.14f*sigma,1.5f)*float(m)/float(n);
		for (uint32_t i = 0; i < m; ++i)
		{
			//const float degree = (Y.row(i) - X.row(j)).array().pow(2.0f).sum() / (sigma * sigma);
			const float degree = (Y.row(i) - X.row(j)).array().pow(2.0f).sum() / sigma;
			P(i, j) = expf(-0.5f*degree) / sum;
			//std::printf(" %f ",P(i,j));
		}
	}
}

inline void get_W(const MatrixX3f &X, const MatrixX3f &Y0, const MatrixXf &P, const MatrixXf &G, MatrixX3f &W, const float lambda, const float sigma, double &Q)
{
	const uint32_t m = Y0.rows();
	const uint32_t n = X.rows();

	const MatrixXf diag = (P*VectorXf::Ones(n)).asDiagonal();
	//const MatrixXf id = lambda*sigma*sigma*MatrixXf::Identity(m, m);
	const MatrixXf id = lambda*sigma*MatrixXf::Identity(m, m);
	const MatrixXf A = (diag * G + id);
	//const MatrixXf A_inv = A.inverse();
	const MatrixXf B = P * X - diag * Y0;

	//W = A.colPivHouseholderQr().solve(B);
	W = A.inverse()*B;
}

// X - template(reference), Y - points to map
template<uint32_t DIM>
void CPD(const MatrixX3f &X, const MatrixX3f &Y0, MatrixX3f &Y, const float lambda, const float beta, const float sigma_initial, const float alpha, double &Q)
{

	typedef Matrix<float, -1, DIM> MatrixXDf;

	// initialization
	const int iterations = 100;
	float sigma = sigma_initial;
	const uint32_t m = Y0.rows();
	const uint32_t n = X.rows();

	Y = Y0;
	MatrixXDf W(m, DIM);
	MatrixXf G(m, m);
	MatrixXf P(m, n);
	MatrixXf Pold(m, n);
	VectorXf Q_vector(iterations);

	construct_G(Y0, G, beta);
	float sum = 0.0f;
	for (uint32_t i = 0; i < n; i++) {
		for (uint32_t j = 0; j < m; j++) {
			const Vector3f degree = (X.row(i) - Y0.row(j));
			sum += degree.dot(degree);
		}
	}
	float sigma2 = (1.0f / (3.0f*float(m)*float(n)))*sum;
	std::printf(" sigma=%f", sigma2);
	for (uint32_t i = 0; i < iterations; ++i)
	{
		//expectation
		Pold = P;
		construct_P(X, Y, P, sigma2);
		//maximization
		//get_W(X, Y0, P, G, W, lambda, sigma);
		get_W(X, Y0, P, G, W, lambda, sigma2,Q);
		//update
		//Y = Y0 + G*W;
		Y = Y0 + G*W;

		//Q
		MatrixXf A = G*W;
		for (int i = 0; i<n; i++)
			for (int j = 0; j < m; j++)
			{
				Q = (0.5 / (sigma2)) * Pold(i, j) * (X.row(i) - Y.row(j)).dot(X.row(i) - Y.row(j));
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
		Q = Q + np*3.0* log(sigma2) + lambda*0.5*max;
		Q_vector(i) = Q;
		std::cout << Q << std::endl;

		//const VectorXf t1 = VectorXf::Ones(n);
		//float np = t1.transpose()*P*VectorXf::Ones(n);
		//sigma *= alpha;
		const MatrixXf F = X.transpose()*(P*VectorXf::Ones(n)).asDiagonal()*X;
		const MatrixXf F1 = (P*X).transpose()*Y;
		const MatrixXf F2 = Y.transpose()*(P*VectorXf::Ones(n)).asDiagonal()*Y;
		
		sigma2 = (1.0 / (np*3.0))*(F.trace() - 2.0f * F1.trace() + F2.trace());
		

	}
	writeQ("F:/path/Q.txt", Q_vector);
	return;
}


#endif