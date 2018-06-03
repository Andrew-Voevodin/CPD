#include <Eigen>

//#include "CPD_FGT_low_rank.h"
#include "CPD.h"
//#include "CDP1.h"
//#include "affine.h"
//#include"rigid.h"
//#include <cpd\nonrigid.hpp>
//#include "FGT.h"
#include "io.h"



int main()
{
	std::string filepath = "F:/C++/coherentPointDrift/";

	float xsr = 0, ysr = 0, zsr = 0;
	float _xsr = 0, _ysr = 0, _zsr = 0;		

	MatrixX3f matX = readPoints(filepath + "distorted_fish.txt", 0, 0, 0, xsr, ysr, zsr);
	MatrixX3f matY = readPoints(filepath + "normal_fish.txt", 0, 0, 0, _xsr, _ysr, _zsr);
	MatrixX3f matZ(matY.cols(), matY.rows());             // Результат
		
	float alpha = 0.97f;//коррект. коэф. для сигма
	float beta = 10.0f;  //"ширина сглаживания"
	float sigma = 3.0f; //сигма в GMM
	float lambda = 10.0f;//представляет собой компромисс между the goodness of maximum likelihood fit and regularization.
	double Q = 0.0;

	CPD<3>(matX, matY, matZ, lambda, beta, sigma, alpha, Q);

	writePoints(filepath + "result.txt", matZ,0, xsr, ysr, zsr);//преобразовать в реальные размеры
	system("pause");
	return 0;
}