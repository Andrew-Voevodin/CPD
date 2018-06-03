#pragma once
#include <Eigen>
#include <fstream>

MatrixX3f readPoints(std::string filepath, int index, int masht, int masht1, float &_xsr, float &_ysr, float &_zsr)
{
	std::ifstream fin;
	fin.open(filepath);
	int size;
	float xsr = 0, ysr = 0, zsr = 0;
	fin >> size;
	MatrixX3f result = MatrixX3f(size, 3);
	if (index == 0) {
		for (int i = 0; i < size; i++)
		{
			fin >> result(i, 0) >> result(i, 1) >> result(i, 2);
			xsr += result(i, 0);
			ysr += result(i, 1);
			zsr += result(i, 2);
		}
	}
	else {
		for (int i = 0; i < size; i++)
		{
			float number = 0;
			fin >> number >> result(i, 0) >> result(i, 1) >> result(i, 2);
			const Vector3f degree = result.row(i);
			xsr += result(i, 0);
			ysr += result(i, 1);
			zsr += result(i, 2);
		}
	}
	fin.close();
	xsr /= size;
	ysr /= size;
	zsr /= size;
	_xsr = xsr;
	_ysr = ysr;
	_zsr = zsr;
	if(masht==1)
		for (int i = 0; i < size; i++)
			for (int j = 0; j < 3; j++) 
			{
				if (masht1 == 1) 
				{
					if (j == 0) result(i, j) = (result(i, j)-xsr)/10.0;
					if (j == 1) result(i, j) = (result(i, j)-ysr)/10.0;
					if (j == 2) result(i, j) = (result(i, j)-zsr)/10.0;
				}
				else 
				{
					if (j == 0) result(i, j) = (result(i, j) - xsr);
					if (j == 1) result(i, j) = (result(i, j) - ysr);
					if (j == 2) result(i, j) = (result(i, j) - zsr);
				}
				//result(i, j) /= 10;
			}
	return result;
}
void writePoints(std::string filepath, MatrixX3f &result, int masht, float xsr, float ysr, float zsr)
{
	if (masht == 0) {
		std::ofstream fout;
		fout.open(filepath);
		int size = result.rows();
		fout << size << "\n";
		for (int i = 0; i < size; i++)
		{
			fout << result(i, 0) << " "
				<< result(i, 1) << " "
				<< result(i, 2) << "\n";
		}
		fout.close();
	}
	if (masht == 1) {
		std::ofstream fout;
		fout.open(filepath);
		int size = result.rows();
		fout << size << "\n";
		for (int i = 0; i < size; i++)
		{
			//fout << (result(i, 0)*10.0 + xsr) << " "
			//	 << (result(i, 1)*10.0 + ysr) << " "
			//	 << (result(i, 2)*10.0 + zsr) << "\n";
			fout << (result(i, 0)*10.0) << " "
				 << (result(i, 1)*10.0) << " "
				 << (result(i, 2)*10.0) << "\n";
		}
		fout.close();
	}
}

void writeQ(std::string filepath, VectorXf &result)
{
	
	std::ofstream fout;
	fout.open(filepath);
	int size = result.rows();
	fout << size << "\n";
	for (int i = 0; i < size; i++)
	{
		fout << result.row(i) << "\n";
	}
	fout.close();
	
}