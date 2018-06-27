#include <cuda_runtime_api.h>
#include "io.h"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"curand.lib")
#include <cstdlib>
#include <string>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <assert.h>


#define BLOCK_SIZE 32

__global__ void copy(float *A, float *B, int n, int m)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		for (int j = 0; j < m; j++)
		{
			B[i*m+j] = A[i*m+j];
		}
	}
}

__global__ void construct_G(float *Y0, float *G, float beta, int m)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
	{
		for (int j = 0; j < m; j++)
		{
			float dev = 0.0;
			for (int k = 0; k < 3; k++)
			{
				dev += (Y0[k * m + i] - Y0[k * m + j])*(Y0[k * m + i] - Y0[k * m + j]);
			}
			float sum = (-1.0 / (2.0*beta*beta))*dev;
			G[j * m + i] = expf(sum);
		}
	}
}

__global__ void construct_P(float *X, float *Y, float *P, float sigma, int m, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		float sum = 0.0;

		for (int j = 0; j < m; j++)
		{
			float dev = 0.0;
			for (int k = 0; k < 3; k++)
			{
				dev = (-1.0 / (2.0*sigma*sigma))*(X[k * n + i] - Y[k * m + j])*(X[k * n + i] - Y[k * m + j]);
			}
			sum += expf(dev);
		}

		for (int j = 0; j < m; j++)
		{
			float dev = 0.0;
			for (int k = 0; k < 3; k++)
			{
				dev = (-1.0 / (2.0*sigma*sigma))*(X[k * n + i] - Y[k * m + j])*(X[k * n + i] - Y[k * m + j]);
			}
			P[m*i + j] = expf(dev) / sum;
		}
	}
}

__global__ void get_W(float *X, float *Y0, float *P, float *P1, float *PX, float *A, float *B, float *G, float *W, float lambda, float sigma, int m, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m)
	{
		float sum = 0.0;
		for (int j = 0; j < n; j++)
		{
			P1[i] += P[j*m + i];
		}

		for (int k = 0; k < m; k++)
		{
			A[k * m + i] = G[k * m + i];
		}
		A[i * m + i] = G[i * m + i] + lambda * sigma * sigma * (1.0 / P1[i]);
		for (int k = 0; k < 3; k++)
		{
			B[k*m + i] = (1.0 / P1[i])*PX[k*m + i] - Y0[k * m + i];
		}
	}
}

void printMatrix(int m, int n, const float*A, int lda, const char* name)
{
	for (int row = 0; row < m; row++){
		for (int col = 0; col < n; col++){
			double Areg = A[row + col*lda];
			printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, Areg);
		}
	}
}

int main()
{
	cusolverDnHandle_t cusolverH = NULL;
	cublasHandle_t cublasH = NULL;
	cublasHandle_t handle = NULL;
	std::string filepath = "F:/C++/coherentPointDrift/";
	std::ifstream fin;
	int sizex = 0;
	int sizey = 0;
	int sizez = 0;
	const float *step=new float(1.0);
	const float *step1 = new float(0.0);

	fin.open(filepath + "fish1.txt");
	fin >> sizex;
	fin.close();
	fin.open(filepath + "fish2.txt");
	fin >> sizey;
	fin.close();
	sizez = sizey;
	float* matX = new float[sizex * 3];
	float* matY = new float[sizey * 3];
	float* matZ = new float[sizez * 3];
	float* G = new float[sizey * sizey];
	float* W = new float[sizey * 3];
	float* P = new float[sizey * sizex];

	readPoints(filepath + "fish1.txt", 0, matX, sizex); // Матрица
	readPoints(filepath + "fish2.txt", 0, matY, sizey); // Матрица /шаблон
	readPoints(filepath + "fish2.txt", 0, matZ, sizez);

	float alpha = 0.97f;//коррект. коэф. для сигма
	float beta = 1.0f;  //"ширина сглаживания"
	float sigma = 1.0f; //сигма в GMM
	float lambda = 1.0f;//представляет собой компромисс между the goodness of maximum likelihood fit and regularization.
	int iterations = 100;
	int lwork = 0;

	float* dX = nullptr;
	float* dY = nullptr;
	float* dZ = nullptr;
	float* dW = nullptr;
	float* dG = nullptr;
	float* dP = nullptr;
	float* dP1 = nullptr;
	float* dPX = nullptr;
	float* dA = nullptr;
	float* dB = nullptr;
	float* d_work = nullptr;
	int* devInfo = nullptr;

	cudaStream_t stream = NULL;
	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;
	cublasStatus_t cublasst = CUBLAS_STATUS_SUCCESS;

	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);
	cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == cudaStat1);
	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaMalloc((void**)&devInfo, sizeof(float));
	cudaMalloc((void**)&dX, sizex * 3 * sizeof(float));
	cudaMalloc((void**)&dY, sizey * 3 * sizeof(float));
	cudaMalloc((void**)&dZ, sizez * 3 * sizeof(float));
	cudaMalloc((void**)&dW, sizey * 3 * sizeof(float));
	cudaMalloc((void**)&dG, sizey * sizey * sizeof(float));
	cudaMalloc((void**)&dP, sizex * sizey * sizeof(float));
	cudaMalloc((void**)&dA, sizey * sizey * sizeof(float));
	cudaMalloc((void**)&dB, sizey * 3 * sizeof(float));
	cudaMalloc((void**)&dP1, sizey * sizeof(float));
	cudaMalloc((void**)&dPX, sizey * 3 * sizeof(float));

	cudaMemcpy(dX, matX, sizex * 3 * sizeof(float),	cudaMemcpyHostToDevice);
	cudaMemcpy(dY, matY, sizey * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dZ, matZ, sizey * 3 * sizeof(float), cudaMemcpyHostToDevice);

	construct_G << <(sizey + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> >(dY, dG, beta, sizey);

	cudaStat1 = cudaDeviceSynchronize();
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaDeviceSynchronize();
	assert(cudaSuccess == cudaStat1);

	for (int i = 0; i < iterations; ++i)
	{
		construct_P << <(sizex + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> >(dX, dZ, dP, sigma, sizey, sizex);

		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);

		cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, sizey, 3, sizex, step, dP, sizex, dX, sizex, step1, dPX, sizey);

		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);

		get_W << <(sizey + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> >(dX, dY, dP, dP1, dPX, dA, dB, dG, dW, lambda, sigma, sizey, sizex);
		
		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);

		/* dA*X=dB */
		
		/*workspace*/
		status = cusolverDnSgetrf_bufferSize(cusolverH,	sizey, sizey, dA, sizey, &lwork);
		assert(CUSOLVER_STATUS_SUCCESS == status);
		
		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);
		
		cudaStat1 = cudaMalloc((void**)&d_work, sizeof(float)*lwork);
		assert(cudaSuccess == cudaStat1);

		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);

		/*dA=LU*/
		status = cusolverDnSgetrf(cusolverH, sizey,	sizey, dA, sizey, d_work, NULL, devInfo);
		assert(CUSOLVER_STATUS_SUCCESS == status);

		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);

		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);

		/*X=dB*/
		cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, sizey, 3, dA, sizey, NULL, dB, sizey, devInfo);


		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);

		cudaStat1 = cudaMemcpy(dW, dB, sizeof(float)*sizey*3, cudaMemcpyDeviceToDevice);
		assert(cudaSuccess == cudaStat1);

		//update
		//Y = Y0 + G*W;

		
		cudaStat1 = cudaMemcpy(dZ, dY, sizeof(float)*sizey * 3, cudaMemcpyDeviceToDevice);
		assert(cudaSuccess == cudaStat1);

		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);

		cublasst = cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, sizey, 3, sizey, step, dG, sizey, dW, sizey, step, dZ, sizey);
		//if (cublasst == CUBLAS_STATUS_SUCCESS) std::cout << 0;
		assert(CUBLAS_STATUS_SUCCESS == cublasst);

		std::cout<<cudaGetLastError();

		//cudaMemcpy(matZ, dZ, sizey * 3 * sizeof(float), cudaMemcpyDeviceToHost);
		//printf("z = (matlab base-1)\n");
		//printMatrix(sizey, 3, matZ, sizey, "Z");
		//printf("=====\n");

		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);
				
		cudaFree(d_work);
		
		cudaStat1 = cudaDeviceSynchronize();
		assert(cudaSuccess == cudaStat1);

		sigma *= alpha;

	}
	
	cudaMemcpy(matY, dY, sizey* 3* sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(matZ, dZ, sizey * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(matX, dX, sizex * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(G, dG, sizey * sizey * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(P, dP, sizey * sizex * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(W, dW, sizey * 3 * sizeof(float), cudaMemcpyDeviceToHost);
	
	writePoints(filepath + "fish_gpu.txt", matZ, sizez);

	cudaFree(dPX);
	cudaFree(dP1);

	cudaFree(dX);
	cudaFree(dY);
	cudaFree(dZ);
	cudaFree(dW);
	cudaFree(dG);
	cudaFree(dP);
	cudaFree(dA);
	cudaFree(dB);
	delete matX;
	delete matY;
	delete matZ;
	delete step;
	delete step1;
	system("pause");
	return 0;
}