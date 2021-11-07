#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "NNLayer.cuh"

using namespace std;

class ReLUActivation : public NNLayer
{
public:
	ReLUActivation(string name);
	~ReLUActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);

private:
	Matrix A;

	Matrix Z;
	Matrix dZ;
};