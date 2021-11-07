#pragma once

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "matrix.cuh"

using namespace std;

class BCECost
{
public:
	float cost(Matrix predictions, Matrix target);

	Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};