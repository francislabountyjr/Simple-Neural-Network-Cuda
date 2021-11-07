#pragma once

#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "Matrix.cuh"

using namespace std;

class NNLayer {
public:
	virtual ~NNLayer() = 0;

	virtual Matrix& forward(Matrix& A) = 0;
	virtual Matrix& backprop(Matrix& dZ, float learning_rate) = 0;

	string getName() { return this->name; };

protected:
	string name;
};

inline NNLayer::~NNLayer() {}