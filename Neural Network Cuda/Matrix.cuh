#pragma once

#include <memory>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "Shape.cuh"

using namespace std;

class Matrix
{
public:
	Shape shape;

	shared_ptr<float> data_device;
	shared_ptr<float> data_host;

	Matrix(size_t x_dim = 1, size_t y_dim = 1);
	Matrix(Shape shape);

	void allocateMemory();
	void allocateMemoryifNotAllocated(Shape shape);

	void copyHostToDevice();
	void copyDeviceToHost();

	float& operator[](const int index);
	const float& operator[](const int index) const;

private:
	bool device_allocated;
	bool host_allocated;

	void allocateCudaMemory();
	void allocateHostMemory();
};