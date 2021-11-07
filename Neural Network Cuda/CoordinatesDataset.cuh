#pragma once

#include <vector>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "Matrix.cuh"

using namespace std;


class CoordinatesDataset
{
public:
	CoordinatesDataset(size_t batch_size, size_t number_of_batches);

	int getNumOfBatches();

	vector<Matrix>& getBatches();
	vector<Matrix>& getTargets();

private:
	size_t batch_size;
	size_t number_of_batches;

	vector<Matrix> batches;
	vector<Matrix> targets;
};