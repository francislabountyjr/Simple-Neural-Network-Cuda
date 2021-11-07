#pragma once

#include <vector>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "NNLayer.cuh"
#include "BCECost.cuh"

class NeuralNetwork
{
public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();;

	Matrix forward(Matrix X);
	void backprop(Matrix predictions, Matrix target);

	void addLayer(NNLayer* layer);
	vector<NNLayer*> getLayers() const;

private:
	vector<NNLayer*> layers;
	BCECost bce_cost;

	Matrix Y;
	Matrix dY;
	float learning_rate;
};