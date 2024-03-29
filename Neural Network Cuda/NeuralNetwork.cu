#include "NeuralNetwork.cuh"
#include "NNException.cuh"

NeuralNetwork::NeuralNetwork(float learning_rate)
	:learning_rate(learning_rate)
{}

NeuralNetwork::~NeuralNetwork()
{
	for (auto layer : layers)
	{
		delete layer;
	}
}

Matrix NeuralNetwork::forward(Matrix X)
{
	Matrix Z = X;

	for (auto layer : layers)
	{
		Z = layer->forward(Z);
	}

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target)
{
	dY.allocateMemoryifNotAllocated(predictions.shape);
	Matrix error = bce_cost.dCost(predictions, target, dY);

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++)
	{
		error = (*it)->backprop(error, learning_rate);
	}

	cudaDeviceSynchronize();
}

void NeuralNetwork::addLayer(NNLayer* layer)
{
	this->layers.push_back(layer);
}

vector<NNLayer*> NeuralNetwork::getLayers() const
{
	return layers;
}
