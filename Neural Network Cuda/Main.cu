#include <iostream>
#include <time.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "NeuralNetwork.cuh"
#include "LinearLayer.cuh"
#include "ReluActivation.cuh"
#include "SigmoidActivation.cuh"
#include "NNException.cuh"
#include "BCECost.cuh"

#include "CoordinatesDataset.cuh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

int main()
{
	srand(time(NULL));

	CoordinatesDataset dataset(100, 21);
	BCECost bce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));

	// Network Training
	Matrix Y;
	for (int epoch = 0; epoch < 1001; epoch++)
	{
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++)
		{
			Y = nn.forward(dataset.getBatches().at(batch));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
		}

		if (epoch % 1 == 0)
		{
			cout << "Epoch: " << epoch << ", Cost: " << cost / dataset.getNumOfBatches() << endl;

			/*Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
			Y.copyDeviceToHost();
			cout << "--------Predictions--------" << endl;
			for (int index = 0; index < Y.shape.x; index++)
			{
				int printVal = Y[index] > 0.5 ? 1 : 0;
				cout << printVal << "\t";
			}
			cout << endl;

			cout << "--------Targets--------" << endl;
			for (int index = 0; index < dataset.getTargets().at(dataset.getNumOfBatches() - 1).shape.x; index++)
			{
				cout << dataset.getTargets().at(dataset.getNumOfBatches() - 1)[index] << "\t";
			}
			cout << endl;*/
		}
	}

	// Compute Accuracy
	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	Y.copyDeviceToHost();

	float accuracy = computeAccuracy(Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	cout << "Accuracy: " << accuracy << endl;

	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets)
{
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++)
	{
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i])
		{
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}
