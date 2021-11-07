#pragma once

#include <exception>
#include <iostream>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

using namespace std;

class NNException : exception
{
public:
	NNException(const char* exception_message)
		:exception_message(exception_message)
	{}

	virtual const char* what() const throw()
	{
		return exception_message;
	}

	static void throwIfDeviceErrorsOccurred(const char* exception_message)
	{
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess)
		{
			cerr << error << ": " << exception_message;
			throw NNException(exception_message);
		}
	}

private:
	const char* exception_message;
};