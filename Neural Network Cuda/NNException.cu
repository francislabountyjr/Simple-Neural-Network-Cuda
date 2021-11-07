#include "NNException.cuh"

NNException::NNException(const char* exception_message)
	:exception_message(exception_message)
{}

const char* NNException::what() const throw()
{
	return exception_message;
}

void NNException::throwIfDeviceErrorsOccurred(const char* exception_message)
{
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		cerr << error << ": " << exception_message;
		throw NNException(exception_message);
	}
}
