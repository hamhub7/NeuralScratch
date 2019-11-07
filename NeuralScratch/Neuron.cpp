#include "Neuron.h"

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
	for (unsigned c = 0; c < numOutputs; ++c)
	{
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}

	m_myIndex = myIndex;
}

void Neuron::feedForward(const Layer& prevLayer)
{
	double sum = 0.0;

	//Sum the previous layer's ouputs (which are our inputs) and add the bias neuron
	
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}

	m_outpuVal = Neuron::transferFunction(sum);
}

double Neuron::transferFunction(double x) 
{
	//tanh - ouput range [-1.0, 1.0]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
	//tanh derivative
	return 1.0 - x * x;
}

void Neuron::calcOutputGradients(double targetVal)
{
	double delta = targetVal - m_outpuVal;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outpuVal);
}

void Neuron::calcHiddenGradients(Layer &nextLayer)
{
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outpuVal);
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
	//The weights to be updated are in the Connection container in the neurons in the preceding layer

	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight; //Eta - overall net learning rate; Alpha - momentum

		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
	double sum = 0.0;

	//Sum our contributions of the errors at the nodes we feed
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}