#pragma once

#include <vector>
#include <cstdlib>
#include <cmath>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection
{
	double weight;
	double deltaWeight;
};

class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputVal(double val) { m_outpuVal = val; }
	double getOutputVal() const { return m_outpuVal; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetVal);
	void calcHiddenGradients(Layer &nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double eta; // [0.0, 1.0] overall net training weight
	static double alpha; // [0.0, n] muliplier of last weight change (momentum)

	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double sumDOW(const Layer& nextLayer) const;
	double m_outpuVal;
	unsigned m_myIndex;
	std::vector<Connection> m_outputWeights;
	static double randomWeight(void) { return rand() / double(RAND_MAX); }
	double m_gradient;
};