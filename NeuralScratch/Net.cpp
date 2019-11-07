#include "Net.h"
#include <cassert>

Net::Net(const std::vector<unsigned>& topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
	{
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

		//Layer made, now fill it with neurons and add a bias neuron to the layer
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
		{
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			std::cout << "Made a neuron!" << std::endl;
		}

		//Force the bias node's ouput val to be 1.0. It is the last one created above
		m_layers.back().back().setOutputVal(1.0);
	}
}

void Net::feedForward(const std::vector<double>& inputVals) 
{
	assert(inputVals.size() == m_layers[0].size() - 1); //Subtract 1 for bias neuron

	//Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); ++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
	}

	//Forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
	{
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) //Minus 1 for bias
		{
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void Net::backProp(const std::vector<double> &targetVals)
{
	//Calculate overall net error (RMS of output neuron errors)
	Layer& ouputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < ouputLayer.size() - 1; ++n)
	{
		double delta = targetVals[n] - ouputLayer[n].getOutputVal();
		m_error += delta * delta;
	}
	m_error /= ouputLayer.size() - 1; //Get average
	m_error = sqrt(m_error); //RMS

	//Implement a recent average measurement
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);

	//Calculate ouput layer gradients
	for (unsigned n = 0; n < ouputLayer.size() - 1; ++n)
	{
		ouputLayer[n].calcOutputGradients(targetVals[n]);
	}

	//Calculate gradients on hidden layers
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		Layer &hiddenLayer = m_layers[layerNum];
		Layer &nextLayer = m_layers[layerNum + 1];

		for (unsigned n = 0; n < hiddenLayer.size() - 1; ++n)
		{
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	//Update connection weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		Layer &layer = m_layers[layerNum];
		Layer &prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n)
		{
			layer[n].updateInputWeights(prevLayer);
		}
	}

}

void Net::getResults(std::vector<double>& resultVals) const 
{
	resultVals.clear();

	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}
