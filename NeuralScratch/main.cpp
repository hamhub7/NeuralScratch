#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include "Net.h"
#include "Neuron.h"
#include "TrainingData.h"

void showVectorVals(std::string label, std::vector<double>& v)
{
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i)
	{
		std::cout << v[i] << " ";
	}

	std::cout << std::endl;
}

void genTrainingData(std::string filename)
{
	std::ofstream outf;
	outf.open(filename, std::ofstream::out | std::ofstream::trunc);

	outf << "topology: 2 4 1" << std::endl;
	for (int i = 1000; i >= 0; --i)
	{
		int n1 = (int)(2.0 * rand() / double(RAND_MAX));
		int n2 = (int)(2.0 * rand() / double(RAND_MAX));
		int t = n1 ^ n2;
		outf << "in: " << n1 << ".0 " << n2 << ".0 " << std::endl;
		outf << "out: " << t << ".0" << std::endl;
	}

	outf.close();
}

int main()
{
	const std::string filename = "D:/NN/trainingData.txt";

	genTrainingData(filename);
	TrainingData trainData(filename);

	std::vector<unsigned> topology;
	trainData.getTopology(topology);
	Net myNet(topology);

	std::vector<double> inputVals, targetVals, resultVals;
	int trainingPass = 0;

	while (!trainData.isEof())
	{
		++trainingPass;
		std::cout << std::endl << "Pass " << trainingPass << std::endl;

		//Get new input data and feed it forward:
		if (trainData.getNextInputs(inputVals) != topology[0]) { break; }

		showVectorVals("Inputs:", inputVals);
		myNet.feedForward(inputVals);

		//Collect the net's actual results:
		myNet.getResults(resultVals);
		showVectorVals("Outputs:", resultVals);

		//Train the net what the outputs should have been
		trainData.getTargetOutputs(targetVals);
		showVectorVals("Targets:", targetVals);
		assert(targetVals.size() == topology.back());

		myNet.backProp(targetVals);

		//Report how well the training is working, averaged over recent runs
		std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;
		std::cout << std::endl << "Done!" << std::endl;

	}

	return 0;
}
