#include <iostream>
#include <assert.h>

#include "REINFORCE.hpp"	// The .cpp file for an class should include the header file (the .hpp file).

using namespace std;				// So we don't have to keep writing std::
using namespace Eigen;				// So we don't have to keep writing Eigen::

#define INIT_ALPHA 0.01
#define MIN_ALPHA 0.01
#define ALPHA_INCR 0.01
#define MAX_PERIOD 10
#define MIN_PERIOD 10
#define ALPHA_DECAY 1.003

// See the .hpp file for a description of this "constructor"
REINFORCE::REINFORCE(const int& stateDim, const int& numActions, const double& gamma) : EpisodicAgent(1)
{
	this->numStates = stateDim;									// Remember the number of states in a private member variable
	this->numActions = numActions;								// Remember the number of actions in a private member variable
	this->gamma = gamma;										// Remember gamma too
	
	theta = MatrixXd::Zero(numStates, numActions);				// Initialize the policy parameters to the zero Matrix

	alpha = INIT_ALPHA;
}

// Ask the agent to select an action given the current state
int REINFORCE::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const
{
	int state = oneHotToInt(s);
	VectorXd actionProbabilities = theta.row(state).array().exp();
	actionProbabilities.array() /= actionProbabilities.sum();

	// Sample an action from actionProbabilities.	
	double temp = uniform_real_distribution<double>(0, 1)(generator), sum = 0;
	for (int a = 0; a < numActions; a++)
	{
		sum += actionProbabilities[a];
		if (temp <= sum)
			return a;	// The function will return 'a'. This stops the for loop and returns from the function.
	}
	return numActions - 1; // Rounding error
}

// Reset the agent entirely - to a blank slate prior to learning
void REINFORCE::reset(std::mt19937_64& generator)
{
	theta.setZero();
	alpha = INIT_ALPHA;
	EpisodicAgent::reset(generator);	// See EpisodicAgent::reset. Here TabularBBO is the subclass and EpisodicAgent is the superclass. EpisodicAgent has its own variables to reset, but calling "reset" calls the function for the subclass. So, this line is saying "also call the reset function for EpisodicAgent too!"
}

void REINFORCE::episodicUpdate(mt19937_64& generator)
{
	// There is only one episode stored
	// Start by computing the unbiased estimate of the policy gradient
	MatrixXd gradientEstimate = MatrixXd::Zero(numStates, numActions);

	// TODO: Write code to compute the REINFORCE (without baseline) unbiased estimator of the policy gradient, and store it in gradientEstimate
	// Note: Drop the gamma^t term that is usually dropped in actual implementations!
	
	const int L = actions[0].size();

	double gt = 0.0;
	MatrixXd actionProbabilities = theta;
	for(int i = 0; i < numStates; i++){
		actionProbabilities.row(i) = actionProbabilities.row(i).array().exp();
		actionProbabilities.row(i).array() /= actionProbabilities.row(i).sum();
	}
	//std::cout << actionProbabilities << endl;
	for(int i = L - 1; i >= 0; i--){
		gt = rewards[0][i] + gamma * gt;
		int state = oneHotToInt(states[0][i]);
		int action = actions[0][i];
		gradientEstimate.row(state) -= gt * actionProbabilities.row(state);
		gradientEstimate(state, action) += gt;
	}
	// Perform the actual update.
	theta = theta + alpha * gradientEstimate;

	if(alpha * ALPHA_DECAY >= MIN_ALPHA){
		alpha *= ALPHA_DECAY;
		//cout << alpha << endl;
	}
}

int REINFORCE::oneHotToInt(const VectorXd& v) const
{
	for (int i = 0; i < (int)v.size(); i++)
		if (v[i] != 0)
			return i;
	assert(false);
	cout << "Error in REINFORCE - expected one hot vector was all zeros!" << endl;
	exit(1);
}