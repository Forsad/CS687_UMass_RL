#include "QLearning.hpp"

#include <limits>
using namespace std;
using namespace Eigen;
using namespace QLearningIamNotSureWhetherICanAddExtraHeaderFilesDuringSubmission;
/*
* In this assignment, the constructor takes the environment name and
* uses it to set hyperparameters like the step size, gamma, and any other parameters.
*/
QLearning::QLearning(const int& stateDim, const int& numActions, const std::string& envName):curEnvName(envName), nStates(stateDim), nActions(numActions)
{
	// @TODO: Fill in this function

	// In the constructor, use envName to set hyperparameters.
	// No not use envName to initialize the policy!
	// I'm leaving in some of my code to show how you might structure this.
	if (envName.compare("Mountain Car") == 0)
	{
		// @TODO: Set hyperparameters for Gridworld.
		alpha = 0.1;
		gamma = 0.99;
		explorationEpsilon = 0.9;
		explorationMult = 0.99;
		minExploreProb = 0.001;
		learningDecay = 0.98;
		minLearningRate = 0.0001;
		//std::cout << "State size is " << stateDim << std::endl;
		//approximator = std::make_shared<FourierBasisApprox>(stateDim, numActions, 2, 24, 10);
		approximator = std::make_shared<PolyFeatLinearApprox>(stateDim, numActions, 5, 0);
		//std::cout << "State size " << stateDim << std::endl;
		//approximator = std::make_shared<FourierBasisApprox>(stateDim, numActions, 5, 25, 10);
	}
	else if (envName.compare("Cart Pole") == 0)
	{
		// @TODO: Set hyperparameters for Cart Pole.
		alpha = 0.08;
		gamma = 0.999;
		explorationEpsilon = 0.2;
		explorationMult = 0.98;
		minExploreProb = 0.01;
		learningDecay = 0.98;
		minLearningRate = 0.0001;
		//std::cout << "State size " << stateDim << std::endl;
		//approximator = std::make_shared<PolyFeatLinearApprox>(stateDim, numActions, 5, 1000);
		approximator = std::make_shared<FourierBasisApprox>(stateDim, numActions, 2, 16, 100);
	}
	else if (envName.compare("Gridworld") == 0)
	{
		// @TODO: Set hyperparameters for Gridworld.
		alpha = 0.1;
		polyDim = 1;
		gamma = 0.99;
		explorationEpsilon = 0.95;
		explorationMult = 0.94;
		minExploreProb = 0.01;
		learningDecay = 0.99;
		minLearningRate = 0.0001;
		//std::cout << "State size is " << stateDim << std::endl;
		//approximator = std::make_shared<FourierBasisApprox>(stateDim, numActions, 2, 24, 10);
		approximator = std::make_shared<PolyFeatLinearApprox>(stateDim, numActions, 1, 4);
	}
	else
	{
		cout << "Error: Unknown environment name in Sarsa constructor." << endl;
		exit(1);
	}

	// @TODO: Fill in the remainder of this function.
}


int QLearning::getSoftMaxAction(const Eigen::VectorXd& s, std::mt19937_64& generator)
{
	// @TODO: Fill in this function.
	VectorXd actionProbabilities(nActions);

	for(int i = 0; i < nActions; i++){

		actionProbabilities(i) = approximator->qwsa(s, i);
	}

	actionProbabilities = actionProbabilities.array().exp();
	actionProbabilities.array() /= actionProbabilities.sum();
	//
	double temp = uniform_real_distribution<double>(0, 1)(generator), sum = 0;
	for (int a = 0; a < nActions; a++)
	{
		sum += actionProbabilities[a];
		if (temp <= sum)
			return a;	// The function will return 'a'. This stops the for loop and returns from the function.
	}
	return nActions - 1; // Rounding error
	//return 0; // Delete this line - this is just here to that the code compiles.
}



int QLearning::getEpsilonGreedyAction(const Eigen::VectorXd& s, std::mt19937_64& generator)
{
	// @TODO: Fill in this function.
	VectorXd actionProbabilities(nActions);

	int maxIdx = 0;
	for(int i = 0; i < nActions; i++){

		actionProbabilities(i) = approximator->qwsa(s, i);
		if(actionProbabilities(i) > actionProbabilities(maxIdx)){
			maxIdx = i;
		}
	}
	double temp = uniform_real_distribution<double>(0, 1)(generator);
	if (temp < explorationEpsilon){
		int randIdx = std::uniform_int_distribution<>(0, nActions - 1)(generator);
		return randIdx;
	}
	return maxIdx;
}

bool QLearning::updateBeforeNextAction()
{
	return true;
}

// Epsilon-greedy or Softmax action selection
int QLearning::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator)
{
	// @TODO: Fill in this function.
	return getEpsilonGreedyAction(s, generator);
	//return 0; // Delete this line - this is just here to that the code compiles.
}

// Tell the agent that it is at the start of a new episode
void QLearning::newEpisode()
{
	// @TODO: Fill in this function
	if(explorationEpsilon * explorationMult > minExploreProb){
			explorationEpsilon *= explorationMult;
	}
	if(alpha * learningDecay > minLearningRate){
		alpha *= learningDecay;
	}
	if (curEnvName.compare("Mountain Car") == 0){
	}

	else if (curEnvName.compare("Cart Pole") == 0){
	}

	else if (curEnvName.compare("Gridworld") == 0){
		//alpha *= 0.999;
	}

}

// Update given a (s,a,r,s') tuple
void QLearning::update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, std::mt19937_64& generator)
{
	// @TODO: Fill in this function.
	Eigen::VectorXd grad;
	approximator->dqwsa(s, a, grad);
	const int featDim = approximator->getFeatDim();
	//std::cout << "Before update" << std::endl;
	double maxQwsa = std::numeric_limits<double>::lowest();
	for(int aPrime = 0; aPrime < nActions; aPrime++){
		auto val = approximator->qwsa(sPrime, aPrime);
		if(val > maxQwsa){
			maxQwsa = val;
		}
	}
	(approximator->W).segment(a * featDim, featDim) += alpha * (r + gamma * maxQwsa - approximator->qwsa(s, a)) * grad;
}

// Let the agent update/learn when sPrime would be the terminal absorbing state
void QLearning::update(const Eigen::VectorXd& s, const int& a, const double& r, mt19937_64& generator)
{
	// @TODO: Fill in this function.
	Eigen::VectorXd grad;
	approximator->dqwsa(s, a, grad);

	const int featDim = approximator->getFeatDim();

	(approximator->W).segment(a * featDim, featDim) += alpha * (r - approximator->qwsa(s, a)) * grad;
}

/*
* My code used this function. It returns a^b, where a and b are both integers.
* You may not need this, but if you do, we are including it.
*/
int QLearning::ipow(const int& a, const int& b) {
	if (b == 0) return 1;
	if (b == 1) return a;
	int tmp = ipow(a, b / 2);
	if (b % 2 == 0) return tmp * tmp;
	else return a * tmp * tmp;
}

/*
* This is another function that my code uses, and which you are welcome to use.
* It is entirely possible that your code won't require this function.
* This function takes as input a vector, buff, that represents a number in
* base (maxDigit+1), and adds one to the counter. Overflow behavior is not defined.
*/
void QLearning::incrementCounter(VectorXd& buff, const int& maxDigit) {
	for (int i = 0; i < (int)buff.size(); i++) {
		buff[i]++;
		if (buff[i] <= maxDigit)
			break;
		buff[i] = 0;
	}
}