#pragma once

#define _USE_MATH_DEFINES 
#include <math.h>			// Defines M_PI
#include <iostream>

#include "Agent.hpp"

#include <memory>
#include <string>
#include <cmath>
#include "linear_approx.hpp"

class QLearning : public Agent
{
public:
	QLearning(const int & stateDim, const int & numActions, const std::string & envName);
	static bool updateBeforeNextAction();
	int getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) override;
	void newEpisode() override;
	void update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, std::mt19937_64& generator) override;
	void update(const Eigen::VectorXd& s, const int& a, const double& r, std::mt19937_64& generator) override;

private:
	// @TODO: Fill in any member variables and additional functions

	// Two helper functions that I used. You may not need these.
	int ipow(const int& a, const int& b);
	void incrementCounter(Eigen::VectorXd& buff, const int& maxDigit);

	int getSoftMaxAction(const Eigen::VectorXd& s, std::mt19937_64& generator);
	int getEpsilonGreedyAction(const Eigen::VectorXd& s, std::mt19937_64& generator);
	std::shared_ptr<LinearApprox> approximator;
	std::string apporxType = "poly";
	double alpha = 0.001;
	int polyDim = 5;
	double gamma = 0.95;
	const std::string curEnvName;
	double explorationEpsilon;
	double explorationMult;
	double minExploreProb;
	double learningDecay = 1.0;
	double minLearningRate = 1.0;
	double lambda = 0.0;
	Eigen::VectorXd e;
	const int nStates;
	const int nActions;
};
