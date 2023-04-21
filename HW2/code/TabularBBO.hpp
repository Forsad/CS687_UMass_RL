#pragma once

// COMPILER THAT YOU USED: [type here the compiler you used]

#include "EpisodicAgent.hpp"

/*
* You will implement this class for HW1. See TabularRandomSearch for a similar example.
*/


#include <utility>
#include <vector>

#define NOAGENTS 10
#define TOTRUNPERAGENT 2

class TabularBBO : public EpisodicAgent
{
public:
	// Do not change the arguments provided to the constructor!
	TabularBBO(const int& stateDim, const int& numActions, const double& gamma, const int& N, const int & maxEps);

	// Ask the agent to select an action given the current state
	int getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const override;

	// Reset the agent entirely - to a blank slate prior to learning
	void reset(std::mt19937_64& generator) override;

	void episodicUpdate(std::mt19937_64& generator) override;
	void newEpisode() override;

private:
	// @TODO: Define aditional variables here. You can also define additional functions here (or in public: if you would like)
	const int noAgents = NOAGENTS;
	const int totRunPerAgent = TOTRUNPERAGENT;
	const int noSurvivingAgents = 4;
	const double mutationVariance = 0.5;
	const double mutationMultiplier = 1.0;
	double curMuatationVariance;
	const double initialVariance = 2;
	const int naturalSelPeriod = noAgents * totRunPerAgent;
    int numStates;				// How many discrete states?
	int numActions;				// How many discrete actions?
	int maxEps;					// How many episodes will be run?
	int totalEpisodes = 0;
	double gamma;				// Discount parameter
	//Eigen::MatrixXd curTheta;
	//double curThetaJHat;
	std::vector<Eigen::MatrixXd> curThetas;	// The current best policy we have found
	std::vector<std::pair<double ,int > > curThetaJHats;		// $\hat J(\theta_\text{cur})$ in LaTeX, this is the estimate of how good the current policy is
	//Eigen::MatrixXd newTheta;	// The policy we're currently running and thinking of switching curTheta to
	//double newThetaJHat;		// This will store our estimate of how good newTheta is.

	void generateMatWithVar(Eigen::MatrixXd &mat, double variance, std::mt19937_64& generator);
	void addMatWithVar(Eigen::MatrixXd &mat, double variance, std::mt19937_64& generator);
};
