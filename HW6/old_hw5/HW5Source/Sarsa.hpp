#pragma once

#define _USE_MATH_DEFINES 
#include <math.h>			// Defines M_PI
#include <iostream>
#include "Agent.hpp"
#include <string>
#include <cmath>
#include <memory>

namespace SarsaIamNotSureWhetherICanAddExtraHeaderFilesDuringSubmission
{
class LinearApprox{
public:

	LinearApprox(int stDim, int nAct): stateDim(stDim), numActions(nAct){}
	virtual ~LinearApprox(){}

	virtual void getFeatures(const Eigen::VectorXd& s, int action, Eigen::VectorXd &feats) = 0;

	//virtual void getQwsaFeatures(const Eigen::VectorXd& s, Eigen::VectorXd &feats) = 0;

	virtual int getFeatDim() = 0;

	//virtual void getAllFeatures(const Eigen::VectorXd& s, int action, Eigen::VectorXd &feats) = 0;

	void dqwsa(const Eigen::VectorXd& s, int action, Eigen::VectorXd &grad){
		getFeatures(s, action, grad);
	}

	double qwsa(const Eigen::VectorXd& s, int action){
		//std::cout << "Calculating qwsa" << std::endl;
		Eigen::VectorXd feats;
		getFeatures(s, action, feats);
		const int featDim = getFeatDim();
		// std::cout << "Before dot" << std::endl;
		// std::cout << "Action is " << action << std::endl;
		// std::cout << "segment size  " << W.segment(action * featDim, featDim).size() << std::endl;
		// std::cout << "feat size " << feats.size() << std::endl;
		// std::cout << "FeatDim is " << featDim << std::endl;
		auto ret =  W.segment(action * featDim, featDim).dot(feats);
		//std::cout << "Done Calculating qwsa" << std::endl;
		return ret;
		//std::cout << "After dot" << std::endl;
	}

	// double allQwsa(const Eigen::VectorXd& s){
	// 	Eigen::VectorXd feats;
	// }

	Eigen::VectorXd W;
protected:
	const int stateDim;
	const int numActions;
};

class PolyFeatLinearApprox: public LinearApprox{

public:

	PolyFeatLinearApprox(int stDim, int numActions, int pdim, int intiVal):LinearApprox(stDim, numActions), polydim(pdim), featDim(pdim * stDim){
		W = Eigen::VectorXd::Constant(pdim * numActions * stDim, intiVal);

	}

	void getFeatures(const Eigen::VectorXd& s, int action, Eigen::VectorXd &feats) override{
		feats = Eigen::VectorXd::Zero(featDim);

		for(int i = 0; i < s.size(); i++){
			double curVal = s(i);
			for(int j = 0; j < polydim; j++){
				feats(i * polydim + j) = curVal;
				curVal *= s(i);
			}
		}
	}
	/*void getFeatures(const Eigen::VectorXd& s, int action, Eigen::VectorXd &feats) override{
		feats = Eigen::VectorXd::Zero(featDim * stateDim);

		for(int i = 0; i < s.size(); i++){
			double curVal = 1.0;
			for(int j = 0; j < polydim; j++){
				curVal *= s(i);
				feats(i * polydim + j) = curVal;
			}
		}
	}*/

	int getFeatDim() override{
		return featDim;
	}

private:
	const int polydim;
	const int featDim;
};


class FourierBasisApprox: public LinearApprox{

public:

	FourierBasisApprox(int stDim, int numActions, int b, int t, int initVal):LinearApprox(stDim, numActions), basis(b), featDim(t){
		W = Eigen::VectorXd::Constant(t * numActions, initVal);

	}

	void incrementFourierCounter(Eigen::VectorXd & buff, const int& maxDigit) {
		for (int i = 0; i < (int)buff.size(); i++) {
			buff[i]++;
			if (buff[i] <= maxDigit)
				break;
			buff[i] = 0;
		}
	}

	void getFeatures(const Eigen::VectorXd& s, int action, Eigen::VectorXd &feats) override{

		feats = Eigen::VectorXd::Zero(featDim);
		Eigen::VectorXd buff = Eigen::VectorXd::Zero(stateDim);
		//std::cout << "Calculating Fourier basis" << std::endl;
		for(int i = 0; i < featDim; i++){
			//std::cout << "buff is " << buff.transpose()  << std::endl;
			double cosval = cos(M_PI * buff.dot(s));
			feats(i) = cosval;
			incrementFourierCounter(buff, basis - 1);
		}

		//std::cout << "Done calculating fourier basis" << std::endl;

	}

	int getFeatDim() override{
		return featDim;
	}

private:
	const int basis;
	const int featDim;
};

}

class Sarsa : public Agent
{
public:
	Sarsa(const int & stateDim, const int & numActions, const std::string & envName);	
	static bool updateBeforeNextAction();
	int getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) override;
	void newEpisode() override;
	void update(const Eigen::VectorXd& s, const int& a, const double& r, const Eigen::VectorXd& sPrime, const int & aPrime, std::mt19937_64& generator) override;
	void update(const Eigen::VectorXd& s, const int& a, const double& r, std::mt19937_64& generator) override;

private:
	// @TODO: Fill in any member variables and additional functions

	// Two helper functions that I used. You may not need these.
	
	int ipow(const int& a, const int& b);
	void incrementCounter(Eigen::VectorXd& buff, const int& maxDigit);
	int getSoftMaxAction(const Eigen::VectorXd& s, std::mt19937_64& generator);
	int getEpsilonGreedyAction(const Eigen::VectorXd& s, std::mt19937_64& generator);
	std::shared_ptr<SarsaIamNotSureWhetherICanAddExtraHeaderFilesDuringSubmission::LinearApprox> approximator;
	std::string apporxType = "poly";
	double alpha = 0.01;
	int polyDim = 5;
	double gamma = 0.95;
	const std::string curEnvName;
	double explorationEpsilon;
	double explorationMult;
	double minExploreProb;
	double learningDecay = 1.0;
	double minLearningRate = 1.0;
	const int nStates;
	const int nActions;
};
