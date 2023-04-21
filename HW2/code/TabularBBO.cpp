#include "TabularBBO.hpp"
#include <iostream>
using namespace std;				// So we don't have to keep writing std::
using namespace Eigen;				// So we don't have to keep writing Eigen::

// See the .hpp file for a description of this "constructor"
// Note: If you want to use a different value for N, pass that value at the end of the line below, e.g., EpisodicAgent(2)
TabularBBO::TabularBBO(const int& stateDim, const int& numActions, const double& gamma, const int& N, const int & maxEps) : EpisodicAgent(NOAGENTS * TOTRUNPERAGENT)
{
    //cout << naturalSelPeriod << endl;
    //char ch = getchar();
    //exit(0);
	this->numStates = stateDim;									// Remember the number of states in a private member variable
	this->numActions = numActions;								// Remember the number of actions in a private member variable
	this->maxEps = maxEps;
	this->gamma = gamma;										// Remember gamma too
	//std::mt19937_64 generator("Totally random seed");
	//curTheta = newTheta = MatrixXd::Zero(numStates, numActions);// Initialize both the current and new theta to the zero Matrix
	//curThetaJHat = -INFINITY;									// Set the current policy's performance to -INFINITY so that we always take the first policy tested.
	epCount = 0;
}
void TabularBBO::newEpisode()
{
    //cout << "episode count is " << epCount << endl;
}

void TabularBBO::addMatWithVar(Eigen::MatrixXd &mat, double variance, std::mt19937_64& generator)
{
    normal_distribution<double> d(0, variance);
    for (int s = 0; s < numStates; s++)
        for (int a = 0; a < numActions; a++)
            mat(s, a) += d(generator);

}

// Ask the agent to select an action given the current state
int TabularBBO::getAction(const Eigen::VectorXd& s, std::mt19937_64& generator) const
{
    //cout << "Printing no size and rows and cols " << curThetas.size() << " " << curThetas.rows() << " " << curThetas.cols() << endl;
    //cout << "num elements vector " << curThetas.size() << endl;
    const int curAgent = epCount / totRunPerAgent;
    //cout << "Current episode is " << epCount << endl;
    //cout << "Current agent is " << curAgent << endl;
    //cout << "No reset episodes is " << N <<endl;
    const auto &newTheta = curThetas.at(curAgent);
	// Convert the one-hot state into an integer from 0 - (numStates-1)
	int state;
	for (state = 0; state < (int)s.size(); state++)
		if (s[state] != 0)
			break;
	assert(state != (int)s.size());					// If this happens, the s-vector was all zeros

	// Get the action probabilities from theta, using softmax action selection.
	VectorXd actionProbabilities = newTheta.row(state).array().exp();
	actionProbabilities.array() /= actionProbabilities.sum();

	// Sample an action from actionProbabilities.
	// The <double> below means that the object "uniform_real_distribution" is a "templated" object. That means
	// that the object itself is defined for a specific type. In this case, we are creating a "double" version of the object.
	// Here, you could use <float> to get a single-precision floating point generator rather than a double precision.
	// using #include <random> we get many different distributions. uniform_real_distribution<double> is one. Its constructor
	// takes the bounds of the uniform real distribution, in this case [0,1]. Below, "uniform_real_distribution<double>(0, 1)" creates
	// an object calling the constructor with (0,1), and then we immediately use it with (generator), which samples the distribution
	// using the provided random number generator
	//
	// A more common form is:
	// >>> uniform_real_distribution<double> myDistribution(0,1);
	// Then, you can sample this distribution at any time with:
	// >>> myDistribution(generator);
	// where "generator" is an object of type mt19937_64 (for example).
	//
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
void TabularBBO::reset(std::mt19937_64& generator)
{
	// You can chain together equal statements like this:
	//curTheta = newTheta = MatrixXd::Zero(numStates, numActions);
	//curThetaJHat = -INFINITY;
    curThetas.clear();
    curThetaJHats.clear();
    //cout << "Current totalEpisodes is " << totalEpisodes << endl;
    for(int i = 0; i < noAgents; i++)
	{
        curThetas.push_back(MatrixXd::Zero(numStates, numActions));
        addMatWithVar(curThetas.back(), initialVariance, generator);
        curThetaJHats.push_back(make_pair(0.0, i));
	}
	totalEpisodes = 0;
	curMuatationVariance = mutationVariance;
	epCount = 0;
	EpisodicAgent::reset(generator);	// See EpisodicAgent::reset. Here TabularRandomSearch is the subclass and EpisodicAgent is the superclass. EpisodicAgent has its own variables to reset, but calling "reset" calls the function for the subclass. So, this line is saying "also call the reset function for EpisodicAgent too!"
}



void TabularBBO::episodicUpdate(mt19937_64& generator)
{
    totalEpisodes += N;
    for(int i = 0; i < noAgents; i++)
    {
        curThetaJHats[i] = make_pair(0.0, i);
    }
    for(int ep = 0; ep < N; ep++)
    {
        int curAgent = ep / totRunPerAgent;
        double curGamma = 1.0;
        //curThetaJHats[curAgent].first = 0
        for(int t = 0; t < (int)rewards[ep].size(); t++)
        {
            curThetaJHats[curAgent].first += curGamma * rewards[ep][t];
            curGamma *= gamma;
        }
    }
    std::sort(curThetaJHats.begin(), curThetaJHats.end(), std::greater<std::pair<double, int> >());
    //cout << "Episode count is " << epCount << endl;
    /*if(totalEpisodes == maxEps)
    {
        cout << "Best reward is " << curThetaJHats[0].first << endl;
    }*/
    /*for(int i = 0; i < noAgents; i++)
    {
        cout << "Reward for agent no " << i << " " << curThetaJHats[i].first << endl;
    }*/
    //exit(0);
    vector<MatrixXd> fittest(noSurvivingAgents);
    //cout << "Printing rewards for episode " << totalEpisodes << endl;
    for(int i = 0; i < noAgents; i++)
    {
        //cout << "Current episode " << totalEpisodes << endl;
        //cout << "Reward " << curThetaJHats[i].first << "; Agent No: " << curThetaJHats[i].second << endl;
    }
    for(int i = 0; i < noSurvivingAgents; i++)
    {
        int survivorIdx = curThetaJHats[0].second;
        if(totalEpisodes < (int)(maxEps*0.9))
        {
            survivorIdx = curThetaJHats[i].second;
        }
        MatrixXd survivor = curThetas[survivorIdx];
        fittest[i] = survivor;

    }
    for(int i = 0; i < noSurvivingAgents; i++)
    {
        curThetas[i] = fittest[i];
    }
    if(totalEpisodes == maxEps)
    {
        //cout << "Fittest zero has reward " << curThetaJHats[0].first / totRunPerAgent << endl;
    }
    curMuatationVariance *= mutationMultiplier;
    for(int i = noSurvivingAgents; i < noAgents; i++)
    {
        if(totalEpisodes < (int)(maxEps*0.9))
        {
            int curAgent = i % noSurvivingAgents;
            curThetas[i] = fittest[curAgent];
            //cout << "Current fittest" << endl;
            //cout << fittest[curAgent] << endl;
            double variance = curMuatationVariance;
            //cout << "Creating agent with variance " << variance << endl;
            addMatWithVar(curThetas[i], variance, generator);
            //cout <<"After adding variance" << endl;
            //cout << "fittest\n" << fittest[curAgent] << endl;
            //cout << "curTheta\n" << curThetas[i] << endl;
        }
        else
        {
            //cout << "Choosing the best agent" << endl;
            curThetas[i] = fittest[0];
        }
    }
}
