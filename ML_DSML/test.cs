using System;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace ML_DSML
{
    public class DSMLCodeGen
    {
        public static void Main(string[] args)
        {
            // The labeled training data
	    	double[] strikes = { 2, 1, 2, 0, 1, 0 };
	    	bool[] fastball = { true, false, true, true, false, false };
	    	double[] balls = { 3, 3, 0, 2, 1, 0 };
	    	double[] outs = {0, 1, 1, 2, 0, 2};
	    	double[] scoreDifferential = {0, 3, 2, 1, 1, 2};
	    	double[] basesLoaded = {1, 0, 0, 0, 1, 0};
	    	double[] inning = {1, 5, 7, 2, 1, 4};
	    	double[] handedness = {1, 1, 1, 1, 1, 0};
	    	double[] numPitches = {18, 32, 9, 1, 88, 34};
	    	double[] homePrior = {.25, .36, .88, .75, .99, .92};
	    	double[] battingTeamPrior = {.15, .78, .59, .75, .14, .06};
	    	double[] countPrior = {.89, .18, .72, .72, .1, .49};
	    	double[] batterPrior = {.7, .53, .91, .56, .9, .8};
	    	double[] homePriorSupport = {180, 58, 75, 234, 75, 250};
	    	double[] battingTeamPriorSupport = {44, 58, 59, 204, 167, 61};
	    	double[] countPriorSupport = {32, 237, 151, 56, 58, 131};
	    	double[] batterPriorSupport = {193, 226, 244, 9, 153, 122};
	    	double[] batterSlugging = {.182, .493, .337, .155, .157, .372};
	    	double[] batterRuns = {37, 84, 12, 35, 53, 74};
	    	double[] previousPitchType = {1, 0, 1, 1, 0, 1};
	    	double[] previousPitchResult = {1, 0, 0, 1, 1, 0};
	    	double[] previousPitchVelocity = {87, 85, 89, 91, 97, 101};
	    	double[] previousPitchVelocityGradient = {12, 1, -13, 1, 13, 12};

            // Create target vector and weights
            VariableArray<bool> y = Variable.Observed(fastball).Named("y");
            Variable<Vector> w = Variable.Random(new VectorGaussian(Vector.Zero(23),
                PositiveDefiniteMatrix.Identity(23))).Named("w");

            BayesPointMachine(strikes, balls, outs, scoreDifferential, basesLoaded, inning, handedness, numPitches, homePrior, battingTeamPrior, countPrior, batterPrior, homePriorSupport, battingTeamPriorSupport, countPriorSupport, batterPriorSupport, batterSlugging, batterRuns, previousPitchType, previousPitchResult, previousPitchVelocity, previousPitchVelocityGradient, w, y);

            //create inference engine object and infer posterior distribution 
            //of weights
            InferenceEngine engine = new InferenceEngine();
            VectorGaussian wPosterior = engine.Infer<VectorGaussian>(w);
            Console.WriteLine("Dist over w=\n" + wPosterior);

            //make predictions on test data
            double[] strikesTest = { 2, 1, 2 };
            double[] ballsTest = { 3, 2, 0 };
            double[] outsTest = {1, 1, 0, 0, 2, 2};
            double[] scoreDifferentialTest = {1, 0, 1, 2, 1, 3};
            double[] basesLoadedTest = {0, 0, 0, 1, 0, 0};
            double[] inningTest = {3, 1, 3, 2, 9, 6};
            double[] handednessTest = {0, 1, 1, 0, 1, 1};
            double[] numPitchesTest = {36, 20, 29, 11, 68, 5};
            double[] homePriorTest = {.05, .87, .77, .62, .86, .70};
            double[] battingTeamPriorTest = {.4, .45, .58, .02, 1, .7};
            double[] countPriorTest = {.03, .38, .92, .06, .7, .23};
            double[] batterPriorTest = {.08, .1, .04, .59, .90, .11};
            double[] homePriorSupportTest = {10, 115, 104, 166, 171, 173};
            double[] battingTeamPriorSupportTest = {162, 151, 27, 207, 144, 163};
            double[] countPriorSupportTest = {38, 22, 114, 130, 195, 182};
            double[] batterPriorSupportTest = {140, 190, 57, 233, 194, 9};
            double[] batterSluggingTest = {.237, .172, .222, .2, .393, .282};
            double[] batterRunsTest = {24, 48, 90, 54, 89, 75};
            double[] previousPitchTypeTest = {1, 1, 1, 0, 1, 1};
            double[] previousPitchResultTest = {1, 0, 1, 0, 0, 0};
            double[] previousPitchVelocityTest = {94, 88, 84, 101, 101, 110};
            double[] previousPitchVelocityGradientTest = {-5, -11, -2, -9, 1, 1};
            VariableArray<bool> ytest = Variable.Array<bool>(new Range(strikesTest.Length)).Named("ytest");
            BayesPointMachine(strikesTest, ballsTest, outsTest, scoreDifferentialTest, basesLoadedTest, inningTest, handednessTest, numPitchesTest, homePriorTest, battingTeamPriorTest, countPriorTest, batterPriorTest, homePriorSupportTest, battingTeamPriorSupportTest, countPriorSupportTest, batterPriorSupportTest, batterSluggingTest, batterRunsTest, previousPitchTypeTest, previousPitchResultTest, previousPitchVelocityTest, previousPitchVelocityGradientTest, Variable.Random(wPosterior).Named("w"), ytest);
            Console.WriteLine("output=\n" + engine.Infer(ytest));

        }

        public static void BayesPointMachine(double[] strikes, double[] balls, double[] outs, double[] scoreDifferential, double[] basesLoaded, double[] inning, double[] handedness, double[] numPitches, double[] homePrior, double[] battingTeamPrior, double[] countPrior, double[] batterPrior, double[] homePriorSupport, double[] battingTeamPriorSupport, double[] countPriorSupport, double[] batterPriorSupport, double[] batterSlugging, double[] batterRuns, double[] previousPitchType, double[] previousPitchResult, double[] previousPitchVelocity, double[] previousPitchVelocityGradient, Variable<Vector> w, VariableArray<bool> y)
        {
            // Create training data vector with bias parameter of 1
            Range j = y.Range;
            Vector[] xVector = new Vector[strikes.Length];
            for (int i = 0; i < xVector.Length; i++)
                xVector[i] = Vector.FromArray(strikes[i], balls[i], outs[i], scoreDifferential[i], basesLoaded[i], inning[i], handedness[i], numPitches[i], homePrior[i], battingTeamPrior[i], countPrior[i], batterPrior[i], homePriorSupport[i], battingTeamPriorSupport[i], countPriorSupport[i], batterPriorSupport[i], batterSlugging[i], batterRuns[i], previousPitchType[i], previousPitchResult[i], previousPitchVelocity[i], previousPitchVelocityGradient[i],  1);
            VariableArray<Vector> x = Variable.Observed(xVector, j).Named("x");

            // Bayes Point Machine, dot product of weights and feature vector
            double noise = 0.1;
            y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]).Named("innerProduct"), noise) > 0;
        }
    }
}