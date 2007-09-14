/** David Lewis's thresholding option B2.
 */
#define  _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <limits>
#include <iomanip>

#include "logging.h"
#include "Design.h"
#include "ModelTypeParam.h"
#include "stats.h"

using namespace std;

/**
This threshold tuning method maximizes the expected value of a
classifier effectiveness measure on the Training Population.

Currently, the only such measure supported is F1 (the F_beta measure
with beta=1), but other measures (T11NU, etc.) could easily be added.

To estimate F1,  we follow the procedure described in
David Lewis' paper, "Evaluating and Optimizing Autonomous Text
Classification Systems" (SIGIR 95), eq. (17):

E[F_beta] = C_F * Product_{i=1,..,n}(1-p_i),   
            if \vec{s} = 0 
E[F_beta] = (beta^2+1) sum_{i=1,...,n}{s_i*p_i}/sum_{i=1,..,n}(s_i+beta^2*p_i),
            otherwise


When documents are ordered in descending order of p_i, and the 
selection threshold is between p_k and p_{k-1}, we have the system
decision variable s_i = 1 iff i<k. Thus

E[F_beta] = (beta^2+1) sum_{i=1,...,k}{p_i}/(k + beta^2* \sum{i=1,...,n}(p_i))

We use the constants C_F=1, beta=1. (beta is just a real constant, 
and has nothing to do with Bayesian feature vector \vec{beta}).  */

class RevCmp {
public:
  bool operator()(const pair<double,double> & s1, 
		  const pair<double,double> & s2)
    const  {
    return (s2.first < s1.first);  // reverse sort order
  }
};


/** Tunes threshold based on estimiated probabilities on the large
    Training Population.
    @param score_and_p_hat Each element of the vector contains the
    score and the estimated probability (the p^ value) of
    @return Returns the threshold in the "score" space. 
 */
  const static double beta=1.0,  beta2 = beta * beta;
double tuneThresholdEst( vector<pair<double,double> > &score_and_p_hat,
			 const class ModelType& modelType) {
  const static double C_F = 1.0;

  const int kLast = score_and_p_hat.size()-1;

  class Crit {
    ModelType::thrtune thrTune;
    double allP; // sum_j(p[j])   = E[TP + FN] 
    double allN; // sum_j(1-p[j]) = E[FP + TN]
    /** We can compute other coefficients as follows:
	(allP = TP + FN; allN=TN + FP; sysP=TP+FP)
	FN = allP - TP
	FP = sysP - TP
	TN = allN - FP = allN - sysP + TP
    */
  public:
    Crit(double allP_, double allN_, 	 ModelType::thrtune thrTune_)
      : allP(allP_), allN(allN_),      thrTune(thrTune_) {}
    /** Returns the classifier effectiveness  measure in question. 
	The greater the value, the better the classifier.
	@param sysP; // = E[TP + FP]
	@param TP; // = sum_{j<=k}(p[j]) = E[TP]
    */
    inline double eval( int sysP, double TP)  const { //always minimize
      switch(thrTune) {
      case ModelType::thrSumErr: 
	// - (FP + FN)
	return 2*TP-allP-sysP;
      case ModelType::thrBER: 
	// balanced error rate: 
	// ber= 0.5*[fp/(tn+fp)+fn/(tp+fn)] = 0.5 (FP/allN + FN/allP)
	// We return  -2*BER*allN*allP = -(FP*allP + FN*allN)
	return - ( (sysP-TP)*allP + (allP - TP)*allN);
	
      case ModelType::thrT11U: 
	// T11U = 2*TP - FP = 3*TP - sysP
	{ 
	  double FP = sysP - TP;
	  return  2*TP - FP;
	}
      case ModelType::thrT13U: 
	// 20.0*TP - FP;
	{ 
	  double FP = sysP - TP;
	  return  20*TP - FP;
	}
      case ModelType::thrF1: 
	return (beta2+1)*TP / (sysP + beta2*allP);
      default: 
	throw logic_error("Threshold tuning rule wrong or undefined");
      }
    }
  };


  // Sort probabilities in descending order
  sort(score_and_p_hat.begin(), score_and_p_hat.end(), RevCmp());

  // E[F(k=0)]
  double prodN = 1;
  double allP = 0;
  double allS=0;
  for(unsigned k=0; k<score_and_p_hat.size(); k++) {
    double p = score_and_p_hat[k].second;
    prodN  *= (1.0 - p);
    allP += p;
    allS += score_and_p_hat[k].first;
  }

 
  Log(4) << endl
	 << "B2 thresholding: |TrainPop|=" 
	 << score_and_p_hat.size() << " docs";
  Log(4) << endl
	 << "B2 thresholding: " 
	 <<  score_and_p_hat[0].first << "<=s<= "
	 <<  score_and_p_hat[kLast].first << "; avg(s)="
	 << (allS/score_and_p_hat.size()) << "; " 
	 <<  score_and_p_hat[0].second << "<=p<="
	 <<  score_and_p_hat[kLast].second << "; avg(p)="
	 << (allP/score_and_p_hat.size());

  Crit crit(allP, score_and_p_hat.size()-allP, modelType.TuneThreshold());
    
  // kBest=k means that it's best to have the threshold between p[k] and p[k+1]
  int kBest = -1;
  /** For E[F1], a special formula is used when the classifier selects nothing
   */
  double eBest = 
    (modelType.TuneThreshold() == ModelType::thrF1) ? 
    C_F * prodN : crit.eval(0,0);


  Log(4) << "prodN=" << prodN << endl;
  Log(4) << "(start) kBest=" << kBest <<", eBest=" << eBest << endl;

  // pSum=sum_{j=1,...,k}(p[j]) = E[TP] 
  double pSum=0;  
  for( unsigned k=0; k<score_and_p_hat.size(); k++) {
    double p = score_and_p_hat[k].second;
    pSum += p;
    double e = crit.eval(k+1, pSum);
    if (e >= eBest) {
      eBest = e;
      kBest = k;
      Log(4) << "kBest=" << kBest <<", p="<< p << ",eBest=" << eBest << endl;
    } else if (e < eBest && k > 1) {
      // we have passed the maximum already
      break;
    }
  }

  double threshold;
  if (kBest == -1) {
    // the best threshold is one that lets nobody in
    threshold =  numeric_limits<double>::max();
    Log(4) << endl 
	   << "B2 thresholding: choosing 'select-none'. s[1]="
	   << score_and_p_hat[0].first<<", p^[0]="
	   << score_and_p_hat[0].second;
  } else if (kBest == kLast) {
    // let everyone of them in...
    threshold =  -numeric_limits<double>::max();
    Log(4) << endl 
	   << "B2 thresholding: choosing 'select-all'. s[last]="
	   << score_and_p_hat[kLast].first<<", p^[last]="
	   << score_and_p_hat[kLast].second;
  } else {
    threshold = (score_and_p_hat[kBest].first + 
		 score_and_p_hat[kBest+1].first)/2.0;
    Log(4) << endl 
	   << "B2 thresholding: threshold between "<<kBest<<" and "<<(kBest+1) 
           << "; s=("<< score_and_p_hat[kBest].first << " : " 
	   << score_and_p_hat[kBest+1].first << ") " 
           <<", p^=("<<score_and_p_hat[kBest].second << " : " 
	   << score_and_p_hat[kBest+1].second << ")";
 
  }
  Log(4) << "; E[F1]=" << (eBest*100) << "%" << endl;
  return threshold;
}


/*
    Copyright (c) 2002, 2003, 2004, 2005, 2006, 2007, Rutgers University, New Brunswick, NJ, USA.

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
    BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
    ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    Except as contained in this notice, the name(s) of the above
    copyright holders, DIMACS, and the software authors shall not be used
    in advertising or otherwise to promote the sale, use or other
    dealings in this Software without prior written authorization.
*/
