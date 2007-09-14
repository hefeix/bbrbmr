#ifndef SPARSE_BAYES_LOGISTIC_
#define SPARSE_BAYES_LOGISTIC_

#include "Matrix.h"
#include "data.h"
#include "BayesParamManager.h"
#include "ModelTypeParam.h"

Vector score( const Vector& beta, const Matrix& tstx );
double dotSparseByDense( const SparseVector& x, const Vector& b );
double tuneThreshold( const vector<double>& score, const BoolVector& y, const class ModelType& modelType );

double tuneThresholdEst( vector<pair<double,double> >& score_and_p_hat, const class ModelType& modelType );  // VM

void  displayEvaluation( ostream& o, const BoolVector& y, const BoolVector& prediction, 
                        const string& comment =string() );
void  displayEvaluationCT( ostream& o, int TP, int FP, int FN, int TN );
double calcROC( std::vector< std::pair<double,bool> >& forROC );
/*void testModel( const char* topic,
                const class ModelType& modelType,
                const Vector& beta,
                bool thresholdScore, double threshold,
                IRowSet & testDesignData,
                class ResultsFile& resFile ); //std::ostream& result, ResultFormat resultFormat );*/
double PointLogLikelihood( double linscore, bool y, const class ModelType& modelType );

typedef vector< vector<double> > THierW;
double dotProductHier( const SparseVector& x, const vector<bool>& groups, const THierW& w );
class ZOLRModel {
    bool m_bTrained;
    std::string m_topic;
    class IDesign* m_pDesign;
    //vector<int> m_featSelect;
    Vector m_beta;
    double m_threshold;
    class ModelType m_modelType;
    BayesParameter m_bayesParam;
    BayesParameter m_bayesParam2; //hier
    bool m_bHier;
    THierW m_hierW;
public:
    void Train( const char* topic,
        RowSetMem & trainData,
		IRowSet*  trainPopData,  // VM
        //const class BayesParameter& bayesParameter,
        const class HyperParamPlan& hyperParamPlan,
        const class HyperParamPlan& hyperParamPlan2,
        const class Squeezer& squeezer,
        const class PriorTermsByTopic& priorTermsByTopic,
        const class DesignParameter& designParameter,
        const class ModelType& modelType,
        class WriteModel& modelFile,
        class ResultsFile& resFile ); //std::ostream& result, ResultFormat resultFormat );
    void Restore( class ReadModel& modelFile, const INameResolver& names );
    void Test( IRowSet & TestRowSet, 
        class ResultsFile& resFile,  //std::ostream& result, ResultFormat resultFormat, 
        double probTestThreshold=-1 );
    ZOLRModel();
    ~ZOLRModel();
private:
    double dotProduct ( const SparseVector& x ) const;
    void tuneModel(   IRowSet & drs,
                    //const class BayesParameter& bayesParameter,
                    const HyperParamPlan& hyperParamPlan,
                    const Vector& priorMean, const Vector& priorScale, const Vector& priorSkew,
                    class Stats& stats );
    void tuneHierModel(   IRowSet & drs,
                    const HyperParamPlan& hyperParamPlan,
                    const HyperParamPlan& hyperParamPlan2,
                    const Vector& priorMode, const Vector& priorScale, const Vector& priorSkew,
                    Stats& stats );
    void squeezerModel(   IRowSet & drs,
                    const BayesParameter& bayesParameter,
                    unsigned squeezeTo,
                    double hpStart,
                    const Vector& priorMean, const Vector& priorScale, const Vector& priorSkew );
    void testModel (
                    IRowSet & drs,
                    ResultsFile& resFile, 
                    double scoreThreshold ) const;
};

#ifdef LAUNCH_SVM
void runSVMlight( const char* topic,
                       RowSetMem & trainData,
                       const class ModelType& modelType,
                       HyperParamPlan& hyperParamPlan,
                       IRowSet & testData,
                       std::ostream& modelFile,
                       std::ostream& result);
void runSVM_cv( const char* topic,
                       RowSetMem & trainData,
                       const class ModelType& modelType,
                       HyperParamPlan& hyperParamPlan,
                       IRowSet & testData,
                       std::ostream& svmModelFile,
                       std::ostream& result);
#endif //LAUNCH_SVM

#endif //SPARSE_BAYES_PROBIT_


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
