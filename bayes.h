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
        class RowSetMem & trainData,
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
#ifdef EM_ENABLED
void runSparseEMprobit( const char* topic,
                       DenseData & trainData,
                       const class BayesParameter& bayesParameter,
                       const class PriorTermsByTopic& priorTermsByTopic,
                       const class DesignParameter& designParameter,
                       const class ModelType& modelType,
                       IRowSet & testData,
                       std::ostream& result);
Matrix rbfkernel( const Matrix& xr, const Matrix& xc, double width);
Matrix rbfkernel( const Matrix& x, double width);
BoolVector applyModel( const Matrix& beta, const Matrix& tstx );
BoolVector showclassrbf( const Matrix& tstx, const Matrix& trnx,
                             const Matrix& beta, double width );
#endif //EM_ENABLED

#endif //SPARSE_BAYES_PROBIT_

/*
    Copyright 2005, Rutgers University, New Brunswick, NJ.

    All Rights Reserved

    Permission to use, copy, and modify this software and its documentation for any purpose 
    other than its incorporation into a commercial product is hereby granted without fee, 
    provided that the above copyright notice appears in all copies and that both that 
    copyright notice and this permission notice appear in supporting documentation, and that 
    the names of Rutgers University, DIMACS, and the authors not be used in advertising or 
    publicity pertaining to distribution of the software without specific, written prior 
    permission.

    RUTGERS UNIVERSITY, DIMACS, AND THE AUTHORS DISCLAIM ALL WARRANTIES WITH REGARD TO 
    THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
    ANY PARTICULAR PURPOSE. IN NO EVENT SHALL RUTGERS UNIVERSITY, DIMACS, OR THE AUTHORS 
    BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER 
    RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, 
    NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR 
    PERFORMANCE OF THIS SOFTWARE.
*/
