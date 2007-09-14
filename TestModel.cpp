// 2.02     Feb 22, 05  fixed bug with prob. threshold: need to convert into score threshold
// 2.07     May 16, 05  Groups - Hierarchical modeling

#include "bayes.h"
#include "Design.h"
#include "ModelFile.h"
#include "ResFile.h"

using namespace std;

void ZOLRModel::Restore( class ReadModel& modelFile, const INameResolver& names )
{
    m_topic = modelFile.Topic();
    //m_featSelect = modelFile.TopicFeats();
    m_modelType = modelFile.getModelType();

    if( modelFile.getDesignParameter().DesignType()==designPlain )
        m_pDesign= new PlainDesign( names );
    else if( modelFile.getDesignParameter().DesignType()==designInteractions ) {
        throw logic_error("Interactions Design not supported with ZO");
    }
    else
        throw logic_error("Undefined Design type");

    if( modelFile.IsHier() ) {
        m_bHier = true;
        unsigned nfeats = m_pDesign->dim();
        m_hierW = vector< vector<double> >( nfeats, vector<double>(modelFile.NGroups(),0.0) );
        for( TWHierSparse::const_iterator itr=modelFile.WHier().begin(); itr!=modelFile.WHier().end();
            itr++)
            for( unsigned k=0; k<itr->second.size(); k++ ) {
                pair<unsigned,double> group_w = itr->second.at(k);
                m_hierW.at(itr->first).at(group_w.first) = group_w.second;
            }
    }
    else {
        m_bHier = false;
        m_beta.resize( modelFile.Beta().size() );
        for( unsigned i=0; i<modelFile.Beta().size(); i++ )
            m_beta[i] = modelFile.Beta()[i];
    }

    m_threshold = modelFile.Threshold();

    m_bTrained = true;
}

ZOLRModel::ZOLRModel() 
    : m_pDesign(0), m_bTrained(false) {}
ZOLRModel::~ZOLRModel() { delete m_pDesign; }

double dotProductHier( const SparseVector& x, const vector<bool>& groups, const THierW& w )
/* assumption: 'SparseVector' is indexed by offset in 'featSelect', i.e. indices start from 0
 * This is guaranteed by DesignRowSet
 */
{
    double score = 0.0;
    try{
    for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
        for( unsigned g=0; g<groups.size(); g++ )
            if( groups.at(g) )
                score += w.at(ix->first).at(g) * ix->second;
    }catch(...){
        throw logic_error("dotProductHier: sparse vector index beyond dense vector size"); }
    //Log(8)<<"\ndotProd x "<<x<<"\ndotProd b "<<b<<"\ndotProd="<<score;
    return score;
}

double ZOLRModel::dotProduct( const SparseVector& x ) const
/* assumption: 'SparseVector' is indexed by offset in 'featSelect', i.e. indices start from 0
 * This is guaranteed by DesignRowSet
 */
{ return dotSparseByDense( x, m_beta );
}

double dotSparseByDense( const SparseVector& x, const Vector& b )
/* assumption: 'SparseVector' is indexed by offset in 'featSelect', i.e. indices start from 0
 * This is guaranteed by DesignRowSet
*/
{
    double score = 0.0;
    try{
    for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
        score += b[ix->first] * ix->second;
    }catch(...){
        throw logic_error("dotSparseByDense: sparse vector index beyond dense vector size"); }
    //Log(8)<<"\ndotProd x "<<x<<"\ndotProd b "<<b<<"\ndotProd="<<score;
    return score;
}

static inline double logoneplusexp( double s ) {
    double e = exp( s );
    if( e>=numeric_limits<double>::max()-1 ) //!_finite(e)
        return s;
    else return log(1+e);
}

double PointLogLikelihood(
                double linscore,
                bool y,
                const class ModelType& modelType
            )
{
    double pointLogLhood;
    if( ModelType::logistic==modelType.Link() )
        pointLogLhood = y ? linscore-logoneplusexp(linscore) : -logoneplusexp(linscore);
    else if( ModelType::probit==modelType.Link() )
        pointLogLhood = y ? log(combinedProbNorm(linscore)) : log(1-combinedProbNorm(linscore));
    else
        throw logic_error("only logistic and probit links supported for likelihood calculations");
    //Log(10)<<"\nPointLikelihood: y p lhood "<<y<<" "<<(y?exp(pointLogLhood):1-exp(pointLogLhood))<<" "<<exp(pointLogLhood)<<"\t";

    return pointLogLhood;
}

double  T11F( int TP, int FP, int FN, int TN )
{
    return 0==TP+FP ? 0.0 : 1.25*TP /( 0.25*FN + FP + 1.25*TP );
}
double  T11SU( int TP, int FP, int FN, int TN )
{
    int T11U = 2*TP - FP;
    double T11NU = 100.0 * T11U /( 2*(TP+FN) );
    return( T11NU>-.5 ? (T11NU+.5)/1.5 : 0 ) ;
}
double tuneThreshold( const vector<double>& score, const BoolVector& y, const ModelType& modelType )
{
    if( modelType.TuneThreshold()==ModelType::thrNo ) {//default thresholding by model
        if( ModelType::SVM==modelType.Link() )
            return 0;
        else if( ModelType::logistic==modelType.Link() ) {
            double p=modelType.ProbThreshold();
            if( 0==p )          return -numeric_limits<double>::max();
            else if( 1==p )     return numeric_limits<double>::max();
            else                return log( p/(1-p) ); //was 0.5;
        }
        else throw logic_error("only logistic link supported for probability threshold");
    }
    class Crit {
        ModelType::thrtune thrTune;
        const unsigned& TP;
        const unsigned& FP;
        const unsigned& FN;
        const unsigned& TN;
    public:
        Crit( const unsigned& TP_, const unsigned& FP_, const unsigned& FN_, const unsigned& TN_, 
            ModelType::thrtune thrTune_)
            : TP(TP_), FP(FP_), FN(FN_), TN(TN_), thrTune(thrTune_) {}
        double eval() const { //always minimize
            switch(thrTune) {
                case ModelType::thrSumErr: return FP + FN;
                case ModelType::thrBER: return FP*(TP+FN) + FN*(TN+FP);
                    //balanced error rate: [fp/(tn+fp)+fn(tp+fn)]/2
                case ModelType::thrT11U: return - 2.0*TP + FP;
                case ModelType::thrT13U: return - 20.0*TP + FP;
                case ModelType::thrF1: return double(FP + FN) / TP;
                default: throw logic_error("Threshold tuning rule wrong or undefined");
            }
        }
    };
        
    size_t n = y.size();
    if( n != score.size() )
        throw DimensionConflict(__FILE__,__LINE__);
    
    vector< pair<double,bool> > s( n );
    for( unsigned i=0; i<n; i++ )
        s[i] = pair<double,bool>( score[i], y[i] );

    std::sort( s.begin(), s.end() );

    //at first, "retrieve" all
    unsigned TP = ntrue(y);
    unsigned FP = n - ntrue(y);
    unsigned FN = 0;
    unsigned TN = 0;
    Crit crit( TP, FP, FN, TN, modelType.TuneThreshold() );
    
    double critBest = crit.eval(); //FP + FN; //sum of errors - 2.0*TP + FP; //-T11SU( TP, FP, FN, TN );  
    double tBest = - numeric_limits<double>::max();

    for( unsigned i=0; i<n; i++ )
    {
        if( 0!=i && s[i].first>s[i-1].first ) //take only 1st in a string of equal values
        {
            double critCurr = crit.eval(); //FP + FN; //- 2.0*TP + FP;
            if( critCurr < critBest ) {
                critBest = critCurr;
                tBest = (s[i].first + s[i-1].first) / 2;
                //Log(10)<<" - best "<<critBest;
            }
        }
        if( s[i].second ) {
            FN++; TP--;
        }else{
            FP--; TN++;
        }
    }

    //finally, check the "retrieve none" situation
    double critCurr = crit.eval(); //FP + FN; //- 2.0*TP + FP;
    if( critCurr < critBest ) {
        critBest = critCurr;
        tBest = s[n-1].first + 1; //what should we add here?
    }

    return tBest;
}


void ZOLRModel::Test( IRowSet & testData, 
                     ResultsFile& resFile,  //std::ostream& result, ResultFormat resultFormat,
                     double probTestThreshold )
{
    if( !m_bTrained )
        throw logic_error("Model not trained, unable to test");

    DesignRowSet drs( m_pDesign, testData );

    double scoreThreshold; 
    if( probTestThreshold>=0 ) { //param is active; convert threshold from prob to score
        scoreThreshold = 
            0==probTestThreshold ?      -numeric_limits<double>::max()
            : 1==probTestThreshold ?    numeric_limits<double>::max()
            :                           log( probTestThreshold/(1-probTestThreshold) );
    }
    else
        scoreThreshold = m_threshold;

    int TP=0, FP=0, FN=0, TN=0;
    double logLhood = 0;
    std::vector< std::pair<double,bool> > forROC;

    if( m_bHier ) Log(8)<<"\nm_hierW: "<<m_hierW;
    while( drs.next() ) //testData
    {
        // calculate prediction
        double predictScore = m_bHier ?
            dotProductHier( drs.xsparse(), drs.groups(), m_hierW )
            : dotProduct( drs.xsparse() );
        double p_hat =
                    ( m_modelType.Link()==ModelType::probit ? combinedProbNorm(predictScore) 
                    : 1.0/(1.0+exp(-predictScore)) );//logit
        bool prediction = predictScore>=scoreThreshold;
            //was:       thresholdScore ? predictScore>=threshold : p_hat>=threshold;

        if( prediction )
            if( drs.y() )  TP++; //testData
            else           FP++;
        else
            if( drs.y() )  FN++; //testData
            else           TN++;

        // loglikelihood
        logLhood += PointLogLikelihood( predictScore, drs.y(), m_modelType );
        //ROC
        if( forROC.capacity() - forROC.size() <= 1 ) //reduce re-allocations
            forROC.reserve( forROC.capacity() + 1000 );
        forROC.push_back( pair<double,bool>(predictScore,drs.y()) );

        // write results file
        resFile.writeline( m_topic, drs.currRowName(), true, //isTest, 
            drs.y(), predictScore, p_hat, prediction );
   }//test data loop

    Log(1)<<"\n\n---Validation results---";
    displayEvaluationCT( Log(1), TP, FP, FN, TN );
    Log(1)<<"\nTest set loglikelihood "<<logLhood<<" Average "<<logLhood/(TP+FP+FN+TN);

    //roc
    double area = calcROC(forROC);
    Log(1)<<"\nROC area under curve "<<area;

    Log(3)<<endl<<"Time "<<Log.time();
}

double calcROC( std::vector< std::pair<double,bool> >& forROC ) //can't be const: need sort
{
    std::sort( forROC.begin(), forROC.end() );
    double area = 0;
    double x=0, xbreak=0;
    double y=0, ybreak=0;
    double prevscore = - numeric_limits<double>::infinity();
    for( vector< pair<double,bool> >::reverse_iterator ritr=forROC.rbegin(); ritr!=forROC.rend(); ritr++ )
    {
        double score = ritr->first;
        bool label = ritr->second;
        if( score != prevscore ) {
            area += (x-xbreak)*(y+ybreak)/2.0;
            xbreak = x;
            ybreak = y;
            prevscore = score;
        }
        if( label )  y ++;
        else     x ++;
    }
    area += (x-xbreak)*(y+ybreak)/2.0; //the last bin
    if( 0==y || x==0 )   area = 0.0;   // degenerate case
    else        area = 100.0 * area /( x*y );
    return area;
}

    /*
        Trec notation (MMS notation):
                            Relevant        Not Relevant
            Retrieved          R+ / A (TP)       N+ / B (FP)
            Not Retrieved      R- / C (FN)       N- / D (TN)

        Text categorization tradition (Zhang&Oles):
            precision = TP / (TP + FP)
            recall  = TP / (TP + FN)
            F1 = 2*precision*recall/(precision+recall) == harmonic mean

        PR tradition:
            sensitivity = TP / (TP + FN)    = recall
            specificity = TN / (TN + FP)

        TREC:
            T11U = 2*TP - FP
            T11NU = T11U / MaxU = T11U /( 2*(TP+FN) )
            T11SU = ( max(T11NU,-.5) + .5 ) / 1.5

                    {          0                 if TP=FP=0
                    {
            T11F =  {       1.25*TP
                    { ----------------------     otherwise
                    { 0.25*FN + FP + 1.25*TP
    */
void  displayEvaluation( ostream& o, const BoolVector& y, const BoolVector& prediction, const string& comment )
{
    if( y.size() != prediction.size() )
        throw DimensionConflict(__FILE__,__LINE__);

    int TP=0, FP=0, FN=0, TN=0;
    /*int TP = ntrue( y&&prediction );
    int FP = ntrue( !y&&prediction );
    int FN = ntrue( y && !prediction );
    int TN = ntrue( !y && !prediction );*/
    for( unsigned i=0; i<y.size(); i++ )
        if( prediction[i] )
            if( y[i] )  TP++; //testData
            else        FP++;
        else
            if( y[i] )  FN++; //testData
            else        TN++;
    displayEvaluationCT( o, TP, FP, FN, TN );
    if( comment.length() )
        o<<endl<<comment;
}

void  displayEvaluationCT( ostream& o, int TP, int FP, int FN, int TN )
{
    o<<"\nConfusion matrix:   Relevant\tNot Relevant"
        <<"\n\tRetrieved    \t"<<TP<<"\t"<<FP
        <<"\n\tNot Retrieved\t"<<FN<<"\t"<<TN;
    double precision = (TP + FP)>0 ? 100.0*TP/(TP + FP) : 100;
    o<<"\nPrecision = "<< precision;
    double recall  = (TP + FN)>0 ? 100.0*TP/(TP + FN) : 100;
    o<<"\nRecall  = "<< recall;
    double F1  = (2*TP+FN+FP)>0 ? 100.0*2*TP/(2*TP+FN+FP) : 100;
    o<<"\nF1 = "<< F1;        //2*precision*recall/(precision+recall);
    int T11U = 2*TP - FP;
    o<<"\nT11U = "<< T11U;
    double T11NU = double(T11U) /( 2*(TP+FN) );
    o<<"\nT11NU = "<< T11NU;
    o<<"\nT11SU = "<<( T11NU>-.5 ? (T11NU+.5)/1.5 : 0 );
    o<<"\nT11F = "<<T11F( TP, FP, FN, TN );
    o<<"\nT13U = "<< 20*TP - FP;
    o<<"\n% errors = "<< 100.0*(FP+FN) / (TP+FP+FN+TN);
}

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
