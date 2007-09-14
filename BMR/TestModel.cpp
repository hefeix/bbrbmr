#include <limits>
#include <algorithm>
#include <iomanip>

#include "logging.h"
#include "Matrix.h"
#include "data.h"
#include "poly.h"
#include "BayesParamManager.h"
#include "ModelTypeParam.h"
#include "Design.h"
#include "ModelFile.h"

using namespace std;

static void displayConfusionTable( ostream& o, INameResolver& names, const vector< vector<unsigned> >& CT );
static void displayCT2by2( ostream& o, int TP, int FP, int FN, int TN, double roc );
static double calcROC( const vector< vector<double> >& allScores,
    const vector<unsigned>& allYs, 
    unsigned k )
{
    unsigned n=allScores.size();
    vector< std::pair<double,bool> > forROC;
    for( unsigned i=0; i<n; i++ )
        forROC.push_back( pair<double,bool>( allScores.at(i).at(k), k==allYs.at(i)) );

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


unsigned argmax( const vector<double>& score ) {
    unsigned ret;
    double maxval = - numeric_limits<double>::max();
    for( vector<double>::const_iterator itr=score.begin(); itr!=score.end(); itr++ )
        if( *itr > maxval ) {
            maxval = *itr;
            ret = itr - score.begin();
        }
    return ret;
}
vector<double> estprob( const vector<double>& score ) {
    vector<double> ret;
    double denom = 0;
    for( vector<double>::const_iterator itr=score.begin(); itr!=score.end(); itr++ )  {
        double e = exp(*itr);
        denom += e;
        ret.push_back( e );
    }
    for( vector<double>::iterator itr=ret.begin(); itr!=ret.end(); itr++ )
        *itr /= denom;
    return ret;
}

double dotSparseByDense( const SparseVector& x, const vector<double>& b )
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

void testModel( const char* topic,
                const class ModelType& modelType,
                const IParamMatrix& beta,
                double threshold,
                IRowSet & drs,
                std::ostream& result, ResultFormat resultFormat )
{
    vector<unsigned> TP(drs.c(),0), FP(drs.c(),0), FN(drs.c(),0); //, TN(drs.c(),0);
    vector< vector<unsigned> > CT( drs.c(), vector<unsigned>( drs.c(), 0 ) );
    vector< vector<double> > allScores;
    vector<unsigned> allYs;
    double logLhood = 0;
    unsigned nAlien = 0; //cases with untrained class
    //vector< std::pair<double,bool> > forROC;
    //vector< pair<unsigned,unsigned> > confusions;

    unsigned n = 0;
    while( drs.next() ) //testData
    {
        // compute predictive score
        vector<double> predictScore( drs.c() );
        for( unsigned k=0; k<drs.c(); k++ )
            predictScore[k] = dotSparseByDense( drs.xsparse(), beta.classparam(k) );
        
        allScores.push_back( predictScore );
        allYs.push_back( drs.y() );

        // get the classes on the test data
        unsigned prediction = argmax( predictScore );
            //Log(12)<<endl<<predictScore<<":"<<prediction;
        if( drs.ygood() ) {
            CT [drs.y()] [prediction] ++;
            if( prediction==drs.y() )  
                TP[drs.y()]++;
            else {
                FN[drs.y()]++;
                FP[prediction]++;
            }
            logLhood += PointLogLikelihood( predictScore, drs.y(), modelType );
            n++;
        }
        else 
            nAlien++;

        /*if( confusions.capacity() - confusions.size() <= 1 ) //reduce re-allocations
            confusions.reserve( confusions.capacity() + 1000 );
        confusions.push_back( pair<unsigned,unsigned>(prediction,drs.y()) );*/

        result << drs.classId( drs.y() ) //label
            <<" "<< estprob( predictScore) //p_hat predictScore
            <<" "<<drs.classId( argmax(predictScore) ) //y_hat*/
	        <<endl;

    }//test data loop

    Log(1)<<"\n\n---Validation results---";
    Log(1)<<"\nCases of trained classes "<<n<<"  Other "<<nAlien<<"  Total "<<n+nAlien;
    Log(1)<<"\nConfusion Table:";
    displayConfusionTable( Log(1), drs, CT );
    Log(1)<<"\nTest set loglikelihood "<<logLhood<<" Average "<<logLhood/n;
    for( unsigned k=0; k<drs.c(); k++ ) {
        Log(1)<<"\n\nOne-vs-All view: Class "<<drs.classId(k);
        double roc = calcROC( allScores, allYs, k );
        displayCT2by2( Log(1), TP[k], FP[k], FN[k], n-TP[k]-FP[k]-FN[k]/*TN*/, roc );
    }

    /*/roc
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
    Log(1)<<"\nROC area under curve "<<area;*/
}

static void displayConfusionTable( ostream& o, INameResolver& names, const vector< vector<unsigned> >& CT )
{
    unsigned nAll=0, nCorrect=0;
    o<<endl<<setw(20)<<"Belongs\\Assigned ";
    for( unsigned k=0; k<names.c(); k++ )
        o<<setw(8)<<names.classId(k);
    for( unsigned k=0; k<names.c(); k++ ) {
        o<<endl<<setw(20)<<names.classId(k);
        for( unsigned r=0; r<names.c(); r++ ) {
            o<<setw(8)<<CT[k][r];
            nAll += CT[k][r];
            if( k==r )
                nCorrect += CT[k][k];
        }
    }
    o<<endl<<"Total errors: "<<nAll-nCorrect<<" out of "<<nAll<<", "<<100.0*(nAll-nCorrect)/nAll<<"%";
}
void makeConfusionTable( ostream& o, INameResolver& names, 
                           const vector<unsigned>& y, const vector<unsigned>& prediction )
{
    vector< vector<unsigned> > CT( names.c(), vector<unsigned>( names.c(), 0 ) );
    for( unsigned i=0; i<y.size(); i++ )
        CT [y[i]] [prediction[i]] ++;

    displayConfusionTable( o, names, CT );
}
void makeCT2by2( ostream& o, INameResolver& names, 
                           const vector<unsigned>& y, 
                           const vector< vector<double> >& allScores,
                           const vector<unsigned>& prediction )
{
    vector<unsigned> TP(names.c(),0), FP(names.c(),0), FN(names.c(),0); //, TN(drs.c(),0);
    unsigned n=0;
    for( unsigned i=0; i<y.size(); i++ ) {
        if( prediction.at(i)==y.at(i) )  
            TP.at(y.at(i))++;
        else {
            FN.at(y.at(i))++;
            FP.at(prediction.at(i))++;
        }
        n++;
    }

    for( unsigned k=0; k<names.c(); k++ ) {
        o<<"\n\nOne-vs-All view: Class "<<names.classId(k);
        double roc = calcROC( allScores, prediction, k );
        displayCT2by2( o, TP.at(k), FP.at(k), FN.at(k), n-TP.at(k)-FP.at(k)-FN.at(k)/*TN*/, roc );
    }
}

/*void  displayEvaluation( ostream& o, const vector<unsigned>& y, const vector<unsigned>& prediction, const string& comment )
{
    if( y.size() != prediction.size() )
        throw DimensionConflict(__FILE__,__LINE__);

    int TP=0, FP=0, FN=0, TN=0;
    //int TP = ntrue( y&&prediction );
    //int FP = ntrue( !y&&prediction );
    //int FN = ntrue( y && !prediction );
    //int TN = ntrue( !y && !prediction );
    for( unsigned i=0; i<y.size(); i++ )
        if( prediction[i] )
            if( y[i] )  TP++; //testData
            else        FP++;
        else
            if( y[i] )  FN++; //testData
            else        TN++;
    //displayEvaluationCT( o, TP, FP, FN, TN );
    if( comment.length() )
        o<<endl<<comment;
}*/

static double  T11F( int TP, int FP, int FN, int TN )
{
    return 0==TP+FP ? 0.0 : 1.25*TP /( 0.25*FN + FP + 1.25*TP );
}
static double  T11SU( int TP, int FP, int FN, int TN )
{
    int T11U = 2*TP - FP;
    double T11NU = 100.0 * T11U /( 2*(TP+FN) );
    return( T11NU>-.5 ? (T11NU+.5)/1.5 : 0 ) ;
}
static void  displayCT2by2( ostream& o, int TP, int FP, int FN, int TN, double roc )
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
    //int T11U = 2*TP - FP;
    //o<<"\nT11U = "<< T11U;
    //double T11NU = double(T11U) /( 2*(TP+FN) );
    //o<<"\nT11NU = "<< T11NU;
    //o<<"\nT11SU = "<<( T11NU>-.5 ? (T11NU+.5)/1.5 : 0 );
    //o<<"\nT11F = "<<T11F( TP, FP, FN, TN );
    o<<"\nROC area under curve "<<roc;
}

double PointLogLikelihood(
                const vector<double> & linscores,
                unsigned y,
                const class ModelType& modelType
            )
{
    double pointLogLhood;
    if( ModelType::logistic==modelType.Link() ) {
        double invProb = 1.0;
        for( unsigned c=0; c<linscores.size(); c++ )
            if( c!=y )
                invProb += exp( linscores[c] - linscores[y] );
        pointLogLhood = -log( invProb );
        Log(11)<<"\nscore/y/invProb/lhood "<<linscores<<" | "<<y<<" "<<invProb<<" "<<pointLogLhood;
    }
    else
        throw logic_error("only logistic link supported for likelihood calculations");

    return pointLogLhood;
}

double oldPointLogLikelihood(
                const vector<double> & linscores,
                unsigned y,
                const class ModelType& modelType
            )
{
    double pointLogLhood;
    if( ModelType::logistic==modelType.Link() ) {
        double s_exp_score = 0.0;
        //double yscore = exp( linscores[y] );
        double logscore;
        unsigned c;
        for( c=0; c<linscores.size(); c++ ) {
            double exp_score = exp( linscores[c] );
            if( exp_score>=numeric_limits<double>::max()-1 ) {//!_finite(e)
                logscore = linscores[c];
                break;
            }
            s_exp_score += exp_score;
        }
        if( c>=linscores.size() ) //no break occured in the loop
            logscore = log(s_exp_score);
        pointLogLhood = linscores[y] - logscore;
        Log(7)<<"\nscore/y/logscore/lhood "<<linscores<<" "<<y<<" "<<logscore<<" "<<pointLogLhood;
    }
    else
        throw logic_error("only logistic link supported for likelihood calculations");

    return pointLogLhood;
}

/*
 * class LRModel - part of implementation
 */

void LRModel::Restore( class ReadModel& modelFile, const INameResolver& names )
{
    m_featSelect = modelFile.TopicFeats();
    m_modelType = modelFile.getModelType();
    
    if( modelFile.getDesignParameter().DesignType()==designPlain )
        m_pDesign= new PlainDesign( names );
    else if( modelFile.getDesignParameter().DesignType()==designInteractions ) {
        throw logic_error("Interactions Design not supported with ZO");
    }
    else
        throw logic_error("Undefined Design type");

    modelFile.ReadParamMatrix( m_pDesign, m_beta );

    m_bTrained = true;
}
void LRModel::Test( IRowSet & testData, std::ostream& result, ResultFormat resultFormat )
{
    if( !m_bTrained )
        throw logic_error("Model not trained, unable to test");

    DesignRowSet testDesignData( m_pDesign, testData );
    testModel( m_topic.c_str(),
                m_modelType,
                m_beta,
                m_threshold,
                testDesignData,
                result, resultFormat);
    Log(3)<<endl<<"Time "<<Log.time();
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
