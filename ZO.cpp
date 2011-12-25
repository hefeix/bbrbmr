/*
 * Zhang & Oles regularized logistic regression
 * - column relaxation algorithm, aka Gauss-Zeidel, aka coordinate descent
 * Some implementation details by Dave Lewis
 *
 */
// 2.02     Feb 14, 05  reporting #feats left in model - bug with zero intercept
//                      reporting ROC area for training
//                      error bar rule for hyperparameter cv (option --errbar)
//                      no iterations limit
//                      fixed bug with convergence parameter
//                      fixed bug with 'wakeUpPlan' array, but sleeeping is off
// 2.03     Mar 01      reporting dropped intercept, ZO.cpp
// 2.04     Mar 03      accurate summation, OneCoordStep()
// 2.10     Jun 20, 05  one std error rule: fixed bug with stderr denom calc //ZO.cpp
// 2.50     Jun 20, 05  this ver is the umbrella for "Groups - Hierarchical modeling"
// 2.51     Jun 27, 05  bug: cv with default hyperpar on 1st level - fixed
// 2.55     Jul 05, 05  hierarchical fitting init by ordinary fit
// 2.56     Jul 10, 05  hierarchical Laplace redesigned
// 2.57     Aug 15, 05  bug: zero ind.prior var with normal prior - fixed
// 2.60     Oct 05, 05  hyperparameter search without a grid
// 2.61     Nov 05, 05  final prior variance reporting for bootstrap
// 2.62     Nov 10, 05  hyperparameter autosearch: observations weighted by inverse stddev
//                      fixed log.prior with infinite prior var
// 3.01     Jan 31, 06  fixed bug: Gaussian penalty should be 1/(2*var), not 1/var

#define  _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <algorithm>
#include <limits>
#include <iomanip>

#include "logging.h"
#include "Matrix.h"
#include "BayesParamManager.h"
#include "PriorTerms.h"
#include "dataBin.h"
#include "Design.h"
#include "bayes.h"
#include "ModelTypeParam.h"
#include "stats.h"
#include "ModelFile.h"
#include "Squeezer.h"
#include "StrataSplit.h"
#include "ResFile.h"

using namespace std;

static Vector Stddevs( IRowSet & drs );
double AvgSquNorm( class IRowSet& rs ); //for default bayes parameter: avg x*x
static double LogLikelihood(
                IRowSet & trainDesignData, // HACK!!! DesignRowSet needed: colId's are 0, 1, 2... 
                const class ModelType& modelType,
                const Vector& beta
            );
static double LogLikelihoodHier(
                IRowSet & rowset, // HACK!!! DesignRowSet needed: colId's are 0, 1, 2... 
                const class ModelType& modelType,
                const THierW& w
            );
static double LogPrior(
                const BayesParameter& bayesParameter,
                const Vector& priorMode,
                const Vector& priorScale,
                const Vector& priorSkew,
                const vector<double>& beta
            );
static pair<double,double> LogPrior(
                const BayesParameter& bayesParameter,
                const BayesParameter& bayesParameter2,
                const Vector& priorMode,
                const Vector& priorScale,
                const Vector& priorSkew,
                const THierW& w
            );
static double LogTraceBased(
                IRowSet & trainDesignData,
                const class ModelType& modelType,
                const BayesParameter& bayesParam,
                const Vector& priorMode,
                const Vector& priorScale,
                const Vector& beta
            );
static void ReportSparsity( ostream& o, const THierW& w, const Vector& priorMode )
{
    int nparams = w.size() * w.at(0).size();
    int nparamsdropped=0, ndesignsdropped=0;
    bool intcptdropped=false;
    for( unsigned j=0; j<w.size(); j++ ) {
        bool designdropped = true;
        const vector<double>& wj = w.at(j);
        for( unsigned g=0; g<wj.size(); g++ )
            if( wj.at(g)==priorMode.at(j) )   nparamsdropped++;
            else designdropped = false;
        if( designdropped )
            ndesignsdropped++;
        if( designdropped && j==w.size()-1 ) //HACK: design aware
            intcptdropped = true;
    }
    o<<"\nParameters at prior mode: "<<nparamsdropped<<" out of "<<nparams
        <<"\nFeatures dropped: "<< ndesignsdropped-(intcptdropped?1:0) <<" (all related parameters at prior mode; intercept not included)";
    if( intcptdropped )  o<<"\nIntercept dropped (all related parameters at prior mode)";
}

struct triplet{
    unsigned var;
    unsigned row;
    double val;
    triplet( unsigned var_, unsigned row_, double val_) 
        : var(var_), row(row_), val(val_) {};
    unsigned i() const {return row;}
    double x() const {return val;}
};
ostream& operator<<( ostream& o, const triplet& t ) {
    o<<t.var<<" "<<t.row<<" "<<t.val;
    return o; }
bool lessTriplet ( const triplet& t1, const triplet& t2 ){
    return t1.var < t2.var;  }

class InvTripletItr : public vector<triplet>::const_iterator {
public:
    InvTripletItr( vector<triplet>::const_iterator itr ) : vector<triplet>::const_iterator(itr) {};
    unsigned i() const {return (*this)->row;}
    double x() const {return (*this)->val;}
};

/*class InvData {
//template <class InvDataItr> class InvData {
public:
    pair<class InvTripletItr,class InvTripletItr> getRange(unsigned var) const;
    //pair<InvDataItr,InvDataItr> getRange(unsigned var) const;
    unsigned n() const;
    bool y( unsigned row ) const;
};*/
class InvData {
//template <> class InvData<InvTripletItr> {
    vector<triplet> data;
    vector<unsigned> varOffsets;
    unsigned nrows;
    vector<bool> m_y;
    unsigned m_dim;
    unsigned m_ngroups;
    vector< vector<bool> > m_groups;
public:
    typedef pair<InvTripletItr,InvTripletItr> TRange;
    typedef InvTripletItr TItr;
    unsigned n() const { return nrows; }
    unsigned dim() const { return m_dim; }
    unsigned ngroups() const { return m_ngroups; }
    bool y( unsigned row ) const { return m_y.at(row); }
    bool group( unsigned row, unsigned g ) const { return m_groups.at(row).at(g); }
    InvData( IRowSet& rowset ) 
        : m_dim(rowset.dim()), m_ngroups(rowset.ngroups())
    {
            //Log(12)<<"\nCreate Inverted data";
        rowset.rewind();
        for( nrows=0; rowset.next(); nrows++ ) {
            const SparseVector& x=rowset.xsparse();
            for( SparseVector::const_iterator xitr=x.begin(); xitr!=x.end(); xitr++ )
                data.push_back( triplet( xitr->first, nrows, xitr->second ) );
            m_y.push_back( rowset.y() );
            if( rowset.ngroups() ) {
                vector<bool> bg;
                for( unsigned g=0; g<rowset.ngroups(); g++ ){
                    bg.push_back( rowset.group(g) );
                }
                m_groups.push_back( bg );
            }
        }
        sort( data.begin(), data.end(), lessTriplet );

        //--- create index - var offsets ---
        varOffsets.resize( rowset.dim(), data.size() ); //init as non-existing
            //Log(12)<<"\nrowset.dim(), data.size() data[0].var "<<rowset.dim()<<" "<<data.size()<<" "<<data[0].var;
        unsigned vcurr = 0;
        varOffsets[vcurr] = 0;
        for( unsigned t=0; t<data.size(); t++ ) { //was BIG BUG: why??? t=1 *NB*
            unsigned vnext = data[t].var;
            if( vnext!=vcurr ) {
                for( unsigned v=vcurr+1; v<=vnext; v++ ) //may be a gap btw vnext and vcurr
                    varOffsets[v] = t;
                vcurr = vnext;
            }
        }
            //Log(10)<<"\nvarOffsets: ";
            //for( unsigned r=0; r<varOffsets.size(); r++ )
                //Log(10)<<varOffsets[r]<<" ";
    }
    pair<InvTripletItr,InvTripletItr> getRange(unsigned var) const {
        if( var>=varOffsets.size() )
            throw logic_error("Inverted data Variable out of range");
        vector<triplet>::const_iterator start = data.begin() + varOffsets[var];
        vector<triplet>::const_iterator fin =
            var==varOffsets.size()-1 ? data.end()
            : data.begin() + varOffsets[var+1];
        return pair<InvTripletItr,InvTripletItr>( start, fin );
    }
};


static bool trace=false;

/*template <class InvDataItr> double OneCoordStep(
    double dw_numerator,
    double dw_denominator,
    const pair<InvDataItr,InvDataItr>& range,
    const InvData<InvDataItr>& invData, //for 'y' only
    const Vector& wTX,
    double trustregion,
    bool highAccuracy
    )*/
double OneCoordStep(
    double dw_numerator,
    double dw_denominator,
    const pair<InvTripletItr,InvTripletItr>& range,
    const class InvData& invData, //for 'y' only
    const Vector& wTX,
    double trustregion,
    bool highAccuracy
    )
{ 
            //ver 2.04  accurate summation
            vector<double> dnums; vector<double> ddenoms;
            if(highAccuracy) {
                dnums.push_back(dw_numerator);
                ddenoms.push_back(dw_denominator);
            }

	        for( InvTripletItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
	        //for( InvDataItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
            {
                unsigned i = ix.i(); //->first;
                double x = ix.x(); //->second;
                if( 0.0==x ) continue; //just in case - shouldn't ever hit this
                int target = invData.y(i) ? 1 : -1;
                double r = 1==target ? wTX[i] : -wTX[i]; //wTX[i] * target;
		        //double q = 1 / (1 + exp(r));  //p = 1 / (1 + exp(- r))
                //try this approximation
                double a = fabs(r);
                double b = fabs(trustregion * x);
                double F; // = a<=b ? 0.25 : 1.0/( 2.0 + exp(a-b) + exp(b-a) );
                if( a<=b ) F = 0.25;
                else {
                    double e = exp(a-b);
                    F = 1.0/( 2.0 + e + 1.0/e );
                }
                /*/original Z&O
		        double F = min( 0.25, exp(fabs(trustregion * x)) * q * (1 - q)); //* p * (1 - p))
                    */
                if( highAccuracy ){
                    dnums.push_back( x*target/(1 + exp(r)) );  //ver 2.04  accurate summation
                    ddenoms.push_back( F*x*x );  //ver 2.04  accurate summation
                }else{
	                dw_numerator += x * target / (1 + exp(r)) ;  //q * x * target; //(1 - p) * x * target
	                dw_denominator += F * x * x;
                }
                //if(trace)                 cout<<"\n x y xy r expr ddnum "<<x<<" "<<target<<" "<<x*target<<" "<<r<<" "<<exp(r)<<" "<<x * target / (1 + exp(r));
                //if(trace)                 cout<<"\n i x y r dw_numerator F dw_denominator "<<i<<" "<<x<<" "<<target<<" "<<r
                  //                    <<"\t  "<<dw_numerator<<" "<<F<<" "<<dw_denominator;
           }
 
            //update w
            double dw;
            if( highAccuracy ){  //ver 2.04
                sort( dnums.begin(), dnums.end() );
                double dnumerator=0;
                for( vector<double>::const_iterator itr=dnums.begin(); itr!=dnums.end(); itr++ )
                    dnumerator += *itr;

                sort( ddenoms.begin(), ddenoms.end() );
                double ddenominator=0;
                for( vector<double>::const_iterator itr=ddenoms.begin(); itr!=ddenoms.end(); itr++ )
                    ddenominator += *itr;

                dw = dnumerator / ddenominator;
            }
            else
                dw = dw_numerator / dw_denominator;

            dw = min( max(dw,-trustregion), trustregion );
                //if(trace) cout<<"\nOUT dnumerator ddenominator trustrgn dw "<<dnumerator<<" "<<ddenominator<<" "<<trustregion<<" "<< dw ;
            return dw;
}

/* approximate likelihood-related terms of one-coord step calculation
(penalty terms ignored}
*/
pair<double,double>  //numer, denom
approxLkli(
    const pair<InvTripletItr,InvTripletItr> & range,
    const InvData& invData, //for 'y' only
    const Vector& wTX,
    double trustregion,
    bool highAccuracy,
    int group=-1 //disabled if <0
    )
{ 
            double dw_numerator = 0;
            double dw_denominator = 0;
            vector<double> dnums; vector<double> ddenoms;  // highAccuracy mode

	        for( InvTripletItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
            {
                unsigned i = ix.i(); //->first;
                if( group>=0 && !invData.group( i, group ) ) continue; //--->>---
                double x = ix.x(); //->second;
                if( 0.0==x ) continue; //just in case - shouldn't ever hit this
                int target = invData.y(i) ? 1 : -1;
                double r = 1==target ? wTX[i] : -wTX[i]; //wTX[i] * target;
                double a = fabs(r);
                double b = fabs(trustregion * x);
                double F; // = a<=b ? 0.25 : 1.0/( 2.0 + exp(a-b) + exp(b-a) );
                if( a<=b ) F = 0.25;
                else {
                    double e = exp(a-b);
                    F = 1.0/( 2.0 + e + 1.0/e );
                }
                /*/original Z&O
		        double F = min( 0.25, exp(fabs(trustregion * x)) * p * (1 - p))
                    */
                double dd_numer = x * target / (1 + exp(r)) ;  //q * x * target; //(1 - p) * x * target
                double dd_denom = F * x * x;
                if( highAccuracy ){  //ver 2.04  accurate summation
                    dnums.push_back( dd_numer );
                    ddenoms.push_back( dd_denom );
                }else{
	                dw_numerator += dd_numer;
	                dw_denominator += dd_denom;
                }
                //if(trace) cout<<"\n x y xy r expr ddnum "<<x<<" "<<target<<" "<<x*target<<" "<<r<<" "<<exp(r)<<" "<<x * target / (1 + exp(r));
                //if(trace) cout<<"\n i x y r dw_numerator F dw_denominator "<<i<<" "<<x<<" "<<target<<" "<<r
                  //  <<"\t  "<<dw_numerator<<" "<<F<<" "<<dw_denominator;
           }
 
            if( highAccuracy ){  //ver 2.04
                sort( dnums.begin(), dnums.end() );
                for( vector<double>::const_iterator itr=dnums.begin(); itr!=dnums.end(); itr++ )
                    dw_numerator += *itr;

                sort( ddenoms.begin(), ddenoms.end() );
                for( vector<double>::const_iterator itr=ddenoms.begin(); itr!=ddenoms.end(); itr++ )
                    dw_denominator += *itr;
            }

            return pair<double,double>( dw_numerator, dw_denominator );
}

class WakeUpPlan {
    vector<bool> plan;        
public:
    bool operator()(unsigned int step ) {
        return true; //HACK
        //if( step<plan.size() ) return plan.at(step);
        //else return 0==step%100;
    }
    WakeUpPlan(unsigned size=1000) : plan( size+1, false ) { //ctor
        for(unsigned i=0;i*i<size;i++) plan[i*i] = true;     }
};

class LessByRef {
    const vector<double>& w;
public:
    LessByRef( const vector<double>& w_ )  : w(w_) {}
    bool operator()( unsigned i, unsigned j ) {
        return w.at(i)<w.at(j);   }
};

double shrunkMedian(
            const vector<double>& wsorted,
            double priorMode,
            double penaltyRatio )
{
    /* ties resolution:
        - leftmost of 'w'
        - prior mode has priority
    */
    //cout<<endl<<"shrunkMedian():  wsorted: "<<wsorted<<" priorMode penaltyRatio "<<priorMode<<" "<<penaltyRatio;cout.flush();
    int G = wsorted.size();
    int iMode = // wsorted[iMode-1] < priorMode <= wsorted[iMode] 
        lower_bound( wsorted.begin(), wsorted.end(), priorMode ) - wsorted.begin();

    vector<double> right_deriv; // at(i)=penalty derivative to the right of wsorted[i]
    for( int i=0; i<iMode; i++ )
        right_deriv.push_back( i+1/*#less*/ - (G-i-1)/*#grtr*/ - penaltyRatio );
    for( int i=iMode; i<G; i++ )
        right_deriv.push_back( i+1/*#less*/ - (G-i-1)/*#grtr*/ + penaltyRatio );
    //cout<<endl<<"iMode "<<iMode<<" right_deriv: "<<right_deriv;

    double sm;
    // shrunkMedian m.b. one of 'w'
    int iSM = lower_bound( right_deriv.begin(), right_deriv.end(), 0 ) 
        - right_deriv.begin();
    //cout<<"\n iSM "<<iSM; cout.flush();

    //prior mode - another candidate for shrunkMedian
    double mode_right_deriv = iMode/*#less*/ - (G-iMode)/*#grtr*/ + penaltyRatio;
    //cout<<"\n mode_right_deriv "<<mode_right_deriv; cout.flush();

    if( iSM<G ) //i.e. wsorted.at( iSM ) is a candidate for shrunkMedian
    {
        //check other candidate: prior mode
        if( 
            //obsolete: prior mode is not one of 'w' (iMode>=G) || priorMode!=wsorted.at(iMode) && 
            mode_right_deriv >= 0 
            &&( iMode==0 || right_deriv.at(iMode-1)<=0 )//prior mode has priority
           )
            sm = priorMode;
        else
            sm = wsorted.at( iSM );
    }
    else // prior mode is the only candidate
        if( mode_right_deriv >= 0 )
            sm = priorMode;
        else throw logic_error("Shrunk median failure");

    //cout<<"\n sm "<<sm; cout.flush();
    return sm;
}

THierW
ZOLRHier(
                const InvData& invData, //input - generates design data
                const BayesParameter& bayesParam, //input
                const BayesParameter& bayesParam2, //input
                const Vector& priorMode,  //input
                const Vector& priorScale,  //input
                const Vector& priorSkew,  //input
                double thrConverge,
                unsigned iterLimit,
                bool highAccuracy,
                THierW w  //input - init value of model coeffs
                 )
{
    Log(3)<<std::endl<<"Starting hierarchical model, Time "<<Log.time();
    Log(3)<<std::endl<<"Level 1 "<<bayesParam;
    Log(3)<<std::endl<<"Level 2 "<<bayesParam2;
    Log(10)<<std::endl<<"priorMode "<<priorMode;
    Log(10)<<std::endl<<"priorScale "<<priorScale;
    Log(10)<<std::endl<<"priorSkew "<<priorSkew;

    const unsigned d = invData.dim(); // #vars in design
    const unsigned n = invData.n(); // #examples
    const unsigned G = invData.ngroups();
    if( d==priorMode.size() && d==priorScale.size() ) ;//ok
    else    throw DimensionConflict(__FILE__,__LINE__);

    //tmp
    if( bayesParam.PriorType()!=bayesParam2.PriorType() )
        throw runtime_error("Uncompatible prior types for levels");
    //convert prior scale into penalty parameter
    vector<double> penalty( d ), penalty2( d );
    for( unsigned j=0; j<d; j++ ) {
        if( normal==bayesParam.PriorType() )
            penalty[j] = 1.0 /( 2*bayesParam.var()*priorScale[j]*priorScale[j] ); //'2*' fixes bug, ver3.01
        else if( Laplace==bayesParam.PriorType() )
            penalty[j] = bayesParam.gamma() / priorScale[j];
        else  throw runtime_error("Unsupported prior type for Level 1");
        if( normal==bayesParam2.PriorType() )
            penalty2[j] = 1.0 /( 2*bayesParam2.var()*priorScale[j]*priorScale[j] ); //'2*' fixes bug, ver3.01
        else if( Laplace==bayesParam2.PriorType() )
            penalty2[j] = bayesParam2.gamma() / priorScale[j];
        else  throw runtime_error("Unsupported prior type for Level 2");
    }

    // initialize dot products
    Vector wTX( 0.0, n );
    for( unsigned j=0; j<d; j++ ) //--design vars--
    {
        const vector<double>& w_thisfeat = w.at(j);
        InvData::TRange range = invData.getRange(j);
	    for( InvData::TItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
        {
            unsigned irow = ix->i();
            for( unsigned g=0; g<G; g++ ) {
                if( invData.group( irow, g ) ){
                    wTX.at(irow) += w_thisfeat.at(g) * ix->x();
                }
            }
        }
    }
    Vector wTXprev = wTX;

    Vector trustregion( 1.0, d );
    unsigned ndropped;

    vector<unsigned> sleep( d, 0 );
    WakeUpPlan wakeUpPlan;

    bool converge = false;
    unsigned k;
    int n1coordsteps=0, ndoubltries=0, nslept=0;
    for( k=0; !converge; k++ ) //---iterations loop--
    {
        THierW wPrev = w;
        double dot_product_total_change = 0.0;
        ndropped = 0;
        for( unsigned j=0; j<d; j++ ) //--design vars--
        {
            pair<InvTripletItr,InvTripletItr> range = invData.getRange(j);
            const vector<double>& wj = w.at(j);

            double penaltyRatio = penalty[j]/penalty2[j];
            //Log(8)<<"\npenaltyRatio "<<penaltyRatio;

            for( unsigned g=0; g<G; g++ ) { //--groups--
                //Log(8)<<"\nj g "<<j<<" "<<g<<"  w_j: "<<wj;
                const double wcurr = wj.at(g);
	            double dw;

                pair<double,double> numer_denom 
                    = approxLkli( range, invData, wTX, trustregion[j], highAccuracy, g );
                Log(10)<<"\nnumer denom "<<numer_denom.first<<" "<<numer_denom.second;

                if( normal==bayesParam.PriorType() && normal==bayesParam2.PriorType() )
                {
                    double wjsum=0;
                    for(unsigned gg=0; gg<wj.size(); gg++ )   wjsum += wj.at(gg);
                    double penaltyDeriv = 2*penalty2[j]*
                        ( wcurr - (priorMode[j]*penaltyRatio+wjsum)/(penaltyRatio+G) );
                    double penaltyDeriv2 = 2*penalty2[j]*
                        (penaltyRatio+G-1)/(penaltyRatio+G);
                    Log(10)<<"\npenaltyDeriv "<<penaltyDeriv<<"  penaltyDeriv2 "<<penaltyDeriv;
                    dw = (numer_denom.first - penaltyDeriv)/(numer_denom.second + penaltyDeriv2);
                    Log(10)<<"\n dw "<<dw;
                }
                else if( Laplace==bayesParam.PriorType() && Laplace==bayesParam2.PriorType() ) {
                    vector<double> wsorted = wj;
                    sort( wsorted.begin(), wsorted.end() );
                    double sm = shrunkMedian( wsorted, priorMode[j], penaltyRatio );
                    Log(10)<<"\nshrunkMedian "<<sm;

                    //TODO: skew

                    /*the value of shrunk median does not change while w_jg stays 
                      on one side of shrunk median
                    */
                    if( wcurr > sm ) {
                        dw = (numer_denom.first - penalty2[j]) / numer_denom.second;
                        Log(10)<<"\nwcurr dw "<<wcurr<<" "<<dw;
                    }
                    else if( wcurr < sm ) {
                        dw = (numer_denom.first + penalty2[j]) / numer_denom.second;
                        Log(10)<<"\nwcurr dw "<<wcurr<<" "<<dw;
                    }
                    else // at shrunk median right now
                    {
                        int nBelowSM=0, nAboveSM=0, nEqualSM=0;
                        double  smallestAbove=numeric_limits<double>::max(),
                                greatestBelow=-numeric_limits<double>::max();
                        for( unsigned gg=0; gg<G; gg++ )
                            if( wj.at(gg) > sm ) {
                                nAboveSM++;
                                if( wj.at(gg)<smallestAbove ) smallestAbove = wj.at(gg);
                            }
                            else if( wj.at(gg) < sm ) {
                                nBelowSM++;
                                if( wj.at(gg)>greatestBelow ) greatestBelow = wj.at(gg);
                            }
                            else
                                nEqualSM++;
                        //cout<<"\nj wj sm g "<<j<<" "<<wj<<" "<<sm<<" "<<g;
                        //cout<<"\nwcurr==sm;  nBelowSM nAboveSM "<<nBelowSM<<" "<<nAboveSM
                          //  <<"  greatestBelow smallestAbove "<<greatestBelow<<" "<<smallestAbove;//<<" "<<currGroupRank;

                        double dwPlus=0, dwMinus=0;
                        double derivPenaltyPlus, derivPenaltyMinus;
                        {//try positive
                            vector<double> w_tmp = wj;
                            w_tmp.at(g) = nAboveSM>0 ? smallestAbove : w_tmp.at(g)+1; //what happens with sm when w_jg changes?
                            sort( w_tmp.begin(), w_tmp.end() );
                            double smShiftPos = shrunkMedian( w_tmp, priorMode[j], penaltyRatio );
                            if( smShiftPos==sm ) //sm stays
                                derivPenaltyPlus = penalty2[j];
                            else { //sm moves with w_jg
                                int medianBalance = nBelowSM + nEqualSM - 1 - nAboveSM;
                                derivPenaltyPlus = ( sm>=priorMode[j] ? penalty[j] : -penalty[j] )
                                             + ( medianBalance==0 ? 0.0 //avoid 0*inf
                                                                : penalty2[j]*medianBalance );
                            }
                            dwPlus = ( numer_denom.first - derivPenaltyPlus )/ numer_denom.second;
                        }
                        if( dwPlus>0 )
                            dw = dwPlus;
                        //RESTORE else
                        {//try negative
                            vector<double> w_tmp = wj;
                            w_tmp.at(g) = nBelowSM>0 ? greatestBelow : w_tmp.at(g)-1; //what happens with sm when w_jg changes?
                            sort( w_tmp.begin(), w_tmp.end() );
                            double smShiftNeg = shrunkMedian( w_tmp, priorMode[j], penaltyRatio );
                            if( smShiftNeg==sm ) //sm stays
                                derivPenaltyMinus = -penalty2[j];
                            else { //sm moves with w_jg
                                int medianBalance = nBelowSM -( nAboveSM + nEqualSM - 1);
                                derivPenaltyMinus = ( sm<=priorMode[j] ? -penalty[j] : penalty[j] )
                                             + ( medianBalance==0 ? 0.0 //avoid 0*inf
                                                                : penalty2[j]*medianBalance );
                            }
                            dwMinus = ( numer_denom.first - derivPenaltyMinus )/ numer_denom.second;
                        }
                        if( dwMinus<0 )
                            dw = dwMinus;
                        else
                            dw = 0;

                        //cout<<"\nDerivPenalty plus minus     "<<derivPenaltyPlus<<" "<<derivPenaltyMinus;

                        /*dbg stupid derivPenalty
                        double f_at_w_jg = 0;
                        for( unsigned gg=0; gg<G;gg++ ) f_at_w_jg += fabs( sm-wj.at(gg) );
                        f_at_w_jg *= penalty2[j];
                        f_at_w_jg += penalty[j]*fabs( sm-priorMode[j] );

                        double derivPlus, derivMinus;
                        {
                        double wplus = nAboveSM>0 ? smallestAbove : wj.at(g)+1;
                        double f_at_wplus = 0;//assume sm stays
                        for( unsigned gg=0; gg<G;gg++ ) {
                            if(gg==g) f_at_wplus += fabs( sm-wplus );
                            else f_at_wplus += fabs( sm-wj.at(gg) );
                            }
                        f_at_wplus *= penalty2[j];
                        f_at_wplus += penalty[j]*fabs( sm-priorMode[j] );
                        double f_at_wplus2 = 0;//assume sm moves
                        for( unsigned gg=0; gg<G;gg++ ) 
                            if(gg==g) ; // sm<--smallestAbove
                            else f_at_wplus2 += fabs( wplus-wj.at(gg) );
                        f_at_wplus2 *= penalty2[j];
                        f_at_wplus2 += penalty[j]*fabs( wplus-priorMode[j] );
                        if( f_at_wplus2<f_at_wplus )  f_at_wplus=f_at_wplus2;
                        derivPlus = f_at_wplus==f_at_w_jg ? 0.0 : (f_at_wplus-f_at_w_jg)/(wplus-sm);
                        //cout<<"\n derivPlus = f_at_wplus==f_at_w_jg ? 0.0 : (f_at_wplus-f_at_w_jg)/(smallestAbove-sm) "
                          //  <<f_at_wplus<<" "<<f_at_w_jg<<" "<<smallestAbove<<" "<<sm;

                        double wminus = nBelowSM>0 ? greatestBelow : wj.at(g)-1;
                        double f_at_wminus = 0;//assume sm stays
                        for( unsigned gg=0; gg<G;gg++ ) 
                            if(gg==g) f_at_wminus += fabs( sm-wminus );
                            else f_at_wminus += fabs( sm-wj.at(gg) );
                        f_at_wminus *= penalty2[j];
                        f_at_wminus += penalty[j]*fabs( sm-priorMode[j] );
                        double f_at_wminus2 = 0;//assume sm moves
                        for( unsigned gg=0; gg<G;gg++ ) 
                            if(gg==g) ; // sm<--greatestBelow
                            else f_at_wminus2 += fabs( wminus-wj.at(gg) );
                        f_at_wminus2 *= penalty2[j];
                        f_at_wminus2 += penalty[j]*fabs( wminus-priorMode[j] );
                        if( f_at_wminus2<f_at_wminus )  f_at_wminus=f_at_wminus2;
                        derivMinus = f_at_wminus==f_at_w_jg ? 0.0 : (f_at_wminus-f_at_w_jg)/(wminus-sm);

                        cout<<"\nTest derivPenalty plus minus "<<derivPlus<<" "<<derivMinus;
                        if(derivPlus!=derivPenaltyPlus) cout<<"\nError derivPlus!";
                        if(float(derivMinus)!=float(derivPenaltyMinus)) cout<<"\nError derivMinus!";
                        }
                        <==test*/

                        /* obsolete
                        double derivPenalty2 = (nBelowSM-nAboveSM) ? penalty2[j]*(nBelowSM-nAboveSM) : 0.0; //avoid 0*inf
                        if( sm >= priorMode[j] )
                            dwPlus = numer_denom.first - penalty[j] - derivPenalty2;
                        if( sm <= priorMode[j] )
                            dwMinus = numer_denom.first + penalty[j] - derivPenalty2;
                        Log(10)<<"\n derivPenalty2 dwPlus dwMinus "
                            <<derivPenalty2<<" "<<dwPlus<<" "<< dwMinus;//<<" "<<currGroupRank;
                        if( dwPlus>0 )          dw = dwPlus / numer_denom.second;
                        else if( dwMinus<0 )    dw = dwMinus / numer_denom.second; 
                        else dw = 0;
                        Log(10)<<"\nwcurr dw "<<wcurr<<" "<<dw;*/

                        //cut around shrunk median
                        if( dw > smallestAbove-wcurr ) dw = smallestAbove-wcurr;
                        if( dw < greatestBelow-wcurr ) dw = greatestBelow-wcurr;
                        Log(10)<<"  dw-cut-sm "<<dw;
                    }

                    //cut at mode and shrunk median
                    if( wcurr<priorMode[j] && priorMode[j]<wcurr+dw
                     || wcurr>priorMode[j] && priorMode[j]>wcurr+dw )  dw = priorMode[j] - wcurr;
                    if( wcurr<sm && sm<wcurr+dw 
                     || wcurr>sm && sm>wcurr+dw )                      dw = sm - wcurr;
                    Log(10)<<" dw-cut "<<dw;

                } //Laplace
                else
                    throw logic_error("Priors of different type on two levels not supported");

                dw = min( max(dw,-trustregion[j]), trustregion[j] );
                Log(10)<<" trustregion[j] dw-trustrgn "<<trustregion[j]<<" "<<dw;

                w.at(j).at(g) += dw;

                //update local data
	            for( InvTripletItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
                {
                    unsigned i = ix->i();
                    double x = ix->x();
                    if( ! invData.group( i, g ) ) continue;
                    if( 0.0==x ) continue; //just in case - shouldn't ever hit this
                    wTX[i] += dw * x;
		            dot_product_total_change += fabs(dw * x); 
                }

                //trust region update
                //Log(6)<<"\n j g trustregion dw trustregion "<<j<<"  "<<g<<"  "<<trustregion[j]<<"  "<<dw<<"  "<<max( 2*fabs(dw), trustregion[j]/2 );
		        trustregion[j] = max( 2*fabs(dw), trustregion[j]/2 );

            }//--groups-- g --
            //Log(8)<<"\nend-of-j "<<j<<"  w_j: "<<wj;
        }//--design vars-- j --

        double sum_abs_r = 0;
        for( unsigned i=0; i<n; i++ )
            sum_abs_r += fabs(wTX[i]);
        //ZO stopping
        double sum_abs_dr = 0.0;
        for( unsigned i=0; i<n; i++ )
            sum_abs_dr += fabs(wTX[i]-wTXprev[i]);

        double dot_product_rel_change = dot_product_total_change/(1+sum_abs_r);//DL stopping
        double rel_sum_abs_dr = sum_abs_dr/(1+sum_abs_r);//ZO stopping

        converge =( k>=iterLimit  ||
            rel_sum_abs_dr<thrConverge );//ZO stopping
            //DL stopping || dot_product_rel_change<thrConverge );

        Log(7)<<"\nZO iteration "<<k+1
            <<"  Dot product abs sum "<<sum_abs_r
            <<"  Rel change "<<rel_sum_abs_dr //ZO stopping //DL stopping dot_product_rel_change
            //<<"  Beta relative increment = "<<norm(Vector(w-wPrev))/norm(wPrev)<<"  Beta components dropped: "<<ndropped
            ;

        wTXprev = wTX;
    }//---iter loop---

    //Log(7)<<std::endl<<"1-coord steps: "<<n1coordsteps<<"  Double tries: "<<ndoubltries<<" Slept: "<<nslept;
    Log(5)<<std::endl<<( k>=iterLimit ? "Stopped by iterations limit"  : "Stopped by original ZO rule" );
    Log(3)<<std::endl<<"Built hierarchical model "<<k<<" iterations, Time "<<Log.time(); 
    Log(0).flush();
    return w;
}


//template <class InvDataItr> 
Vector /*returns model coeffs*/
ZOLR(
                const InvData& invData, //input - generates design data
                //const InvData<InvDataItr>& invData, //input - generates design data
                const BayesParameter& bayesParam, //input
                const Vector& priorMode,  //input
                const Vector& priorScale,  //input
                const Vector& priorSkew,  //input
                double thrConverge,
                unsigned iterLimit,
                bool highAccuracy,
                Vector w  //input - init value of model coeffs
                 )
{
    Log(3)<<std::endl<<"Starting ZO model, Time "<<Log.time();
    Log(3)<<std::endl<<bayesParam;
    Log(10)<<std::endl<<priorMode;
    Log(10)<<std::endl<<priorScale;
    Log(10)<<std::endl<<priorSkew;

    unsigned d = invData.dim(); // #vars in design
    unsigned n = invData.n(); // #examples
    if( d==priorMode.size() && d==priorScale.size() ) ;//ok
    else    throw DimensionConflict(__FILE__,__LINE__);

    //convert prior scale into penalty parameter
    vector<double> penalty( d );
    for( unsigned j=0; j<d; j++ )
        if( normal==bayesParam.PriorType() )
            penalty[j] = 1.0 /( 2*bayesParam.var()*priorScale[j]*priorScale[j] );  //'2*' fixes bug, ver3.01
        else if( Laplace==bayesParam.PriorType() )
            penalty[j] = bayesParam.gamma() / priorScale[j];
        else
            throw runtime_error("ZO only allows normal or Laplace prior");

    // initialize the search
    Vector wTX( 0.0, n );
    for( unsigned j=0; j<d; j++ ) //--design vars--
    if( w[j] != 0.0 )
    {
        //pair<InvDataItr,InvDataItr> range = invData.getRange(j);
	    //for( InvDataItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
        pair<InvTripletItr,InvTripletItr> range = invData.getRange(j);
	    for( InvTripletItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
        {
            wTX[ ix->i() ] += w[j] * ix->x();
        }
    }
    Vector wTXprev = wTX;

    Vector trustregion( 1.0, d );
    unsigned ndropped;

    vector<unsigned> sleep( d, 0 );
    class WakeUpPlan {
        vector<bool> plan;        
    public:
        bool operator()(unsigned int step ) {
            return true; //HACK
            //if( step<plan.size() ) return plan.at(step);
            //else return 0==step%100;
        }
        WakeUpPlan(unsigned size=1000) : plan( size+1, false ) { //ctor
            for(unsigned i=0;i*i<size;i++) plan[i*i] = true;     }
    };
    WakeUpPlan wakeUpPlan;

    bool converge = false;
    unsigned k;
    int n1coordsteps=0, ndoubltries=0, nslept=0;
    for( k=0; !converge; k++ ) //---iterations loop--
    {
        Vector wPrev = w;
        double dot_product_total_change = 0.0;
        ndropped = 0;
        for( unsigned j=0; j<d; j++ ) //--design vars--
        {
	        double dw, dw_numerator, dw_denominator;

            pair<InvTripletItr,InvTripletItr> range = invData.getRange(j);
            //pair<InvDataItr,InvDataItr> range = invData.getRange(j);
            //Log(10)<<"\nVar "<<j<<" range: "<<range.first.i()<<"/"<<range.first.x()<<" - "<<range.second.i()<<"/"<<range.second.x();

            if( normal==bayesParam.PriorType() )
            {
                dw_numerator = (w[j]==priorMode[j]) ? 0.0  //in case of infinite penalty // ver 2.57
                    : - 2 * ( w[j] - priorMode[j] ) * penalty[j];
                dw_denominator = 2 * penalty[j];
                dw = OneCoordStep( dw_numerator, dw_denominator, range, invData, wTX, trustregion[j], highAccuracy );
                /*switch to approxLkli: small numeric differences
                pair<double,double> numer_denom = approxLkli( range, invData, wTX, trustregion[j], highAccuracy );
                dw = (numer_denom.first + dw_numerator)/(numer_denom.second + dw_denominator);
                */
                //cout<<"\n j w[j] priorMode[j] penalty[j] dw_numerator "<<j<<" "<<w[j]<<" "<<priorMode[j]<<" "<<penalty[j]<<" "<<dw_numerator;*/
           }
            else { //Laplace
                if( w[j]!=priorMode[j] || wakeUpPlan( sleep[j] ) ){
                    pair<double,double> numer_denom = approxLkli( range, invData, wTX, trustregion[j], highAccuracy );
                    n1coordsteps++;
                    if( w[j]-priorMode[j]>0 ) {
                        dw = (numer_denom.first - penalty[j]) / numer_denom.second;
                    }
                    else if( w[j]-priorMode[j]<0 ) {
                        dw = (numer_denom.first + penalty[j]) / numer_denom.second;
                        //dw = min( max(dw,-trustregion), trustregion );
                    }
                    else // at mean right now
                    { // try both directions
                        //if(k>0) Log(9)<<"\nTry Wake-up j/sleep[j] "<<j<<"/"<<sleep[j];
                        dw = 0;
                        if( priorSkew[j] > -1 ) { //positive allowed
                            double dwPlus = (numer_denom.first - penalty[j]) / numer_denom.second;
                            //dwPlus = min( max(dwPlus,-trustregion), trustregion );
                            if( dwPlus>0 )
                                dw = dwPlus;
                        }
                        if( dw==0 ) //positive step not allowed or unsuccessful
                        {
                            if( priorSkew[j] < 1 ) { //negative allowed
                                double dwMinus = (numer_denom.first + penalty[j]) / numer_denom.second;
                                //dwMinus = min( max(dwMinus,-trustregion), trustregion );
                                if( dwMinus<0 )
                                    dw = dwMinus; //othw dw stays at 0
                            }
                        }
                    }
                }
                else
                    nslept++;//Log(9)<<"\nNo Wake-up j/sleep[j] "<<j<<"/"<<sleep[j]

                dw = min( max(dw,-trustregion[j]), trustregion[j] ); //for normal prior this is done in OneCoordStep

                //check if we crossed the break point
                if(( w[j] < priorMode[j] && priorMode[j] < w[j]+dw )
                    || ( w[j]+dw < priorMode[j] && priorMode[j] < w[j] )
                    )
                    dw = priorMode[j] - w[j]; //stay at mean
            } //Laplace

            w[j] += dw;
            if( w[j]==priorMode[j] )   ndropped++;
            //update local data
	        for( InvTripletItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
	        //for( InvDataItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
            {
                unsigned i = ix->i();
                double x = ix->x();
                if( 0.0==x ) continue; //just in case - shouldn't ever hit this
		        //double old = wTX[i];
                wTX[i] += dw * x;
		        dot_product_total_change += fabs(dw * x); 
            }
            //trust region update
		    trustregion[j] = max( 2*fabs(dw), trustregion[j]/2 );

            if( 0==w[j]-priorMode[j] ) sleep[j]++;
            else {
                //if( sleep[j]>0 && k>0 ) Log(9)<<"\nWake-up j/sleep[j] "<<j<<"/"<<sleep[j];
                sleep[j]=0;
            }

        }//--design vars-- j --

        double sum_abs_r = 0;
        for( unsigned i=0; i<n; i++ )
            sum_abs_r += fabs(wTX[i]);
        //ZO stopping
        double sum_abs_dr = 0.0;
        for( unsigned i=0; i<n; i++ )
            sum_abs_dr += fabs(wTX[i]-wTXprev[i]);

        double dot_product_rel_change = dot_product_total_change/(1+sum_abs_r);//DL stopping
        double rel_sum_abs_dr = sum_abs_dr/(1+sum_abs_r);//ZO stopping

        converge =( k>=iterLimit  ||
            rel_sum_abs_dr<thrConverge );//ZO stopping
            //DL stopping || dot_product_rel_change<thrConverge );

        Log(7)<<"\nZO iteration "<<k+1
            <<"  Dot product abs sum "<<sum_abs_r
            <<"  Rel change "<<rel_sum_abs_dr //ZO stopping //DL stopping dot_product_rel_change
            <<"  Beta relative increment = "<<norm(Vector(w-wPrev))/norm(wPrev)
            <<"  Beta components dropped: "<<ndropped;
            //<<"  Time "<<Log.time();
        /*dbg
        Log(4)<<"\nBeta sparse:";
        for( unsigned i=0; i<w.size(); i++ )  if(0!=w[i]) Log(4)<<" "<<i<<":"<<w[i];  */

        wTXprev = wTX;
    }//---iter loop---

    Log(7)<<std::endl<<"1-coord steps: "<<n1coordsteps<<"  Double tries: "<<ndoubltries<<" Slept: "<<nslept;
    Log(5)<<std::endl<<( k>=iterLimit ? "Stopped by iterations limit"  : "Stopped by original ZO rule" );
    Log(3)<<std::endl<<"Built ZO model "<<k<<" iterations, Time "<<Log.time(); 
    Log(0).flush();
    return w;
}

class CVLoop { //run for a given fold: train and validation sets given
    vector<double> loglikeli;
public:
    const vector<double>& Loglikeli() const { return loglikeli; }
    CVLoop( 
        IRowSet& drstrain,
        InvData& invData,
        IRowSet& drsvalid,
        ModelType modelType,
        const HyperParamPlan& hpPlan,
        const Vector& priorMode,
        const Vector& priorScale,
        const Vector& priorSkew
        )
    {        
        unsigned planSize = hpPlan.plan().size();
        Vector localbeta = priorMode; //( 0.0, drs.dim() );
        for( unsigned iparam=0; iparam<planSize; iparam++ )//hyper-parameter loop
        {
            Log(5)<<"\nHyperparameter plan #"<<iparam+1<<" value="<<(hpPlan.plan()[iparam]);
            //BayesParameter localBayesParam( bayesParameter.PriorType(), paramPlan[iparam], bayesParameter.skew());
            BayesParameter localBayesParam = hpPlan.InstBayesParam( iparam );

            // build the model
            localbeta = ZOLR( invData, //<InvTripletItr>
                localBayesParam, priorMode, priorScale, priorSkew, 
                modelType.ThrConverge(), modelType.IterLimit(), modelType.HighAccuracy(),
                localbeta );
            //Log(9)<<"\nlocalBeta "<<localbeta;

            double eval = LogLikelihood( drsvalid, modelType, localbeta );

            loglikeli.push_back( eval );
        }//hyper-parameter loop
    }
};//class CVLoop

static double AvgSquNorm( const Stats& stats ) { //separated from 'class Stats' to exclude constant term
    double s =0.0;
    for( unsigned j=0; j<stats.Means().size(); j++ ) {
        if( j==stats.Means().size()-1 ) //HACK: drop constant term
            break;
        s += stats.Means()[j]*stats.Means()[j] + stats.Stddevs()[j]*stats.Stddevs()[j];
    }
    return s;
}

class CVLoopHier { //run all hyperpar values for given train and validation sets
    vector<vector<double> > loglikeli; //level2 is inner
public:
    const vector<vector<double> >& Loglikeli() const { return loglikeli; }
    CVLoopHier( 
        IRowSet& drstrain,
        InvData& invData,
        IRowSet& drsvalid,
        ModelType modelType,
        const HyperParamPlan& hpPlan,
        const HyperParamPlan& hpPlan2,
        const Vector& priorMode,
        const Vector& priorScale,
        const Vector& priorSkew
        )
    {        
        unsigned planSize = hpPlan.plan().size();
        /*2.55 THierW wlocal;
        for( unsigned j=0; j<drstrain.dim(); j++ )
            wlocal.push_back( vector<double>( drstrain.ngroups(), priorMode.at(j) ) );*/

        for( unsigned iparam=0; iparam<hpPlan.plan().size(); iparam++ )//hyper-parameter loop
        {
            vector<double> loglikeliPlan2;

            //v2.55 init with non-hier model
            BayesParameter localBayesParam = hpPlan.InstBayesParam( iparam );
            Vector beta = ZOLR( invData,
                    localBayesParam, priorMode, priorScale, priorSkew,
                    modelType.ThrConverge(), modelType.IterLimit(), modelType.HighAccuracy(),
                    priorMode );
            THierW wlocal;
            for( unsigned j=0; j<drstrain.dim(); j++ )
                wlocal.push_back( vector<double>( drstrain.ngroups(), beta.at(j) ) );

            for( unsigned iparam2=0; iparam2<hpPlan2.plan().size(); iparam2++ )//hyper-parameter2 loop
            {
                Log(5)<<"\nHyperparameter plan Level 1: "<<(hpPlan.plan()[iparam])<<" Level 2: "<<(hpPlan2.plan()[iparam2]);
                BayesParameter localBayesParam2 = hpPlan2.InstBayesParam( iparam2 );

                wlocal = ZOLRHier( invData,
                    localBayesParam, localBayesParam2, priorMode, priorScale, priorSkew, 
                    modelType.ThrConverge(), modelType.IterLimit(), modelType.HighAccuracy(),
                    wlocal );

                double eval = LogLikelihoodHier( drsvalid, modelType, wlocal );
                loglikeliPlan2.push_back( eval );

            }//hyper-parameter2 loop

            loglikeli.push_back( loglikeliPlan2 );

        }//hyper-parameter loop
    }
};//class CVLoopHier

void ZOLRModel::tuneHierModel(   IRowSet & drs,
                    const HyperParamPlan& hparPlan_arg,
                    const HyperParamPlan& hparPlan2,
                    const Vector& priorMode, const Vector& priorScale, const Vector& priorSkew,
                    Stats& stats )
{
    HyperParamPlan hparPlan;
    if( hparPlan_arg.AutoHPar() ) { //norm-based
        double avgsqunorm = m_modelType.Standardize()*drs.dim() ? 1.0 : AvgSquNorm(stats);
        double invPriorVar = avgsqunorm/drs.dim();
        double hpar = 
            normal==hparPlan_arg.PriorType() ? 1.0/invPriorVar
            : /*Laplace*/ 0==hparPlan_arg.skew() ? sqrt(invPriorVar*2.0)
            : sqrt(invPriorVar); //assumes skew can only be 1 or -1
        Log(3)<<"\nAvg square norm (no const term) "<<avgsqunorm<<" Prior var "<<1.0/invPriorVar<<" Hyperparameter "<<hpar;
        hparPlan = HyperParamPlan( hparPlan_arg.PriorType(), hpar, hparPlan_arg.skew() );
    }
    else
        hparPlan = hparPlan_arg;

    if( hparPlan2.AutoHPar() ) //norm-based
        throw runtime_error("Norm-based default not supported for Level 2 priors");

    if( hparPlan.hasGrid() || hparPlan2.hasGrid() )     // cv needed
    {
        const double randmax = RAND_MAX;

        //prepare CV folds and runs
        const HyperParamPlan & cvPlan = hparPlan.hasGrid() ? hparPlan : hparPlan2; //which has grid?
                // 2.51     Jun 27, 05  bug: cv with default on 1st level - fixed
        unsigned nfolds = cvPlan.nfolds(); //10
        unsigned nruns = cvPlan.nruns(); //2
        if( nfolds>drs.n() ) {
            nfolds = drs.n();
            Log(1)<<"\nWARNING: more folds requested than there are data. Reduced to "<<nfolds;
        }
        if( nruns>nfolds )  nruns = nfolds;

        vector<int> rndind = PlanStratifiedSplit( drs, nfolds );

        vector<bool> foldsAlreadyRun( nfolds, false );

        //cv loop
        vector< vector<double> > lglkliMean( hparPlan.plansize(), vector<double>(hparPlan2.plansize(),0.0) );
        vector< vector<double> > lglkliMeanSqu( hparPlan.plansize(), vector<double>(hparPlan2.plansize(),0.0) );
        for( unsigned irun=0; irun<nruns; irun++ )
        {
            //next fold to run
            unsigned x = unsigned( rand()*(nfolds-irun)/(randmax+1) );
            unsigned ifold, y=0;
            for( ifold=0; ifold<nfolds; ifold++ )
                if( !foldsAlreadyRun[ifold] ) {
                    if( y==x ) break;
                    else y++;
                }
            foldsAlreadyRun[ifold] = true;
            Log(5)<<"\nCross-validation "<<nfolds<<"-fold; Run "<<irun+1<<" out of "<<nruns<<", fold "<<ifold+1;

            //run cv part for the given fold
            SamplingRowSet cvtrain( drs, rndind, ifold, ifold+1, false ); //int(randmax*ifold/nfolds), int(randmax*(ifold+1)/nfolds)
            SamplingRowSet cvtest( drs, rndind, ifold, ifold+1, true ); //int(randmax*ifold/nfolds), int(randmax*(ifold+1)/nfolds)
            Log(5)<<"\ncv training sample "<<cvtrain.n()<<" rows"<<"  test sample "<<cvtest.n()<<" rows";
            InvData invData( cvtrain ); //<InvTripletItr>
            CVLoopHier hyperParLoop( cvtrain, invData, cvtest,
                m_modelType, hparPlan, hparPlan2,
                priorMode, priorScale, priorSkew );

            // process cv fold results
            Log(5)<<"\nCross-validation test log-likelihood: ";
            for( unsigned ipar=0;  ipar<hparPlan .plansize(); ipar++ ) {
                for( unsigned ipar2=0; ipar2<hparPlan2.plansize(); ipar2++ )
                {
                    double eval = hyperParLoop.Loglikeli().at(ipar).at(ipar2);
                    Log(5)<<eval<<" ";
                    //arith ave of loglikeli, i.e. geometric mean of likelihoods
                    lglkliMean.at(ipar).at(ipar2) = lglkliMean.at(ipar).at(ipar2)*irun/(irun+1) + eval/(irun+1);
                    lglkliMeanSqu.at(ipar).at(ipar2) = lglkliMeanSqu.at(ipar).at(ipar2)*irun/(irun+1) + eval*eval/(irun+1);
                }
            }
        }//cv loop

        vector< vector<double> > lglkliStdErr( hparPlan.plansize(), vector<double>(hparPlan2.plansize(),0.0) );
        for( unsigned ipar=0;  ipar<hparPlan .plansize(); ipar++ )
            for( unsigned ipar2=0; ipar2<hparPlan2.plansize(); ipar2++ )
                lglkliStdErr.at(ipar).at(ipar2) = 
                    sqrt( lglkliMeanSqu.at(ipar).at(ipar2) 
                        - lglkliMean.at(ipar).at(ipar2)*lglkliMean.at(ipar).at(ipar2) )//stddev
                    / sqrt(double(nruns));

        // best by cv
        double bestEval = - numeric_limits<double>::max();
        unsigned bestParam=unsigned(-1), bestParam2=unsigned(-1);
        Log(5)<<"\nCross-validation results: \n prior variance Level 1, Level 2, cv mean loglikelihood, std.error";
        for( unsigned ipar=0;  ipar<hparPlan .plansize(); ipar++ )
            for( unsigned ipar2=0; ipar2<hparPlan2.plansize(); ipar2++ )
            {
                if( lglkliMean.at(ipar).at(ipar2) > bestEval ) {
                    bestEval = lglkliMean.at(ipar).at(ipar2);
                    bestParam = ipar; bestParam2 = ipar2;
                }
                Log(5)<<"\n\t"<<hparPlan.planAsVar(ipar)<<"\t"<<hparPlan2.planAsVar(ipar2)
                    <<"\t"<<lglkliMean.at(ipar).at(ipar2)<<"\t"<<lglkliStdErr.at(ipar).at(ipar2);
            }
        if( bestParam==unsigned(-1) )
            throw runtime_error("No good hyperparameter value found");

        //error bar rule
        if( hparPlan.ErrBarRule() ) {
            throw runtime_error("One std.error rule not supported with hierarchies");
            /*double valueneeded = bestEval - lglkliStdErr.at(bestParam).at(bestParam2);
            unsigned oldBestParam = bestParam;
            for( unsigned i=0; i<planSize; i++ )
                if( lglkliMean[i]>=valueneeded ) {
                    bestParam = i;
                    break;
                }
            Log(6)<<"\nError bar rule selected prior value "<<hpplan[bestParam]<<" instead of "<<hpplan[oldBestParam];
            */
        }
        Log(3)<<"\nBest prior var Level 1: "<<hparPlan.planAsVar(bestParam)
            <<"  Level 2: "<<hparPlan2.planAsVar(bestParam2)
            <<" cv-average loglikely "<<bestEval;

        m_bayesParam = hparPlan.InstBayesParam( bestParam );
        m_bayesParam2 = hparPlan2.InstBayesParam( bestParam2 );
        Log(3)<<std::endl<<"Starting final model after cv, Time "<<Log.time();
    }
    else {
        m_bayesParam = hparPlan.bp();
        m_bayesParam2 = hparPlan2.bp();
    }

    //build final model
    drs.rewind();
    InvData invData( drs );

    //v2.55 init with non-hier model
    Vector beta = ZOLR( invData,
            m_bayesParam, priorMode, priorScale, priorSkew,
            m_modelType.ThrConverge(), m_modelType.IterLimit(), m_modelType.HighAccuracy(),
            priorMode );
    THierW wInit;
    for( unsigned j=0; j<drs.dim(); j++ )
        wInit.push_back( vector<double>( drs.ngroups(), beta.at(j) ) );

    m_hierW = ZOLRHier( invData,
            m_bayesParam, m_bayesParam2, priorMode, priorScale, priorSkew,
            m_modelType.ThrConverge(), m_modelType.IterLimit(), m_modelType.HighAccuracy(),
            wInit );
}

static double NormBasedDefaultVar( 
        unsigned dim, Stats& stats, bool standardize )
{
    double avgsqunorm = standardize ? 1.0 : AvgSquNorm(stats);
    double priorVar = dim/avgsqunorm;
    Log(3)<<"\nAvg square norm (no const term) "<<avgsqunorm<<" Prior var "<<priorVar;
    return priorVar;
}

#include "HParSearch.h"

void ZOLRModel::tuneModel(   IRowSet & drs,
                    const HyperParamPlan& hyperParamPlan,
                    const Vector& priorMode, const Vector& priorScale, const Vector& priorSkew,
                    Stats& stats )
{
    Vector beta = priorMode; //( 0.0, drs.dim() );
    if( hyperParamPlan.fixed() || hyperParamPlan.normDefault() ) {  // no search; fixed given or default
        if( hyperParamPlan.normDefault() ) {
            double priorVar = NormBasedDefaultVar( drs.dim(), stats, m_modelType.Standardize() );
            double hpar = var2hyperpar( hyperParamPlan.PriorType(), hyperParamPlan.skew(), priorVar );
                /*normal==hyperParamPlan.PriorType() ? 1.0/invPriorVar
                : *Laplace* 0==hyperParamPlan.skew() ? sqrt(invPriorVar*2.0)
                : sqrt(invPriorVar); //assumes skew can only be 1 or -1*/
            Log(3)<<" Hyperparameter "<<hpar;
            m_bayesParam = BayesParameter( hyperParamPlan.PriorType(), hpar, hyperParamPlan.skew() );
        }
        else if( hyperParamPlan.fixed() )
            m_bayesParam = hyperParamPlan.bp();
        else throw logic_error("Inconsistent 'HyperParamPlan' setting");
        
        /*drs.rewind();
        InvData invData( drs );
        m_beta = ZOLR( invData, //<InvTripletItr>
            m_bayesParam, priorMode, priorScale, priorSkew,
            m_modelType.ThrConverge(), m_modelType.IterLimit(), m_modelType.HighAccuracy(),
            beta );*/
    }
    else if( hyperParamPlan.hasGrid() ) {    //cv with the grid

        unsigned planSize = hyperParamPlan.plansize();
        vector<double> lglkliMean( planSize, 0.0 );
        vector<double> lglkliMeanSqu( planSize, 0.0 );
        const double randmax = RAND_MAX;

        const std::vector<double>& hpplan = hyperParamPlan.plan();

        //prepare CV
        unsigned nfolds = hyperParamPlan.nfolds(); //10
        if( nfolds>drs.n() ) {
            nfolds = drs.n();
            Log(1)<<"\nWARNING: more folds requested than there are data. Reduced to "<<nfolds;
        }
        unsigned nruns = hyperParamPlan.nruns(); //2
        if( nruns>nfolds )  nruns = nfolds;

        vector<int> rndind = PlanStratifiedSplit( drs, nfolds );

        vector<bool> foldsAlreadyRun( nfolds, false );

        //cv loop
        for( unsigned irun=0; irun<nruns; irun++ )
        {
            //next fold to run
            unsigned x = unsigned( rand()*(nfolds-irun)/(randmax+1) );
            unsigned ifold, y=0;
            for( ifold=0; ifold<nfolds; ifold++ )
                if( !foldsAlreadyRun[ifold] ) {
                    if( y==x ) break;
                    else y++;
                }
            foldsAlreadyRun[ifold] = true;
            Log(5)<<"\nCross-validation "<<nfolds<<"-fold; Run "<<irun+1<<" out of "<<nruns<<", fold "<<ifold+1;

            //run cv part for a given fold
            SamplingRowSet cvtrain( drs, rndind, ifold, ifold+1, false ); //int(randmax*ifold/nfolds), int(randmax*(ifold+1)/nfolds)
            SamplingRowSet cvtest( drs, rndind, ifold, ifold+1, true ); //int(randmax*ifold/nfolds), int(randmax*(ifold+1)/nfolds)
            Log(5)<<"\ncv training sample "<<cvtrain.n()<<" rows"<<"  test sample "<<cvtest.n()<<" rows";
            InvData invData( cvtrain ); //<InvTripletItr>
            CVLoop hyperParLoop( cvtrain, invData, cvtest,
                m_modelType, hyperParamPlan,
                priorMode, priorScale, priorSkew );

            // process cv fold results
            Log(5)<<"\nCross-validation test log-likelihood: ";
            for( unsigned iparam=0; iparam<planSize; iparam++ ) {
                double eval = hyperParLoop.Loglikeli().at( iparam );
                Log(5)<<eval<<" ";
                //arith ave of loglikeli, i.e. geometric mean of likelihoods
                lglkliMean[iparam] = lglkliMean[iparam]*irun/(irun+1) + eval/(irun+1);
                lglkliMeanSqu[iparam] = lglkliMeanSqu[iparam]*irun/(irun+1) + eval*eval/(irun+1);
            }
        }//cv loop

        vector<double> lglkliStdErr; //( hpplan.size(), 0.0 );
        for(unsigned i=0; i<planSize; i++ ){
            double stddev_square = lglkliMeanSqu[i] - lglkliMean[i]*lglkliMean[i];
            if( stddev_square<0 ) stddev_square=0; //could happen due to precision loss
            lglkliStdErr.push_back( sqrt( stddev_square / double(nruns) ) ); // 2.10     Jun 20, 05  bug with stderr denom calc
            //cout<<"\nstderr "<<lglkliMean[i]<<" "<<lglkliMeanSqu[i]<<" "<<lglkliStdErr[i];
        }

        // best by cv
        double bestEval = - numeric_limits<double>::max();
        unsigned bestParam = unsigned(-1);
        Log(6)<<"\nCross-validation results - hyperparameter values, "
            <<(Laplace==hyperParamPlan.PriorType()?"prior var, ":"")
            <<"cv mean loglikelihood, std.error:";
        for( unsigned i=0; i<planSize; i++ ) {
            if( lglkliMean[i]>bestEval ) {
                bestEval = lglkliMean[i];
                bestParam = i;
            }
            Log(6)<<"\n\t"<<hpplan[i];
            if(Laplace==hyperParamPlan.PriorType()) 
                Log(6)<<"\t"<<hyperpar2var(hyperParamPlan.PriorType(),hyperParamPlan.skew(),hpplan[i]);
            Log(6)<<"\t"<<lglkliMean[i]<<"\t"<<lglkliStdErr[i];
        }
        if( bestParam==unsigned(-1) )
            throw runtime_error("No good hyperparameter value found");
        if( hyperParamPlan.ErrBarRule() ) {
            double valueneeded = lglkliMean[bestParam] - lglkliStdErr[bestParam];
            unsigned oldBestParam = bestParam;
            for( unsigned i=0; i<planSize; i++ )
                if( lglkliMean[i]>=valueneeded ) {
                    bestParam = i;
                    break;
                }
            Log(6)<<"\nError bar rule selected hyperparameter value "<<hpplan[bestParam]<<" instead of "<<hpplan[oldBestParam];
        }
        Log(3)<<"\nBest hyperparameter value "<<hpplan[bestParam]<<" cv-average loglikely "<<bestEval;

        //build final model
        m_bayesParam = hyperParamPlan.InstBayesParam( bestParam );
        Log(3)<<std::endl<<"Starting final model after cv, Time "<<Log.time();

    }
    else if( hyperParamPlan.search() ) {  // new: search without a grid

        //prepare CV
        const double randmax = RAND_MAX;
        unsigned nfolds = hyperParamPlan.nfolds(); //10
        if( nfolds>drs.n() ) {
            nfolds = drs.n();
            Log(1)<<"\nWARNING: more folds requested than there are data. Reduced to "<<nfolds;
        }
        unsigned nruns = hyperParamPlan.nruns();
        if( nruns>nfolds )  nruns = nfolds;
        vector<int> rndind = PlanStratifiedSplit( drs, nfolds );

        //prepare search
        double tryvalue = NormBasedDefaultVar( drs.dim(), stats, m_modelType.Standardize() );
        UniModalSearch searcher;
        const double eps = 0.05; //search stopper

        Vector localbeta = priorMode; //reuse this through all loops; can do better?

        //search loop
        while( true )
        {
            //try tryvalue
            BayesParameter bp = hyperParamPlan.InstByVar( tryvalue );
            double lglkli = 0; //would be the result for tryvalue
            double lglkliSqu = 0; //for stddev

            vector<bool> foldsAlreadyRun( nfolds, false );
            for( unsigned irun=0; irun<nruns; irun++ ) //cv loop
            {
                //next fold to run
                unsigned x = unsigned( rand()*(nfolds-irun)/(randmax+1) );
                unsigned ifold, y=0;
                for( ifold=0; ifold<nfolds; ifold++ )
                    if( !foldsAlreadyRun[ifold] ) {
                        if( y==x ) break;
                        else y++;
                    }
                foldsAlreadyRun[ifold] = true;
                Log(5)<<"\nCross-validation "<<nfolds<<"-fold; Run "<<irun+1<<" out of "<<nruns<<", fold "<<ifold+1;

                //run cv part for a given fold
                SamplingRowSet cvtrain( drs, rndind, ifold, ifold+1, false ); //int(randmax*ifold/nfolds), int(randmax*(ifold+1)/nfolds)
                SamplingRowSet cvtest( drs, rndind, ifold, ifold+1, true ); //int(randmax*ifold/nfolds), int(randmax*(ifold+1)/nfolds)
                Log(5)<<"\ncv training sample "<<cvtrain.n()<<" rows"<<"  test sample "<<cvtest.n()<<" rows";

                // build the model
                InvData invData( cvtrain );
                localbeta = ZOLR( invData,
                    bp, priorMode, priorScale, priorSkew, 
                    m_modelType.ThrConverge(), m_modelType.IterLimit(), m_modelType.HighAccuracy(),
                    localbeta );
                    //Log(9)<<"\nlocalBeta "<<localbeta;

                double run_lglkli = LogLikelihood( cvtest, m_modelType, localbeta );
                lglkli = lglkli*irun/(irun+1) + run_lglkli/(irun+1); //average across runs
                lglkliSqu = lglkliSqu*irun/(irun+1) + run_lglkli*run_lglkli/(irun+1); //for stddev

            }//cv loop

            double lglkli_stddev = sqrt( lglkliSqu - lglkli*lglkli );
            Log(5)<<"\nPrior variance "<<tryvalue<<" \tcv loglikeli "<<lglkli<<" \tstddvev "<<lglkli_stddev;
            searcher.tried( tryvalue, lglkli, lglkli_stddev );
            pair<bool,double> step = searcher.step();
            tryvalue = step.second;
            if( ! step.first )
                break;                      //----->>----
        }//search loop
        Log(3)<<"\nBest prior variance "<<tryvalue; 

        //build final model
        m_bayesParam = hyperParamPlan.InstByVar( tryvalue ); //SIC! use smoothing from the model //was searcher.bestx();
        Log(3)<<std::endl<<"Starting final model, Time "<<Log.time();

    }  // new: search without a grid

    Log(3)<<"\nFinal prior variance value "<<m_bayesParam.priorVar(); //v2.61 !IMPORTANT: used by bootstrap.pl
    drs.rewind();
    InvData invData( drs );
    m_beta = ZOLR( invData, //<InvTripletItr>
        m_bayesParam, priorMode, priorScale, priorSkew,
        m_modelType.ThrConverge(), m_modelType.IterLimit(), m_modelType.HighAccuracy(),
        beta );
}

static unsigned nonzeroes( const Vector& beta, const Vector& priorMode ) {
    unsigned nz=0;
    for( unsigned i=0; i<beta.size()-1; //sic! - intercept not counted
        i++ )
        if( beta[i]!=priorMode[i] ) nz++;
    return nz;
}
void ZOLRModel::squeezerModel(   IRowSet & drs,
                    const BayesParameter& bayesParameter,
                    unsigned squeezeTo,
                    double hpStart,
                    const Vector& priorMode, const Vector& priorScale, const Vector& priorSkew )
{
    Vector beta = priorMode; //( 0.0, drs.dim() );
    BayesParameter localBayesParam;
    if( Laplace!=bayesParameter.PriorType() )
        throw runtime_error("Squeezer error: only Laplace prior compatible with squeezer");

    drs.rewind();
    InvData invData( drs ); //<InvTripletItr>

    Log(3)<<"\nStarting squeezer, target #feats="<<squeezeTo<<" Starting hp="<<hpStart;
    // build initial model
    localBayesParam = BayesParameter( bayesParameter.PriorType(), hpStart, bayesParameter.skew() );
    beta = ZOLR( invData, localBayesParam, priorMode, priorScale, priorSkew, //<InvTripletItr>
        m_modelType.ThrConverge(), m_modelType.IterLimit(), m_modelType.HighAccuracy(),
        beta );
    unsigned nfeats = nonzeroes( beta, priorMode );
    Log(3)<<"\nSqueezer hp="<<hpStart<<" #feats="<<nfeats;
    if( nfeats<=squeezeTo ) {
        Log(3)<<"\nFinal parameter value "<<hpStart;
        m_beta = beta;
        m_bayesParam = localBayesParam;
        return;        //-------------->>--
    }

    double hplow = hpStart;
    //search for upper bound
    double hpup = hplow;
    unsigned nFeatsUp, nFeatsLow;//goes along with hplow and hpup - opposite
    const double a=10;
    while( nfeats>squeezeTo ) {
        hplow = hpup;
        nFeatsUp = nfeats;
        hpup *= a;
        localBayesParam = BayesParameter( bayesParameter.PriorType(), hpup, bayesParameter.skew() );
        beta = ZOLR( invData, localBayesParam, priorMode, priorScale, priorSkew, //<InvTripletItr>
            m_modelType.ThrConverge(), m_modelType.IterLimit(), m_modelType.HighAccuracy(),
            beta );
        nfeats = nonzeroes( beta, priorMode );
        Log(3)<<"\nSqueezer hp="<<hpup<<" #feats="<<nfeats;
    }
    nFeatsLow = nfeats;

    double hp = hpup;//init in case it's final
    Vector bestBeta = beta;
    bool monotone = true;
    while( nfeats!=squeezeTo && monotone )
    {
        hp = sqrt(hpup*hplow);
        localBayesParam = BayesParameter( bayesParameter.PriorType(), hp, bayesParameter.skew() );
        beta = ZOLR( invData, localBayesParam, priorMode, priorScale, priorSkew, //<InvTripletItr>
            m_modelType.ThrConverge(), m_modelType.IterLimit(), m_modelType.HighAccuracy(),
            beta );
        nfeats = nonzeroes( beta, priorMode );
        Log(3)<<"\nSqueezer hp="<<hp<<" #feats="<<nfeats;
        Log(5)<<setprecision(12)<<" hplow="<<hplow<<" hpup="<<hpup;
        if( nfeats > squeezeTo ) {
            if( nfeats>nFeatsUp ) {
                monotone = false;
            }
            else {
                nFeatsUp = nfeats;
                hplow=hp;
            }
        }
        else {  //( nfeats < squeezer.NFeats() )
            if( nfeats<=nFeatsLow ) { //equality means loop
                monotone = false;
            }
            else {
                hpup=hp;
                nFeatsLow = nfeats;
                bestBeta = beta;
            }
        }
    }
    if(!monotone) {
        hp = hpup;
        beta = bestBeta;
        Log(5)<<"\n!non-monotone";
    }

    Log(3)<<"\nFinal parameter value "<<hp;
    m_bayesParam = BayesParameter( bayesParameter.PriorType(), hp, bayesParameter.skew() );
    m_beta = beta;
}

void ZOLRModel::Train( const char* topic,
                       RowSetMem & trainData,
                       IRowSet * trainPopData,
                       const HyperParamPlan& hyperParamPlan,
                       const HyperParamPlan& hyperParamPlan2,
                       const Squeezer& squeezer,
                       const class PriorTermsByTopic& priorTermsByTopic,
                       const class DesignParameter& designParameter,
                       const class ModelType& modelType,
                       WriteModel& modelFile,
                       ResultsFile& resFile ) //std::ostream& result, ResultFormat resultFormat )
{
    //const BayesParameter& bayesParameter = hyperParamPlan.bp();
    m_bTrained = false;
    m_topic = topic;
    m_modelType = modelType;
    m_bHier = hyperParamPlan2.valid() && trainData.ngroups()>=1;
    if(m_bHier &&( hyperParamPlan.skew() || hyperParamPlan2.skew() ))
        throw runtime_error("Skewed priors not supported with hierarchies");

    if( designParameter.DesignType()==designPlain )
        m_pDesign= new PlainDesign( trainData );
    else if( designParameter.DesignType()==designInteractions ) {
        throw logic_error("Interactions Design not supported with ZO");
    }
    else
        throw logic_error("Undefined Design type");
    Log(3)<<"\nDesign: "<<m_pDesign->dim()<<" variables";

    DesignRowSet drs( m_pDesign, trainData );
    Stats stats( drs );
    drs.rewind();

    // prior terms
    Vector priorMode( 0.0, m_pDesign->dim() );
    Vector priorScale( 1.0, m_pDesign->dim() );
    Vector priorSkew( hyperParamPlan.skew(), m_pDesign->dim() );
    const IndPriors& indPriors = priorTermsByTopic.GetForTopic(string(topic));
        //!HACK: flat intercept desigh assumed==>
    for( size_t i=0; i<trainData.dim() /*skip the 'unit' column added above*/; i++ ) 
        if( indPriors.HasIndPrior( trainData.colId( i ) ) )
        {
            priorMode[i] = indPriors.PriorMode( trainData.colId( i ) );
            priorScale[i] = sqrt( indPriors.PriorVar( trainData.colId( i ) ) );
            priorSkew[i] = indPriors.PriorSkew( trainData.colId( i ) );
            Log(5)<<"\nIndPrior "<<trainData.colId( i )<<" "<<priorMode[i]<<" "<<priorScale[i]<<" "<<priorSkew[i];
        }
    if( indPriors.HasIndPrior( 0 ) ) //intercept
    {
        priorMode[trainData.dim()] = indPriors.PriorMode( 0 );
        priorScale[trainData.dim()] = sqrt( indPriors.PriorVar( 0 ) );
        priorSkew[trainData.dim()] = indPriors.PriorSkew( 0 );
        Log(5)<<"\nIndPrior intercept "<<priorMode[trainData.dim()]<<" "<<priorScale[trainData.dim()]<<" "<<priorSkew[trainData.dim()];
    }

    IRowSet* p_workingRs = &drs;
    IRowSet* p_stdRS=0;
    Vector origPriorMean; //, origPriorScale;
    if( modelType.Standardize() ) {
        Vector means = stats.Means();
        Vector stddevs = stats.Stddevs();
        // !!! intercept - aware !
        means[ means.size()-1 ]  = 0;
        stddevs[ stddevs.size()-1 ]  = 1;
        p_stdRS = new StandardizedRowSet( drs, means, stddevs );
        p_workingRs = p_stdRS;
        origPriorMean = priorMode;
        for( unsigned i=0; i<priorMode.size(); i++ )  {
            if( 0<stddevs[i] )
                priorMode[i] *= stddevs[i];
        }
        Log(9)<<"\norigPriorMean "<<origPriorMean<<"\nstats.Means "<<stats.Means()<<"\npriorMean "<<priorMode
	      <<"\nstats.Stddevs "<<stats.Stddevs()<<"\npriorScale "<<priorScale;
    }

    //this sets m_beta and m_bayesParam
    if( squeezer.enabled() )
        squeezerModel( *p_workingRs, hyperParamPlan.bp(), squeezer.NFeats(), squeezer.HPforTopic(topic),
            priorMode, priorScale, priorSkew);
    else if( m_bHier )
        tuneHierModel( *p_workingRs, //bayesParameter, 
            hyperParamPlan, hyperParamPlan2,
            priorMode, priorScale, priorSkew, stats );
    else
        tuneModel( *p_workingRs, //bayesParameter, 
            hyperParamPlan,
            priorMode, priorScale, priorSkew, stats );

    if( modelType.Standardize() ) {
        delete p_stdRS;
        priorMode = origPriorMean;
        //recalc beta's
        Log(12)<<"\nBeta-prime "<<m_beta;
        double interceptAdjust = 0;
        for( unsigned i=0; i<m_beta.size()-1/*!intercept*/; i++ )  {
            if( 0!=stats.Stddevs()[i] )
                m_beta[i] /= stats.Stddevs()[i];
            interceptAdjust -= m_beta[i] * stats.Means()[i];
        }
        m_beta[ m_beta.size()-1 ]  += interceptAdjust;
    }

    //Beta components dropped finally
    if( m_bHier )
        ReportSparsity( Log(3), m_hierW, priorMode );
    else {
        unsigned bdropped = 0;
        for( unsigned i=0; i<m_beta.size(); i++ ) //intercept included
            if( priorMode[i] == m_beta[i] )
                bdropped ++;
        Log(3)<<"\nBeta components dropped finally: "<<bdropped<<" Left: "<<m_beta.size()-bdropped;
        if( priorMode[ m_beta.size()-1 ] == m_beta[ m_beta.size()-1 ] ) Log(3)<<"\tIntercept dropped.";
    }
    //Log(10)<<"\nBeta "<<m_beta;

    // tune threshold
    vector<double> resubstScore( drs.n(), 0.0 );
    BoolVector y( false, drs.n() );
    bool trecFmt = false;
    for( unsigned r=0; drs.next(); r++ ) {
      resubstScore[r] = m_bHier ?
            dotProductHier( drs.xsparse(), drs.groups(), m_hierW )
            : dotProduct( drs.xsparse() ); //dotSparseByDense( drs.xsparse(), m_beta )
      y[r] = drs.y();
    }

    double threshold;
    
    if (modelType.TuneEst()) {
      // Tuning based on the estimated probabilities for the entire
      // (large) Training Population;
#ifdef USE_LEMUR

      Stats stat1(*trainPopData );
      Log(4) << "\nB2: |trainPopData| = " << stat1.AvgSquNorm();


      DesignRowSet tpdrs( m_pDesign, *trainPopData );

      Stats stat2(tpdrs );
      Log(4) << "\nB2: |trainPopData| = " << stat2.AvgSquNorm();

      tpdrs.rewind();

      vector<pair<double,double> > score_and_p_hat;
     
      for( unsigned r=0; tpdrs.next(); r++ ) {
        double score = m_bHier ?
            dotProductHier( tpdrs.xsparse(), tpdrs.groups(), m_hierW )
            : dotProduct( tpdrs.xsparse() ); //dotSparseByDense( tpdrs.xsparse(), m_beta )
        double p_hat =
            (modelType.Link()==ModelType::probit? combinedProbNorm(score) 
            : 1.0/(1.0+exp(-score)) );//logit
        score_and_p_hat.push_back(pair<double,double>(score,p_hat));
      }

      threshold = tuneThresholdEst(score_and_p_hat, modelType);
#else
      throw logic_error("B2 tuning not supported w/o LEMUR");      
#endif
    } else {
      // Tuning based on the (small) labeled Training set
      threshold = tuneThreshold( resubstScore, y, modelType );
    }

    //Log(12)<<"\nTuned threshold="<<threshold;

    // save model
    m_threshold = threshold;
    m_bTrained = true;
    vector<int> featSelect;
    for( unsigned i=0; i<trainData.dim(); i++ )
        featSelect.push_back( trainData.colId(i) );
    if( m_bHier )
      modelFile.WriteTopicHierSparse( m_topic, m_modelType, designParameter, featSelect, m_hierW, m_threshold );
    else
      modelFile.WriteTopic( m_topic, m_modelType, designParameter, featSelect, m_beta, m_threshold );

    // report model
    if( m_bHier ){//TODO
    }else{
        Log(2)<<std::endl<<"Beta:";
        m_pDesign->displayModel( Log(2), m_beta, threshold );
    }

    // resubstitution - evaluate
    BoolVector resubst( drs.n() );
    for( unsigned i=0; i<drs.n(); i++ )
        resubst[i] =( resubstScore[i]>=threshold );
    Log(3)<<"\n\n---Resubstitution results---";
    displayEvaluation( Log(3), y, resubst );

    double trainLogLikeli = m_bHier ?
            LogLikelihoodHier( drs, modelType, m_hierW )
            : LogLikelihood( drs, modelType, m_beta );
    Log(3)<<"\nTraining set loglikelihood "<<trainLogLikeli<<" Average "<<trainLogLikeli/drs.n();
    double logPrior;
    if(m_bHier) {
        pair<double,double> logPrior1and2 = LogPrior( 
            m_bayesParam, m_bayesParam2, priorMode, priorScale, priorSkew, m_hierW );
        logPrior = logPrior1and2.first + logPrior1and2.second;
        Log(3)<<"\nLog prior Level 1 "<<logPrior1and2.first<<"  Level 2 "<<logPrior1and2.second
            <<"  Total "<<logPrior;
    }
    else{
        logPrior = LogPrior( m_bayesParam, priorMode, priorScale, priorSkew, m_beta );
        Log(3)<<"\nLog prior (penalty) "<<logPrior;
    }
    Log(3)<<" Log posterior "<<trainLogLikeli+logPrior;

    //roc
    std::vector< std::pair<double,bool> > forROC;
    for( unsigned i=0; i<drs.n(); i++ )
        forROC.push_back( pair<double,bool>( resubstScore[i], y[i] ) );
    double area = calcROC(forROC);
    Log(3)<<"\nROC area under curve "<<area;

    if( Log.level()>=12 ) {
        Log()<<"Resubstitution Scores";
        for( unsigned i=0; i<drs.n(); i++ )
            Log()<<endl<<resubstScore[i]<<":"<<resubst[i];
    }

    // write results file
    drs.rewind();
    for( unsigned r=0; drs.next(); r++ ) {
        double p_hat =
            ( modelType.Link()==ModelType::probit ? combinedProbNorm(resubstScore[r]) 
            : 1.0/(1.0+exp(-resubstScore[r])) );
        resFile.writeline( topic, drs.currRowName(), false, //isTest
            y[r], resubstScore[r], p_hat, (resubstScore[r]>=threshold) );
    }

}

double LogLikelihood(
                IRowSet & rowset, // HACK!!! DesignRowSet needed: colId's are 0, 1, 2... 
                const class ModelType& modelType,
                const Vector& beta
            )
{
    rowset.rewind();
    double logLhood = 0;

    // fit term
    for( unsigned r=0; rowset.next(); r++ ) {
        double score = dotSparseByDense( rowset.xsparse(), beta );
        logLhood += PointLogLikelihood(
                score,
                rowset.y(),
                modelType );
    }

    return logLhood;
}

double LogLikelihoodHier(
                IRowSet & rowset, // HACK!!! DesignRowSet needed: colId's are 0, 1, 2... 
                const class ModelType& modelType,
                const THierW& w
            )
{
    rowset.rewind();
    double logLhood = 0;

    // fit term
    for( unsigned r=0; rowset.next(); r++ ) {
        double score = dotProductHier( rowset.xsparse(), rowset.groups(), w );
        logLhood += PointLogLikelihood(
                score,
                rowset.y(),
                modelType );
    }

    return logLhood;
}

double LogTraceBased(
                IRowSet & trainDesignData, // HACK!!! DesignRowSet needed: colId's are 0, 1, 2... 
                const class ModelType& modelType,
                const BayesParameter& bayesParam,
                const Vector& priorMode,
                const Vector& priorScale,
                const Vector& beta
            )
{
    unsigned d = beta.size();
    trainDesignData.rewind();
    Vector diagonal( 0.0, d ); //of the Hessian of minus log likelihood
    for( unsigned r=0; trainDesignData.next(); r++ ) {

        SparseVector x = trainDesignData.xsparse();

        double score = 0.0;
        for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
            score += beta[ix->first] * ix->second;
        double denom;
        if( ModelType::logistic==modelType.Link() )
            denom = exp(score) + exp(-score) + 2;
        else
            throw logic_error("only logistic link is supported for parameter tuning");

        for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
            diagonal[ix->first] += ix->second*ix->second / denom;

    }

    //new for ver 3
    for( unsigned j=0; j<d; j++ )
        switch( bayesParam.PriorType() ) {
        case Laplace:
            if( priorMode[j]==beta[j] )
                //see penalty calculation at the beginning of ZOLR
                diagonal[j] += (bayesParam.gamma()/priorScale[j])*(bayesParam.gamma()/priorScale[j])/2;
            break;
        case normal:
            diagonal[j] += 1/(2*bayesParam.var()*priorScale[j]*priorScale[j]); //'2*' fixes bug, ver3.01
            break;
        default:
            throw logic_error("only normal and Laplace priors supported for evidence calculations");
        }
    
    double trace = 0.0;
    for( unsigned j=0; j<d; j++ )
        trace += diagonal[j];

    return - 0.5 * double(d) * log(trace/double(d));
}

double LogPrior(
                const BayesParameter& bayesParameter,
                const Vector& priorMode,
                const Vector& priorScale,
                const Vector& priorSkew,
                const vector<double>& beta
            )
{
    double penaltyLogLhood = 0;
    for( unsigned i=0; i<beta.size(); i++ ) {
        double betaLogLhood;
        if( numeric_limits<double>::infinity()==priorScale[i] )
            betaLogLhood = 0;
        else if( normal==bayesParameter.PriorType() ) {
            double sigma = priorScale[i]*sqrt(bayesParameter.var());
            double u = (beta[i]-priorMode[i]) / sigma;
            betaLogLhood = -log(2*M_PI)/2 - log(sigma) - u*u/2;
        }
        else if( Laplace==bayesParameter.PriorType() ){
            //Laplace distribution density: (1/(2*b)) * exp(-fabs(x)/b)
            //log density: -log(2)-log(b)-fabs(x)/b
            //c.d.f. (1/2)*( 1 + (x>0 ? 1 : -1)*(1-exp(-fabs(x)/b) )
            //scale: sqrt(2)*b, thus b=scale/sqrt(2)
            double inverse_b = bayesParameter.gamma() / priorScale[i];
            if( priorSkew[i]==0 )
                betaLogLhood = - log(2.0) + log(inverse_b) - fabs(beta[i]-priorMode[i]) * inverse_b;
            else if( priorSkew[i]==1 ) {
                if( beta[i]>=priorMode[i] )
                    betaLogLhood = log(inverse_b) - (beta[i]-priorMode[i]) * inverse_b;
                else 
                    betaLogLhood = - numeric_limits<double>::infinity();
            }
            else if( priorSkew[i]==-1 ) {
                if( beta[i]<=priorMode[i] )
                    betaLogLhood = log(inverse_b) - (-beta[i]+priorMode[i]) * inverse_b;
                else 
                    betaLogLhood = - numeric_limits<double>::infinity();
            }
        }
        penaltyLogLhood += betaLogLhood;
        //Log(10)<<"\nbeta/lhood/log "<<beta[i]<<" "<<exp(betaLogLhood)<<" "<<betaLogLhood;
    }

    return penaltyLogLhood;
}

pair<double,double> //1st and 2nd level
LogPrior(
                const BayesParameter& bayesParameter,
                const BayesParameter& bayesParameter2,
                const Vector& priorMode,
                const Vector& priorScale,
                const Vector& priorSkew,
                const THierW& w
            )
{
    double logPrior2=0;
    int ngroups = w.at(0).size();
    Vector pseudoMode, pseudoScale, pseudoSkew; //imitations for the group level calcs
    vector<double> wcenter( w.size(), 0.0 ); //shrunk medians or means
    for( unsigned j=0; j<w.size(); j++ ) {
        double penaltyRatio;
        double sm; // shrunk mean or median
        switch(bayesParameter2.PriorType()) {
        case  normal:
            penaltyRatio = bayesParameter2.var()/bayesParameter.var();
            double wsum; wsum=0;
            for( int g=0; g<ngroups; g++ )   wsum+=w.at(j).at(g);
            sm = (priorMode[j]*penaltyRatio + wsum)/(penaltyRatio + ngroups);
            break;
        case Laplace:
            penaltyRatio = bayesParameter.gamma()/bayesParameter2.gamma();
            vector<double> wsorted = w.at(j);
            sort( wsorted.begin(), wsorted.end() );
            sm = shrunkMedian( wsorted, priorMode[j], penaltyRatio );
            //Log(6)<<"\nshrunkMedian  "<<wsorted<<" "<<penaltyRatio<<"  "<<priorMode[j]<<"  "<<sm;
            break;
        }
        wcenter.at(j) = sm;
        pseudoMode.resize( ngroups, sm );
        pseudoScale.resize( ngroups, priorScale.at(j) );
        pseudoSkew.resize( ngroups, 0 );  //skew unsupported for hierarchies
        logPrior2 += LogPrior( bayesParameter2, pseudoMode, pseudoScale, pseudoSkew, w.at(j) );
    }

    //Log(6)<<"\n wcenter "<<wcenter;
    return pair<double,double>( 
        LogPrior( bayesParameter, priorMode, priorScale, priorSkew, wcenter ),
        logPrior2 
       );
}


/*double AvgSquNorm( IRowSet& rs ) //for default bayes parameter: avg x*x
{
    rs.rewind();
    double avgss = 0;
    double n = 0;
    while( rs.next() ) {
        SparseVector x = rs.xsparse();
        double xsqu = 0;
        for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
            xsqu += ix->second * ix->second;
        avgss = avgss*n/(n+1) + xsqu/(n+1);
        n ++;
    }
    return avgss;
}*/

//  cout<<"\n! "<<__LINE__;cout.flush();

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
