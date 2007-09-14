/*
 * Zhang & Oles regularized logistic regression
 * applied to polytomous here
 *
 * 1.01   Feb 23 05     error bar rule for hyperparameter cv (option --errbar)
 *                      'wakeUpPlan' does not depend on iter.limit
 * 1.05   Jun 20, 05    one std error rule: fixed bug with stderr denom calc //PolyZO.cpp
 * 1.60   Oct 24, 05    sparse beta in model file
 *                      Individual priors
 *                      fixed log.prior with infinite prior var
 * 3.01   Jan 31, 06    fixed bug: Gaussian penalty should be 1/(2*var), not 1/var
 * 4.02   May 30, 07    reserve space for inverted index at the beginning
 */
#define  _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <limits>
#include <iomanip>

#include "global.h"
#include "logging.h"
#include "Matrix.h"
#include "BayesParamManager.h"
#include "PriorTerms.h"
#include "dataPoly.h"
#include "Design.h"
#include "poly.h"
#include "ModelTypeParam.h"
#include "stats.h"
#include "ModelFile.h"
#include "StrataSplit.h"

using namespace std;

#ifdef _MSC_VER
#define finite(x) (_finite(x))
#endif

void /*returns model coeffs thru the last arg*/
QN(
                IRowSet& xdata,
                const BayesParameter& bayesParam, //input
                const IParamMatrix & priorMean,  //input
                const IParamMatrix & priorScale,  //input
                const ModelType& modelType,  //input
                const class FixedParams& fixedParams,  //input
                ParamMatrixSquare & w  // i/0: init/result values of model coeffs, 
            );

static Vector Stddevs( IRowSet & drs );
static double LogLikelihood(
                IRowSet & trainDesignData, // HACK!!! DesignRowSet needed: colId's are 0, 1, 2... 
                const class ModelType& modelType,
                const IParamMatrix& beta
            );
static double LogPrior(
                const BayesParameter& bayesParameter,
                const IParamMatrix& priorMean,
                const IParamMatrix& priorScale,
                const IParamMatrix& beta
            );

/*template<class T> std::ostream& operator<<( std::ostream& s, const vector< vector<T> >& v ) {
    for( size_t i=0; i<v.size(); i++ ) {
        s<<endl;
        for( size_t j=0; j<v[i].size(); ++ ) 
            s<<v[i][j]<<" ";
    }
    return s;
};*/

template <class InvDataItr> class InvData {
public:
    pair<InvDataItr,InvDataItr> getRange(unsigned var) const;
    unsigned n() const;
    unsigned c() const; // #classes in 'y'
    unsigned y( unsigned row ) const;
};

template <class InvDataItr> double OneCoordStep(
    double dw_numerator,
    double dw_denominator,
    const pair<InvDataItr,InvDataItr>& range,
    const InvData<InvDataItr>& invData, //for 'y' only
    unsigned k, //class
    const vector< vector<double> >& WTX, //added for exp overflow
    const vector<double>& wkTX,
    const vector<double>& exp_wkTX,
    const vector<double>& sumk_exp_wTX,
    double trustregion
    )
{
            //that might be a good idea double ii = 0;
	        for( InvDataItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
            {
                unsigned i = ix.i(); //case #
                double x = ix.x(); //value
                if( 0.0==x ) continue; //just in case - shouldn't ever hit this
                unsigned target = invData.y(i)==k ? 1 : 0;

                //if( exp_wkTX[i] > sumk_exp_wTX[i] )   Log(6)<<"\nErrSumExp: k i x target sum_exp_wTX[i] exp_wkTX[i] "
                //        <<k<<" "<< i<<" "<< x<<" "<< target<<" "<<setprecision(18)<<sumk_exp_wTX[i] <<" "<<setprecision(18)<<exp_wkTX[i];
                double D;
                double delta = trustregion * fabs(x);
                bool numbranch = false;
                //Log(10)<<"\ndelta = trustregion * fabs(x) "<<delta<<" "<<trustregion<<" "<<x;
                if( finite(exp_wkTX[i]) && finite(sumk_exp_wTX[i]) && sumk_exp_wTX[i]>0 ) { //
                    const double exp_wkTX_i = exp_wkTX[i];
                    const double K = sumk_exp_wTX[i] - exp_wkTX_i;
                    const double e_delta = exp(delta);
                    const double e_rlo = exp_wkTX_i / e_delta;
                    const double e_rhi = exp_wkTX_i * e_delta;
                    if( e_rlo<=K && K<=e_rhi )
                        D = 4;
                    else
                        D = min( e_rlo/K+K/e_rlo, e_rhi/K+K/e_rhi ) + 2;
                    dw_numerator += (target ? x : 0) - x * exp_wkTX_i / sumk_exp_wTX[i]; //x*p_hat
                }
                else
                {
                    numbranch = true;
                    double invPminus1 = 0.0;
                    double qlo=0, qhi=0; // K/e_rlo, K/e_rhi
                    for( unsigned kk=0; kk<invData.c(); kk++ ) {
                        if( kk!=k ) {
                            invPminus1 += exp( WTX[kk][i] - WTX[k][i] );
                            qlo += exp( WTX[kk][i] - (WTX[k][i]-delta) ); //wkTX[i]
                            qhi += exp( WTX[kk][i] - (WTX[k][i]+delta) ); //wkTX[i]
                        }
                    }
                    if( 1<=qlo && qhi<=1 ) //e_rlo<=K && K<=e_rhi )
                        D = 4;
                    else
                        D = min( qlo+1/qlo, qhi+1/qhi ) + 2;
                    dw_numerator += (target ? x : 0) - x/(1+invPminus1);  //ver 0.24
                        //bad when invP==inf: x * ( target ? invPminus1 : -1 ) / (1+invPminus1);  
                        //ML only dbg = dw_numerator*ii/(ii+1) + (x *( target - p_hat ))/(ii+1);
                   //Log(7)<<"\nNumbranch exp_wkTX[i] sumk_exp_wTX[i] dw_numerator "<<exp_wkTX[i]<<" "<<sumk_exp_wTX[i]<<" "<<dw_numerator
                    //Log(10)<<"\nqlo qhi D dNum dDenom num denom "<<setw(23)<< qlo<<setw(23)<< qhi<<setw(23)<< D<<" ";
                }
                dw_denominator += x * x / D ;  //ML only= dw_denominator*ii/(ii+1) + (x * x / D)/(ii+1) ;
                //if( numbranch )
                //    Log(6)<<"\nNumbranch exp_wkTX[i] sumk_exp_wTX[i] dw_numerator "<<exp_wkTX[i]<<" "<<sumk_exp_wTX[i]<<" "<<dw_numerator;
                //Log(10)<<setw(23)<<( x * ( target ? invPminus1 : -1 ) / (1+invPminus1) )<<setw(23)<< x * x / D<<setw(23)<<dw_numerator<<setw(23)<<dw_denominator;
                //Log(10)<<"\ninvPhat "<<setw(23)<< invPminus1;
                //Log(7)<<setw(14)<< x *( target - p_hat )<<setw(14)<< x * x / D<<setw(14)<<dw_numerator*(ii+1)<<setw(14)<<dw_denominator*(ii+1);

                /*if( !finite(dw_numerator) ) {
                    /*** one possible cause:
                        wkTX_i becomes very negative like -750, so sumk_exp_wTX[i] approaches zero
                        *
                    Log(3)<<"\nErrNumer: k i x target sum_exp_wTX[i] exp_wkTX[i] dw_numer numbranch\n\t"
                        <<k<<" "<<i<<" "<<x<<" "<<target<<" "<<sumk_exp_wTX[i] <<" "<<exp_wkTX[i]<<" "<<dw_numerator<<" "<<numbranch;
                }
                if( !finite(dw_denominator) ) {
                    Log(3)<<"\nErrDenom: k i x sum_exp_wTX[i] exp_wkTX[i] D x*x/D dw_denom "
                        <<k<<" "<< i<<" "<< x<<" "<<sumk_exp_wTX[i] <<" "<< exp_wkTX[i]<<" "<<D<<" "<<x*x/D<<" "<<dw_denominator;
                }*/
                //ii++;
            }

            //update w
            double dw = 0.0==dw_numerator ? 0.0 : dw_numerator / dw_denominator;
            dw = min( max(dw,-trustregion), trustregion );
            if( !finite(dw) )
                Log(1)<<"\nErrDW k dw_numerator dw_denominator dw "<<k<<" "<<dw_numerator<<" "<<dw_denominator<<" "<<dw;
            return dw;
}

template <class InvDataItr> 
void /*returns model coeffs thru the last arg*/
ZOLR(
                unsigned d, //input - #vars in design
                unsigned c, //input - #classes
                const InvData<InvDataItr>& invData, //input - generates design data
                const BayesParameter& bayesParam, //input
                const IParamMatrix & priorMean,  //input
                const IParamMatrix & priorScale,  //input
                const ModelType& modelType,  //input
                const class FixedParams& fixedParams,  //input
                ParamMatrixSquare & w  // i/0: init/result values of model coeffs, 
            )
{
    Log(3)<<std::endl<<"Starting PolyZO model, - Time "<<Log.time();
    
    //_controlfp( /*_EM_INVALID | _EM_DENORMAL | _EM_ZERODIVIDE | _EM_OVERFLOW _EM_UNDERFLOW*/ 0, _MCW_EM );
    
    unsigned n = invData.n(); //input - #examples
    if( d==priorMean.D() && d==priorScale.D() ) ;//ok
    else    throw DimensionConflict(__FILE__,__LINE__);

    //convert prior scale into penalty parameter
    ParamMatrixSquare penalty( d, c );
    for( unsigned j=0; j<d; j++ )
    for( unsigned k=0; k<c; k++ )
    {
        if( normal==bayesParam.PriorType() ) 
	    penalty(j,k) = 1.0 /( 2*bayesParam.var()*priorScale(j,k)*priorScale(j,k) ); //'2*' fixes bug, ver2.01
        else if( Laplace==bayesParam.PriorType() )
            penalty(j,k) = bayesParam.gamma() / priorScale(j,k);
        else
            throw runtime_error("ZO only allows normal or Laplace prior");
//        Log(11)<<"\n k/j/priorScale(j,k)/priorMean(j,k)/penalty(j,k)/"<<k<<" "<<j<<" "<<priorScale(j,k)<<" "<<priorMean(j,k)<<" "<<penalty(j,k);
        cout<<k<<" "<<j<<" "<<priorScale(j,k)<<" "<<priorMean(j,k)<<" "<<penalty(j,k)<<endl;
    }

    /*ver 1.60 No longer in effect with the introduction of individual priors
    //HACK - no penalty for intercept
    for( unsigned k=0; k<c; k++ )
        penalty(d-1,k) = 0.0;
    Log(3)<<std::endl<<"No penalty for the intercept!";
    */
    // Log.setLevel(15);
    // initialize temp data
    vector< vector<double> > wTX( c, vector<double>(n,0.0) ); //dot products, by classes by cases
    for( unsigned j=0; j<d; j++ ) //--design vars--
    {
        pair<InvDataItr,InvDataItr> range = invData.getRange(j);
	    for( InvDataItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
            for( unsigned k=0; k<c; k++ ) //--classes--
            if( w(j,k) != 0.0 )
            {
                wTX[k][ ix->i() ] += w(j,k) * ix->x();
            }
    }
    Log(11)<<"\nwTX at start "<<wTX;
    double Lprev = 0;

    //vector< vector<double> > wTXprev = wTX; TODO remove it
#define KJ
#ifndef KJ
            vector< vector<double> > e_wTX( c, vector<double>(n,0.0) );
            for( unsigned i=0; i<n; i++ )
                for( unsigned k=0; k<c; k++ )
                    e_wTX[k][i] = exp( wTX[k][i] );
#endif

    /*dbg
    double sum=0;
    for( unsigned k=0; k<c; k++ )
        cout<<endl<<wTX[k][6973]<<" "<<exp( wTX[k][6973] )<<" "<<(sum+=exp( wTX[k][6973] ));
    */

    ParamMatrixSquare trustregion( d, c,  1.0);//could make it of integers
    unsigned ndropped;

    const unsigned IterLimit = 10000;  //(unsigned)sqrt(d);

    ParamMatrixSquare sleep( d, c,  0.0);
    class WakeUpPlan {
        vector<bool> plan;        
    public:
        bool operator()(unsigned int step ) {
            if( step<plan.size() ) return plan.at(step);
            else return 0==step%100;
        }
        WakeUpPlan(unsigned size=1000) : plan( size+1, false ) { //ctor
            for(unsigned i=0;i*i*i<size;i++) plan[i*i*i] = true;     }
    };
    WakeUpPlan wakeUpPlan;

    bool converge = false;
    unsigned itr;
    unsigned steps=0, stepsskipped=0, stepsslept=0;
    for( itr=0; !converge; itr++ ) //---iterations loop--
    {
        double sum_abs_dr = 0.0;
        ndropped = 0;
        unsigned classParams = modelType.ReferenceClass() ? c-1 : c; //should rather check fixedParams.refClassId() ?
        for( unsigned k=0; k<classParams; k++ ) //--classes--
        //for( unsigned j=0; j<d; j++ ) //--design vars--
        {
            //this could be outside all loops, but then error cumulates. ver 0.23
            vector<double> S( n, 0.0 );
            for( unsigned i=0; i<n; i++ )
                for( unsigned k=0; k<c; k++ )
                    S[i] += exp( wTX[k][i] );

#ifdef KJ //init class-specific data
            vector<double> e_wkTX( n, 0.0 );
            for( unsigned i=0; i<n; i++ )
                e_wkTX[i] = exp( wTX[k][i] );
#endif
            vector<double> dwkTX( n, 0.0 );
 
            for( unsigned j=0; j<d; j++ ) //--design vars--
            //for( unsigned k=0; k<c; k++ ) //--classes--
            {


		// added by shenzhi
		if(penalty(j,k)==numeric_limits<double>::infinity()) {
		    w(j,k) = priorMean(j,k);
		    continue;
		}

                if( fixedParams(j,k) ){
                        //Log(9)<<"\nSkip j/k "<<j<<" "<<k;
                        stepsskipped++;
                        continue;       //----->>--
                }
                else  steps++;

                if( w(j,k)==priorMean(j,k) )
                    if( ! wakeUpPlan( (int)sleep(j,k) ) ) {
                        sleep(j,k) ++;
                        stepsslept++;
                        ndropped++;
                        continue;       //----->>--
                    }
                    //else
                        //if(itr>0) Log(9)<<"\nTry Wake-up j/k/sleep "<<j<<" "<<k<<" "<<sleep(j,k);

                double dw;

                pair<InvDataItr,InvDataItr> range = invData.getRange(j);

#ifndef KJ
                const vector<double>& e_wkTX = e_wTX[k];
#endif
                if( normal==bayesParam.PriorType() )
                {
                    double dw_numerator = - 2 * ( w(j,k) - priorMean(j,k) ) * penalty(j,k);
                    double dw_denominator = 2 * penalty(j,k);
                    dw = OneCoordStep(
                        dw_numerator, dw_denominator, range, invData, k, wTX, wTX[k], e_wkTX, S, trustregion(j,k) );
                }
                else { //Laplace
                    if( w(j,k)-priorMean(j,k)>0 ) { //step for positive
                        dw = OneCoordStep( -penalty(j,k), 0, range, invData, k, wTX, wTX[k], e_wkTX, S, trustregion(j,k) );
                    }
                    else if( w(j,k)-priorMean(j,k)<0 ) { //step for negative
                        dw = OneCoordStep( penalty(j,k), 0, range, invData, k, wTX, wTX[k], e_wkTX, S, trustregion(j,k) );
                    }
                    else // at mean right now
                    { // try both directions, positive first
                        double dwPlus = OneCoordStep( -penalty(j,k), 0, range, invData, k, wTX, wTX[k], e_wkTX, S, trustregion(j,k) );
                        if( dwPlus>0 )
                            dw = dwPlus;
                        else {  //try negative
                            double dwMinus = OneCoordStep( penalty(j,k), 0, range, invData, k, wTX, wTX[k], e_wkTX, S, trustregion(j,k) );
                            if(!finite(dwMinus))   Log(1)<<"\nInfDW zero neg "<<j<<" "<<k;
                            if( dwMinus<0 )
                                dw = dwMinus;
                            else
                                dw = 0;
                        }
                    }

                    //check if we crossed the break point
                    if(( w(j,k) < priorMean(j,k) && priorMean(j,k) < w(j,k)+dw )
                        || ( w(j,k)+dw < priorMean(j,k) && priorMean(j,k) < w(j,k) )
                        )
                        dw = priorMean(j,k) - w(j,k); //stay at mean
                }

                w(j,k) += dw;
                if( w(j,k)==priorMean(j,k) )   ndropped++;
                //if(dw!=0) Log(7)<<"\nCoordStep k/j/dw/w/trustrgn "<<k<<" "<<j<<" "<<dw<<" "<<w(j,k);//<<" "<<trustregion(j,k);

                /*dbg*
                double L=LogLikelihood(data,modelType,w);
                Log(9)<<"\nk/j/w(j,k)/dw/loglikelihood "<<k<<" "<<j<<" "<<w(j,k)<<" "<<dw<<" "
                    <<setprecision(16)<<L;
                if( L<Lprev ) Log(9)<<"  JUMP  "<<Lprev-L;
                Lprev = L;
                *dbg*/

                //update local data
	            for( InvDataItr ix=range.first; ix!=range.second; ix++ ) // i such that xi,j != 0.0
                {
                    unsigned i = ix->i();
                    double x = ix->x();
                    if( 0.0==x ) continue; //just in case - shouldn't ever hit this
                    wTX[k][i] += dw * x;
                    dwkTX[i] += dw * x;
                    double e_r = exp( wTX[k][i] );
#ifdef KJ
                    double Sold = S[i];
                    S[i] -= e_wkTX[i]; S[i] += e_r;
                       /*dbg if(S[i]==0)   Log(6)<<"\nZeroSumExp k i x ewTXold ewTXnew Sold Snew "
                            <<k<<" "<<i<<" "<<x<<" "<<e_wkTX[i]<<" "<<e_r<<" "<<Sold<<" "<<S[i];*/
                    e_wkTX[i] = e_r;                
#else
                    S[i] -= e_wTX[k][i]; S[i] += e_r;
                    e_wTX[k][i] = e_r;
#endif
                }
                //trust region update
		        trustregion(j,k) = max( 2*fabs(dw), trustregion(j,k)/2 );   //2*fabs(dw)*1.1

                if( w(j,k)==priorMean(j,k) ) sleep(j,k)++;
                else {
                    //if( sleep[j]>0 && itr>0 ) Log(9)<<"\nWake-up j/sleep[j] "<<j<<"/"<<sleep[j];
                    sleep(j,k)=0;
                }

            }//--design vars-- j --
#ifdef KJ
            for( unsigned i=0; i<n; i++ )
                sum_abs_dr += fabs( dwkTX[i] );
            Log(9)<<"\n  Class "<<k<<"  Dot product abs change "<<sum_abs_dr<<"  Time "<<Log.time();
#endif
        }//--classes-- k --

        double sum_abs_r = 0;
        for( unsigned i=0; i<n; i++ ) 
            for( unsigned k=0; k<c; k++ ) {
                if( !finite(wTX[k][i]) )
                    Log(1)<<"\nNAN wTX[k][i] k i "<<wTX[k][i]<<" "<<k<<" "<<i;
                sum_abs_r += fabs(wTX[k][i]);
            }

#ifndef KJ
        sum_abs_dr_old = 0.0;
        for( unsigned i=0; i<n; i++ )
            for( unsigned k=0; k<c; k++ )
                sum_abs_dr_old += fabs(wTX[k][i]-wTXprev[k][i]);
#endif

        double rel_sum_abs_dr = sum_abs_dr/(1+sum_abs_r);//ZO stopping

        converge =( itr>=IterLimit 
            || rel_sum_abs_dr < modelType.ThrConverge()  );

        Log(7)<<"\nIteration "<<itr+1
            <<"  Dot product abs sum "<<sum_abs_r
            <<"  Rel change "<<rel_sum_abs_dr //ZO stopping
            <<"  Beta's dropped: "<<ndropped
            <<"  Time "<<Log.time();
        Log(7).flush();
        /*dbg*
        double L=LogLikelihood(data,modelType,w);
        Log(9)<<"\nLoglikelihood "<<setprecision(16)<<L;
        if( L<Lprev ) Log(9)<<"  JUMP  ";
        Lprev = L;
        *dbg*/

    }//---iter loop---

    Log(3)<<std::endl<<"Built ZO model "<<itr<<" iterations, Time "<<Log.time();
    Log(5)<<"\nTotal steps "<<steps<<", skipped "<<stepsskipped<<", slept "<<stepsslept;
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
template <> class InvData<InvTripletItr> {
    vector<triplet> data;
    vector<unsigned> varOffsets;
    unsigned nrows;
    unsigned m_c;
    vector<unsigned> m_y;
public:
    unsigned n() const { return nrows; }
    unsigned c() const { return m_c; } // #classes in 'y'
    unsigned y( unsigned row ) const { return m_y[row]; }
    InvData( IRowSet& rowset )
    {
        m_c = rowset.c();
        rowset.rewind();

        for( nrows=0; rowset.next(); nrows++ ) {
            const SparseVector& x=rowset.xsparse();
            for( SparseVector::const_iterator xitr=x.begin(); xitr!=x.end(); xitr++ )
                data.push_back( triplet( xitr->first, nrows, xitr->second ) );
            m_y.push_back( rowset.y() );
        }

        sort( data.begin(), data.end(), lessTriplet );

        //--- create index - var offsets ---
        varOffsets.resize( rowset.dim(), data.size() ); //init as non-existing
	//Log(12)<<"\nrowset.dim(), data.size() data[0].var "<<rowset.dim()<<" "<<data.size()<<" "<<data[0].var;
        unsigned vcurr = 0;
        varOffsets[vcurr] = 0;
        for( unsigned t=0; t<data.size(); t++ ) {
            unsigned vnext = data[t].var;
            if( vnext!=vcurr ) {
                for( unsigned v=vcurr+1; v<=vnext; v++ ) //may be a gap btw vnext and vcurr
                    varOffsets[v] = t;
                vcurr = vnext;
            }
        }
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

class HyperParLoop {
    IRowSet& drs;
    ModelType& modelType;
    const IParamMatrix& priorMean;
    const IParamMatrix& priorScale;
    const class FixedParams& fixedParams;
    vector<double> lglkli;
    mutable vector<double> allWeights;
    mutable double maxWeight;
    mutable unsigned maxWeightIndex;
public:
    const vector<double>& LogLikelihoods() const { return lglkli; }
    //it's all in the ctor
    HyperParLoop( //perform loop of hyperparameter values for a given train/validation split
        IRowSet& drs_,
        IRowSet& drsTest,
        ModelType modelType_,
        const HyperParamPlan& hpPlan,
        const IParamMatrix& priorMean_,
        const IParamMatrix& priorScale_,
        const class FixedParams& fixedParams_
        )   : drs(drs_), modelType(modelType_),
            priorMean(priorMean_), priorScale(priorScale_), fixedParams(fixedParams_)
    {        
        unsigned planSize = hpPlan.plan().size();
        ParamMatrixSquare localbeta( drs.dim(), drs.c() );
        InvData<InvTripletItr> * p_invData =
            ModelType::ZO==modelType.Optimizer() ? new InvData<InvTripletItr>(drs) : 0;

        for( unsigned iparam=0; iparam<planSize; iparam++ )//hyper-parameter loop
        {
            Log(5)<<"\nHyperparameter plan #"<<iparam+1<<" value="<<(hpPlan.plan()[iparam]);
            BayesParameter localBayesParam = hpPlan.InstBayesParam( iparam );

            // build the model
            if( ModelType::ZO==modelType.Optimizer() )
                ZOLR<InvTripletItr>( drs.dim(), drs.c(), *p_invData, 
                    localBayesParam, priorMean, priorScale, modelType, fixedParams,
                    localbeta );
#ifdef QUASI_NEWTON
            else if( ModelType::QN_2coord==modelType.Optimizer() || ModelType::QN_smooth==modelType.Optimizer() )
                QN( drs, localBayesParam, priorMean, priorScale, modelType, fixedParams,
                    localbeta );
#endif //QUASI_NEWTON
            else 
                throw logic_error("Unsupported optimizer");

            double eval = LogLikelihood( drsTest, modelType, localbeta );
            lglkli.push_back(eval);
        }//hyper-parameter loop

        delete p_invData;
    }
};//class HyperParLoop

static double AvgSquNorm( const Stats& stats ) { //separated from class Stats' to exclude constant term
    double s =0.0;
    for( unsigned j=0; j<stats.Means().size(); j++ ) {
        if( j==stats.Means().size()-1 ) //HACK: drop constant term
            break;
        s += stats.Means()[j]*stats.Means()[j] + stats.Stddevs()[j]*stats.Stddevs()[j];
    }
    return s;
}

void LRModel::tuneModel(   IRowSet & drs,
                    //const class BayesParameter& bayesParameter,
                    const HyperParamPlan& hyperParamPlan,
                    const IParamMatrix& priorMean, const IParamMatrix& priorScale,
                    const class FixedParams& fixedParams,
                    Stats& stats )
{
    // build the model
    if( ! hyperParamPlan.hasGrid() ) {    //==paramPlan.size() ) {
        BayesParameter localBayesParam;
        if( hyperParamPlan.AutoHPar() ) {
            double avgsqunorm = m_modelType.Standardize() ? 1.0 : AvgSquNorm(stats)/drs.dim(); //stats.AvgSquNorm();
            double hpar = 
                normal==hyperParamPlan.PriorType() ? 1.0/avgsqunorm
                : /*Laplace*/ sqrt(avgsqunorm*2.0);
            localBayesParam = BayesParameter( hyperParamPlan.PriorType(), hpar, hyperParamPlan.skew() );
            Log(5)<<"\nAverage square norm (no const term) "<<avgsqunorm<<" Hyperparameter "<<hpar;
        }
        else if( hyperParamPlan.fixed() )
            localBayesParam = hyperParamPlan.bp();
        else throw logic_error("Inconsistent 'HyperParamPlan' setting");

        drs.rewind();
        
	// m_beta = ParamMatrixSquare( drs.dim(), drs.c(),  0.0);  // v4.04

	if( ModelType::ZO==m_modelType.Optimizer() ) {
            InvData<InvTripletItr> invData( drs );
            ZOLR<InvTripletItr>( drs.dim(), drs.c(), invData,
                localBayesParam, priorMean, priorScale, m_modelType, fixedParams, m_beta );
        }
#ifdef QUASI_NEWTON
        else if( ModelType::QN_2coord==m_modelType.Optimizer() || ModelType::QN_smooth==m_modelType.Optimizer() )
            QN( drs, localBayesParam, priorMean, priorScale, m_modelType, fixedParams, m_beta );
#endif //QUASI_NEWTON
        else 
            throw logic_error("Unsupported optimizer");

        m_bayesParam = localBayesParam;
    }
    else {    //model averaging/selection

#define CV_   //tr3 is the alternative
#if defined CV_
        unsigned planSize = hyperParamPlan.plan().size();
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
            unsigned x = unsigned( rand()*(nfolds-irun)/randmax );
            unsigned ifold, y=0;
            for( ifold=0; ifold<nfolds; ifold++ )
                if( !foldsAlreadyRun[ifold] ) {
                    if( y==x ) break;
                    else y++;
                }
            foldsAlreadyRun[ifold] = true;
            Log(5)<<"\nCross-validation "<<nfolds<<"-fold; Run "<<irun+1<<" out of "<<nruns<<", fold "<<ifold;

            //training
            SamplingRowSet cvtrain( drs, rndind, ifold, ifold+1, false ); //int(randmax*ifold/nfolds), int(randmax*(ifold+1)/nfolds)
            Log(5)<<"\ncv training sample "<<cvtrain.n()<<" rows";
            SamplingRowSet cvtest( drs, rndind, ifold, ifold+1, true ); //int(randmax*ifold/nfolds), int(randmax*(ifold+1)/nfolds)
            Log(5)<<"\ncv test sample "<<cvtest.n()<<" rows";
            /*dbg
            cout<<"\nTrain ";
            while( cvtrain.next() ) cout<<" "<<cvtrain.currRowName();
            cvtrain.rewind();
            cout<<"\nTest ";
            while( cvtest.next() ) cout<<" "<<cvtest.currRowName();
            cvtest.rewind();
            dbg*/

            HyperParLoop hyperParLoop( cvtrain, cvtest, //invData,
                m_modelType, hyperParamPlan, //bayesParameter, hpplan, 
                priorMean, priorScale, fixedParams );

            Log(5)<<"\nCross-validation test log-likelihood: ";
            for( unsigned iparam=0; iparam<planSize; iparam++ ) {
                double eval = hyperParLoop.LogLikelihoods()[iparam];
                Log(5)<<eval<<" ";
                //arith ave of loglikeli, i.e. geometric mean of likelihoods
                lglkliMean[iparam] = lglkliMean[iparam]*irun/(irun+1) + eval/(irun+1);
                lglkliMeanSqu[iparam] = lglkliMeanSqu[iparam]*irun/(irun+1) + eval*eval/(irun+1);
            }

        }//cv loop

        vector<double> lglkliStdErr; //( hpplan.size(), 0.0 );
        for(unsigned i=0; i<planSize; i++ )
            lglkliStdErr.push_back( sqrt( lglkliMeanSqu[i] - lglkliMean[i]*lglkliMean[i] )//stddev
                    / sqrt(double(nruns)) ); // 1.05     Jun 20, 05  bug with stderr denom calc

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
        if( ModelType::ZO==m_modelType.Optimizer() ) {
            InvData<InvTripletItr> invData( drs );
            ZOLR<InvTripletItr>( drs.dim(), drs.c(), invData, 
                m_bayesParam, priorMean, priorScale, m_modelType, fixedParams, m_beta );
        }
#ifdef QUASI_NEWTON
        else if( ModelType::QN_2coord==m_modelType.Optimizer() || ModelType::QN_smooth==m_modelType.Optimizer() )
            QN( drs, m_bayesParam, priorMean, priorScale, m_modelType, fixedParams, m_beta );
#endif //QUASI_NEWTON
        else 
            throw logic_error("Unsupported optimizer");

#endif // CV_ | tr3_ | EM_

    }//model averaging/selection

}

static unsigned countequal( const IParamMatrix& beta, const IParamMatrix& priorMean ) {
    unsigned nz=0;
    for( unsigned j=0; j<beta.D(); j++ )
    for( unsigned k=0; k<beta.C(); k++ )
        if( beta(j,k) == priorMean(j,k) ) nz++;
    return nz;
}

static void displayModel( ostream& o, const IParamMatrix& beta, const INameResolver& datasource )
{
    o<<std::endl<<std::setw(20)<<"Feature\\Class";
    for( unsigned k=0; k<beta.C(); k++ )
        o<<std::setw(20)<<k;
    for( unsigned j=0; j<beta.D(); j++ ) {
        o<<std::endl<<std::setw(20)<<datasource.colName(j);
        for( unsigned k=0; k<beta.C(); k++ )
            o<<std::setw(20)<<beta(j,k);
    }
}

class Hessian {
    unsigned c, d;
    vector< vector<double> > fim;
public:
    Hessian( unsigned c_, unsigned d_ )
        : c(c_), d(d_), 
        fim( c*d, vector<double>( c*d, 0.0 ) )
        //fim( vector<double>(c*d*(c*d+1)/2, 0.0) )
    {}
    void update( const vector<double>& estprob, const SparseVector& xsparse ) {
        //XXT
        vector< vector<double> > XXT( d, vector<double>( d, 0.0 ) );
        for( SparseVector::const_iterator ix1=xsparse.begin(); ix1!=xsparse.end(); ix1++ )
            for( SparseVector::const_iterator ix2=ix1; ix2!=xsparse.end(); ix2++ )
                XXT[ix1->first][ix2->first]
                = XXT[ix2->first][ix1->first] 
                = ix1->second*ix2->second;
        //diagPminusPPT
        vector< vector<double> > diagPminusPPT( c, vector<double>( c, 0.0 ) );
        for( unsigned k1=0; k1<c; k1++ ) {
            diagPminusPPT[k1][k1] = estprob[k1] - estprob[k1]*estprob[k1]; //diag elem
            for( unsigned k2=k1+1; k2<c; k2++ )
                diagPminusPPT[k1][k2] = diagPminusPPT[k2][k1] = - estprob[k1]*estprob[k2];
        }
        //Kronecker product XXT by diagPminusPPT
        for( unsigned j1=0; j1<d; j1++ )
            for( unsigned k1=0; k1<c; k1++ )
                for( unsigned j2=0; j2<d; j2++ )
                    for( unsigned k2=0; k2<c; k2++ )
                        fim[j1*c+k1][j2*c+k2] += XXT[j1][j2] * diagPminusPPT[k1][k2];
    }
    void display( ostream & o ) const {
        for( unsigned i1=0; i1<c*d; i1++ ) {
            o<<endl;
            for( unsigned i2=0; i2<c*d; i2++ )
                o<<" "<<fim[i1][i2];        // /sqrt(fim[i1][i1]*fim[i2][i2]);
        }
    }
};

void LRModel::Train( const char* topic,
                       RowSetMem & trainData,
                       const HyperParamPlan& hyperParamPlan,
                       const class PriorTermsByTopic& priorTermsByTopic,
                       const class DesignParameter& designParameter,
                       const class ModelType& modelType,
                       WriteModel& modelFile,
                       std::ostream& result, ResultFormat resultFormat )
{
    const BayesParameter& bayesParameter = hyperParamPlan.bp();
    m_bTrained = false;
    m_topic = topic;
    m_modelType = modelType;
    m_featSelect.clear();
    for( unsigned i=0; i<trainData.dim(); i++ )
        m_featSelect.push_back( trainData.colId(i) );

    if( designParameter.DesignType()==designPlain )
        m_pDesign= new PlainDesign( trainData );
    else if( designParameter.DesignType()==designInteractions ) {
        throw logic_error("Interactions Design not supported");
    }
    else
        throw logic_error("Undefined Design type");
    Log(3)<<"\nDesign: "<<m_pDesign->dim()<<" variables";

    DesignRowSet drs( m_pDesign, trainData );
    Stats stats( drs );
    drs.rewind();

    m_beta = ParamMatrixSquare( drs.dim(), drs.c() );

    // priors
    ParamMatrixSquare priorMode( m_pDesign->dim(), drs.c() );
    ParamMatrixSquare priorScale( m_pDesign->dim(), drs.c(), 1.0 );

    // individual priors
    const IndPriors& indPriors = priorTermsByTopic.GetForTopic(string(topic));
        //!HACK: flat intercept desigh assumed==>
    for( unsigned k=0; k<trainData.c(); k++ ) 
      for( size_t j=0; j<trainData.dim() /*skip the 'unit' column added above*/; j++ ) 
        if( indPriors.HasClassIndPrior( trainData.classId(k), trainData.colId(j) ) )
        {
            priorMode(j,k) = indPriors.ClassPriorMode( trainData.classId(k), trainData.colId(j) );
            priorScale(j,k) = sqrt( indPriors.ClassPriorVar( trainData.classId(k), trainData.colId(j) ) );
            Log(6)<<"\nIndPrior: class "
                <<trainData.classId(k)<<" feature "<<trainData.colId(j)<<" mode "<<priorMode(j,k)<<" stddev "<<priorScale(j,k);
        }
    for( unsigned k=0; k<trainData.c(); k++ ) 
        if( indPriors.HasClassIndPrior( trainData.classId(k), 0 ) ) //intercepts
        {
            unsigned intercept = m_pDesign->interceptColNum();
            priorMode(intercept,k) = indPriors.ClassPriorMode( trainData.classId(k), 0 );
            priorScale(intercept,k) = sqrt( indPriors.ClassPriorVar( trainData.classId(k), 0 ) );
            Log(6)<<"\nIndPrior for intercept: class "<<trainData.classId(k)<<" mode "<<priorMode(intercept,k)<<" stddev "<<priorScale(intercept,k);
        }

    //FixedParams
    const vector< vector<bool> > dummy;
    FixedParams fixedParams( 
        modelType.AllZeroClassMode() ? trainData.allzeroes() : dummy,
        modelType.ReferenceClass(), 
//        drs.c()-1 //ref class is the last if any; fixes bug for ver2.02
	modelType.ReferenceClassId()
        );
    Log(3)<<"\nTotal parameters: "<<drs.dim()*( modelType.ReferenceClass() ? drs.c()-1 : drs.c() )<<"\t Of them fixed: "<<fixedParams.count();

    /*/HACK - ext score
    Log(1)<<"\nExternal score prior var coefficient: "<<10; //1.0/25;
    priorScale[ pDesign->dim()-2 ]  *= 10;   ///= 25;*/

    IRowSet* p_workingRs = &drs;
    IRowSet* p_stdRS=0;
    ParamMatrixSquare origPriorMean;//( m_pDesign->dim(), drs.c() );
    if( modelType.Standardize() ) {
        Vector means = stats.Means();
        Vector stddevs = stats.Stddevs();
        // !!! intercept - aware !
        means[ means.size()-1 ]  = 0;
        stddevs[ stddevs.size()-1 ]  = 1;
        p_stdRS = new StandardizedRowSet( drs, means, stddevs );
        p_workingRs = p_stdRS;
        origPriorMean = priorMode;
        for( unsigned k=0; k<priorMode.C(); k++ )
        for( unsigned j=0; j<priorMode.D(); j++ )
            if( 0<stddevs[j] )
                priorMode(j,k) *= stddevs[j];
    }

    m_beta = priorMode;   // v4.04  

    //this sets m_beta and m_bayesParam
    tuneModel( *p_workingRs, //bayesParameter, 
        hyperParamPlan,
        priorMode, priorScale, fixedParams, stats );

    if( modelType.Standardize() ) {
        delete p_stdRS;
        priorMode = origPriorMean;
        //recalc beta's
        double interceptAdjust = 0;
        for( unsigned k=0; k<m_beta.C(); k++ ) {
            for( unsigned j=0; j<m_beta.D()-1 /*!intercept*/; j++ )  {
                if( 0!=stats.Stddevs()[j] )
                    m_beta(j,k) /= stats.Stddevs()[j];
                interceptAdjust -= m_beta(j,k) * stats.Means()[j];
            }
            m_beta( m_beta.D()-1, k )  += interceptAdjust;
        }
    }

    //Beta components dropped finally
    unsigned bdropped = countequal( priorMode, m_beta );
    Log(3)<<"\nBeta components dropped finally: "<<bdropped
	  <<" Left: "<< m_beta.C()*m_beta.D()-bdropped;
    
//#define HESSIAN
    vector< vector<double> > resubstScore( drs.n(), vector<double>(drs.c(),0.0) );
    vector<unsigned> y( drs.n() );
#ifdef HESSIAN
    Hessian H( drs.c(), drs.dim() );
#endif //HESSIAN
    for( unsigned i=0; drs.next(); i++ ) {
        for( unsigned k=0; k<drs.c(); k++ )
            resubstScore[i][k] = dotSparseByDense( drs.xsparse(), m_beta.classparam(k) );
        y[i] = drs.y();
#ifdef HESSIAN
        H.update( estprob(resubstScore[i]), drs.xsparse() );
#endif //HESSIAN
    }
#ifdef HESSIAN
    Log(3)<<std::endl<<"Hessian computed, Time "<<Log.time()<<std::endl<<"Hessian:";
    H.display( Log(3) );
#endif //HESSIAN

    // report model
    Log(10)<<std::endl<<"Beta:";
    displayModel( Log(10), m_beta, drs );

    // resubstitution - evaluate
    vector<unsigned> resubst( drs.n() );
    for( unsigned i=0; i<drs.n(); i++ )
        resubst[i] = argmax( resubstScore[i] );
    Log(3)<<"\n\n---Resubstitution results---";
    Log(3)<<"\nConfusion Table:";   //TODO? make it Log(6)
    makeConfusionTable( Log(3), drs, y, resubst );

    double trainLogLikeli = LogLikelihood( drs, modelType, m_beta );
    Log(3)<<"\nTraining set loglikelihood "<<setprecision(12)<<trainLogLikeli<<" Average "<<trainLogLikeli/drs.n()<<setprecision(6);
    double logPrior = LogPrior( m_bayesParam, priorMode, priorScale, m_beta );
    Log(3)<<"\nLog prior (penalty) "<<logPrior<<" Log posterior "<<trainLogLikeli+logPrior;

    makeCT2by2( Log(3), drs, y, resubstScore, resubst );

    if( Log.level()>=12 ) {
        Log()<<"Resubstitution Scores";
        for( unsigned i=0; i<drs.n(); i++ )
            Log()<<endl<<resubstScore[i]<<":"<<resubst[i];
    }

    m_bTrained = true;
    modelFile.setRefClassInfo(modelType.ReferenceClass(), modelType.ReferenceClassId()); // v4.03 SL
    modelFile.WriteParams(strModelname, m_modelType, designParameter, m_pDesign, m_featSelect, m_beta, drs );

    // write results file
    drs.rewind();
    for( unsigned r=0; drs.next(); r++ ) {
        result << drs.classId( y[r] ) //label
            <<" "<< estprob(resubstScore[r]) //p_hat
            <<" "<< drs.classId( argmax(resubstScore[r]) ) //y_hat
	        <<endl;
    }
}

double LogLikelihood(
                IRowSet & rowset, // HACK!!! DesignRowSet needed: colId's are 0, 1, 2... 
                const class ModelType& modelType,
                const IParamMatrix& beta
            )
{
    rowset.rewind();
    double logLhood = 0;
    double i;

    // fit term
    for( i=0; rowset.next(); i++ ) {
        vector<double> scores;
        for( unsigned k=0; k<rowset.c(); k++ )
            scores.push_back( dotSparseByDense( rowset.xsparse(), beta.classparam(k) ) );
        logLhood = logLhood*(i/(i+1)) + PointLogLikelihood( scores, rowset.y(), modelType )/(i+1);
    }

    return logLhood*i;
}


double LogPrior(
                const BayesParameter& bayesParameter,
                const IParamMatrix& priorMean,
                const IParamMatrix& priorScale,
                const IParamMatrix& beta
            )
{
    double penaltyLogLhood = 0;
    for( unsigned j=0; j<beta.D(); j++ ) 
    for( unsigned k=0; k<beta.C(); k++ ) 
    {
        double betaLogLhood;
        if( numeric_limits<double>::infinity()==priorScale(j,k) )
            betaLogLhood = 0;
        else if( normal==bayesParameter.PriorType() ) {
            double sigma = priorScale(j,k)*sqrt(bayesParameter.var());
            double u = (beta(j,k)-priorMean(j,k)) / sigma;
            betaLogLhood = -log(2*M_PI)/2 - log(sigma) - u*u/2;
        }
        else if( Laplace==bayesParameter.PriorType() ){
            //Laplace distribution density: (1/(2*b)) * exp(-fabs(x)/b)
            //log density: -log(2)-log(b)-fabs(x)/b
            //c.d.f. (1/2)*( 1 + (x>0 ? 1 : -1)*(1-exp(-fabs(x)/b) )
            //scale: sqrt(2)*b, thus b=scale/sqrt(2)
            double inverse_b = bayesParameter.gamma() / priorScale(j,k);
            betaLogLhood = - log(2.0) + log(inverse_b) - fabs( beta(j,k)-priorMean(j,k) ) * inverse_b;
        }
        penaltyLogLhood += betaLogLhood;
        //Log(8)<<"\nbeta/lhood/log "<<beta[i]<<" "<<exp(betaLogLhood)<<" "<<betaLogLhood;
    }

    return penaltyLogLhood;
}


//  cout<<"\n! "<<__LINE__;cout.flush();



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
