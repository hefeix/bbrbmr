/*
 *  quasi-Newton optimization wrapper
 */
#define  _USE_MATH_DEFINES
#include <math.h>
//#include <algorithm>
//#include <limits>
//#include <iomanip>

#include "logging.h"
//#include "Matrix.h"
#include "BayesParamManager.h"
//#include "PriorTerms.h"
#include "Design.h"
#include "poly.h"
#include "ModelTypeParam.h"

using namespace std;

#ifdef _MSC_VER
#define finite(x) (_finite(x))
#endif

//prototyped from blmvm-1.0\example-c\combustionv1.c
extern "C" {
#include "blmvmabc.h"
}
    //int BLMVMFunctionGradient(void*,double[],double[],int,double*);
    //int BLMVMConverge(void*,double[],int,double,int*);
    //extern int BLMVMSolve(void*,double[],double[],double[],int,int);

typedef struct {
    double f,fmin;             /* Current function value, stop is less than     			     */
    double pgnorm,pgtol;       /* Current residual norm, stop if less than      			     */
    int iter, maxiter;         /* Current number of iterations, stop if greater than                     */
    int fgeval, maxfgeval;     /* Current number of objective/gradient evaluations, stop ig greater than */
    int whystop;               /* Reason why we stopped BLMVM                                            */
} CConverge;

class Context {     /* This user-defined context contains application specific information */
    IRowSet& data;
    const BayesParameter& bayesParam;
    const IParamMatrix & priorMean;
    const IParamMatrix & priorScale;
    const ModelType& modelType;
    const class FixedParams& fixedParams;
    unsigned c, d, nparams, nPseudoParams;
    bool dupcoord;
    vector<double> penalty;
    ParamMatrixSquare & w; //fixed terms never change
    ParamMatrixSquare grad; //only to avoid reallocation
    double esmooth; //constant for smoothing L1
    double *px, *pxl, *pxu;
    int history;
public:
    CConverge cc;
    Context(
        IRowSet& data_,
        const BayesParameter& bayesParam_,
        const IParamMatrix & priorMean_,
        const IParamMatrix & priorScale_,
        const ModelType& modelType_,
        const FixedParams& fixedParams_,
        ParamMatrixSquare & w );
    ~Context() {
        free(px);
        free(pxl);
        free(pxu);
    }

    int FuncGrad( double *x, double *g, int nparams, double *f );
    int Converge( double *residual, int nparams, double stepsize, int *whystop );

    unsigned NParams() const { return  nPseudoParams; }
    double* Low() {return pxl;} 
    double* X() {return px;} 
    double* Up() {return pxu;} 
    int History() {return history;} 
    double Esmooth() { return esmooth; }
    double set_esmooth( double e ) { return esmooth=e; }

    void square2flat( const IParamMatrix & w, double* x ) {
        unsigned i=0;
        for( unsigned j=0; j<w.D(); j++ )
            for( unsigned k=0; k<w.C(); k++ )
                if( ! fixedParams(j,k) )
                    x[i++] = w(j,k);
    }
    void flat2square( double* x, ParamMatrixSquare & w ) {
        unsigned i=0;
        for( unsigned j=0; j<w.D(); j++ )
            for( unsigned k=0; k<w.C(); k++ )
                if( ! fixedParams(j,k) )
                    w(j,k) = x[i++];
    }
};
Context:: Context(
        IRowSet& data_,
        const BayesParameter& bayesParam_,
        const IParamMatrix & priorMean_,
        const IParamMatrix & priorScale_,
        const ModelType& modelType_,
        const FixedParams& fixedParams_,
        ParamMatrixSquare & w_ ) //, CConverge& cc_ )
        : data(data_), bayesParam(bayesParam_), 
        priorMean(priorMean_), priorScale(priorScale_), modelType(modelType_), fixedParams(fixedParams_),
        w(w_),
        d( data.dim() ), c( data.c() ),
        grad( d, c, 0.0 ),
        px(0), pxl(0), pxu(0)
{
    if( data.dim()==priorMean.D() && data.dim()==priorScale.D() ) ;//ok
    else    throw DimensionConflict(__FILE__,__LINE__);

    dupcoord = ( Laplace==bayesParam.PriorType() && ModelType::QN_2coord==modelType.Optimizer() );

    nparams = c*d - fixedParams.count();
    nPseudoParams = dupcoord ? nparams : nparams*2;

    /* initial vector, lower bound, upperbound */
    px=(double*)malloc(nPseudoParams*sizeof(double)); 
    pxl=(double*)malloc(nPseudoParams*sizeof(double));
    pxu=(double*)malloc(nPseudoParams*sizeof(double));

    /* set bounds: no bounds */
    for( unsigned i=0; i<nPseudoParams; i++ ) {
        pxl[i] = dupcoord ? 0 : -numeric_limits<double>::max();
        pxu[i] = numeric_limits<double>::max();
    }

    /* initial point */
    square2flat( w, px );
    if(dupcoord)
        for( unsigned i=0; i<nparams; i++ )
            if( px[i]>=0 ) px[nparams+i]=0;
            else{ px[nparams+i]=-px[i]; px[i]=0; }

    // init convergence data
    cc.fmin = 0; //-numeric_limits<double>::max(); -1e30;
    cc.pgtol = modelType.ThrConverge(); //.001;
    cc.iter = 0;
    cc.maxiter = numeric_limits<int>::max(); //100;
    cc.fgeval = 0;
    cc.maxfgeval = numeric_limits<int>::max(); //10000; 

    history = 8;

    //convert prior scale into penalty parameter
    Log(3)<<std::endl<<"No penalty for the intercept!";
    for( unsigned j=0; j<d; j++ )
    for( unsigned k=0; k<c; k++ )
    {
        if( fixedParams(j,k) )  continue;
        double p;
        if( normal==bayesParam.PriorType() )
            p = 1.0 /( bayesParam.var()*priorScale(j,k)*priorScale(j,k) );
        else if( Laplace==bayesParam.PriorType() )
            p = bayesParam.gamma() / priorScale(j,k);
        else
            throw runtime_error("Only normal or Laplace priors allowed");
        if( j==d-1 ) p = 0.0; //HACK - no penalty for intercept
        penalty.push_back( p );
        Log(11)<<"\n k/j/priorScale(j,k)/priorMean(j,k)/penalty(j,k)/"<<k<<" "<<j<<" "<<priorScale(j,k)<<" "<<priorMean(j,k)<<" "<<p;
    }
    Log(3)<<std::endl<<"No penalty for the intercept!";
    if( penalty.size() != nparams )
        throw logic_error("Penalty array size exception");
}

void /*returns model coeffs thru the last arg*/
QN(
                IRowSet& data,
                const BayesParameter& bayesParam, //input
                const IParamMatrix & priorMean,  //input
                const IParamMatrix & priorScale,  //input
                const ModelType& modelType,  //input
                const class FixedParams& fixedParams,  //input
                ParamMatrixSquare & w  // i/0: init/result values of model coeffs, 
            )
{
    //throw logic_error("Quasi-Newton NOT IMPLEMENTED YET");
    if( Laplace!=bayesParam.PriorType()
        && normal!=bayesParam.PriorType() )
            throw runtime_error("Unsupported prior type; only normal or Laplace currently supported");

    if( ModelType::QN_smooth==modelType.Optimizer() );
    else if( ModelType::QN_2coord==modelType.Optimizer() );
    else 
        throw logic_error("Unsupported optimizer for quasi-Newton");

    /*w(1,1)=.005;dbg*/

    Context context(data, bayesParam, priorMean, priorScale, modelType, fixedParams, w );

    Log(3)<<endl<<"Starting BLMVM optimization, "
        <<( Laplace==bayesParam.PriorType() ? 
        ( ModelType::QN_smooth==modelType.Optimizer() ? "Smoothed, " : "Double coordinates, ") //Crude L1
        : "" )
        <<"Time "<<Log.time();
    Log(3)<<endl<<"Line search 2";
    if( Laplace==bayesParam.PriorType() && ModelType::QN_smooth==modelType.Optimizer() )
        Log(3)<<"\nesmooth "<<context.set_esmooth( 1e-25 ); //1e+200*numeric_limits<double>::min() );

    /*for(unsigned i=0;i<nparams;i++) Log(8)<<"\nx "<<x[i];dbg*/
    int info=BLMVMSolve( (void*)&context,
        context.Low(), context.X(), context.Up(), context.NParams(), context.History() );

    // opt point (context has modified 'w' thru its reference)
    Log(9)<<"\nFinal Beta "<<w;
    string diagnos;
    switch(context.cc.whystop) {
        case 1: diagnos = "Solution found"; break;
        case 2: diagnos = "Max iterations reached"; break;
        case 3: diagnos = "Max  reached"; break;
        case 4: diagnos = "Min Objective reached"; break;
        default: diagnos = "Stopped due to numerical difficulties";
    }
    Log(3)<<endl<<"Optimization finished "<<context.cc.iter<<" iterations, "<<diagnos<<", Time "<<Log.time();
    Log(4)<<endl<<"Function/Gradient evaluations "<<context.cc.fgeval;
#ifdef NEVERDEF /////
    // init convergence data
    CConverge cc1;
    cc1.fmin = -numeric_limits<double>::max();  //-1e30;
    cc1.pgtol = 0.001;
    cc1.iter = 0;
    cc1.maxiter = numeric_limits<int>::max(); //100;
    cc1.fgeval = 0;
    cc1.maxfgeval = numeric_limits<int>::max(); //10000; 

    w(1,1)=.005;/*dbg*/

    Context context1(data, bayesParam, priorMean, priorScale, modelType, fixedParams, w, cc1 );
    /* initial point */
    context1.square2flat( w, x );

    Log(3)<<endl<<"Starting BLMVM optimization, "
        <<( Laplace==bayesParam.PriorType() ? 
        ( ModelType::QN_smooth==modelType.Optimizer() ? "Smoothed, " : "Double coordinates, ") //Crude L1
        : "" )
        <<"Time "<<Log.time();
    if( Laplace==bayesParam.PriorType() && ModelType::QN_smooth==modelType.Optimizer() )
        Log(3)<<"\nesmooth "<<context.set_esmooth( 1e-50 ); //1e+200*numeric_limits<double>::min() );

    /*for(unsigned i=0;i<nparams;i++) Log(8)<<"\nx "<<x[i];dbg*/
    info=BLMVMSolve( (void*)&context1, xl, x, xu, nparams, history );

    // opt point (context has modified 'w' thru its reference)
    Log(8)<<"\nFinal Beta "<<w;
    switch(context1.cc.whystop) {
        case 1: diagnos = "Solution found"; break;
        case 2: diagnos = "Max iterations reached"; break;
        case 3: diagnos = "Max Function/Gradient evaluations reached"; break;
        case 4: diagnos = "Min Objective reached"; break;
        default: diagnos = "Stopped due to numerical difficulties";
    }
    Log(3)<<endl<<"Optimization finished "<<context.cc.iter<<" iterations, "<<diagnos<<", Time "<<Log.time();
#endif ///////

    //printf("Optimal Objective value: %4.4e\n",cbctx.cc.f);
    //printf("Sensitivity to lower bounds: \n"); xl[k])
    //printf("Sensitivity to upper bounds: \n"); xu[k])

}

//comments from blmvm-1.0\example-c\combustionv1.c:
/* --------------------------------------------------------------------------
 *    BLMVMFunctionGradient -- User must implement this routine that evaluates 
 *    the objective function and its gradient at current point x 
 *
 *    Input:   ctx - pointer to user-defined context.
 *             x - current variables (array of length n).
 *             g - empty (array of length n)
 *             n - number of variables.
 *             f - address of current objective function value;
 *
 *    Output:  ctx - pointer to user-defined context.
 *             x - unchanged.
 *             g - the gradient of objective function f(x) evaluated at x.
 *             n - unchanged.
 *             f - objective function evaluated at x.
 *
 *             return:  0 for normal return and nonzero to abort mission.
 *                                                                         */
int BLMVMFunctionGradient(void *ctx,double *x,double *g,int n, double *f)
{
    Context* pCtx = (Context*)ctx;
    return pCtx->FuncGrad( x, g, n, f );
}
int Context:: FuncGrad( double *x, double *gFlat, int n, double *f )
{
    if( n != nPseudoParams )
        throw logic_error("Wrong parameter vector size returned from BLMVMFunctionGradient");

    //Log(8)<<"\nX "; for(unsigned i=0;i<nparams;i++) Log(8)<<" "<<x[i];
    {//flat2square( wFlat, w ); //fixed terms already there
    unsigned i=0;
    for( unsigned j=0; j<w.D(); j++ )
        for( unsigned k=0; k<w.C(); k++ )
            if( ! fixedParams(j,k) ) {
                if(dupcoord)  w(j,k) = x[i] - x[nparams+i];
                else  w(j,k) = x[i];
                i++;
            }
    }
    //Log(9)<<"\nBeta "<<w;

    for( unsigned j=0; j<d; j++ )
        for( unsigned k=0; k<c; k++ )
            grad( j, k ) = 0.0;

    //---loss term---
    double lossTerm = 0;
    data.rewind();
    while( data.next() ) {

        vector<double> linscores( c, 0.0 );
        try{ //pass one
        for( SparseVector::const_iterator ix=data.xsparse().begin(); ix!=data.xsparse().end(); ix++ )
            for( unsigned k=0; k<c; k++ )
                linscores[k] += w(ix->first,k) * ix->second;
        }catch(...){
            throw logic_error("sparse vector index beyond dense vector size"); }
        //Log(9)<<"\nlinscore "<<linscores;

        vector<double> invPhat( c );
        bool simpleNum = false;
        if( simpleNum ) {
            vector<double> expscores( c );
            double sumexp = 0.0;
            for( unsigned k=0; k<c; k++ )  sumexp +=( expscores[k] = exp( linscores[k] ) );
            if( 0==sumexp || !finite(sumexp) ) throw logic_error("Need better numerics");
            for( unsigned k=0; k<c; k++ )  invPhat[k] = sumexp / expscores[k];
            Log(9)<<"\nexpscore "<<expscores;
        }else{
            for( unsigned k=0; k<c; k++ )  {
                invPhat[k] = 1.0;
                for( unsigned kk=0; kk<c; kk++ )
                    if( kk!=k )
                        invPhat[k] += exp( linscores[kk] - linscores[k] );
                //p_hat[k] = 1.0/invPhat;
            }
        }
        lossTerm += log( invPhat[data.y()] );
        //Log(9)<<"\tinvPhat "<<invPhat; //Log(9)<<"\tp_hat "<<p_hat
        //Log(9)<<"\ty p lossTerm "<<data.y()<<" "<<1/invPhat[data.y()]<<" "<<log( invPhat[data.y()] );

        //pass two - gradient
        for( SparseVector::const_iterator ix=data.xsparse().begin(); ix!=data.xsparse().end(); ix++ )
            for( unsigned k=0; k<c; k++ )
                if( ! fixedParams(ix->first,k) ) {
                    grad(ix->first,k) -= ( k==data.y() ? ix->second : 0 ) - ix->second/invPhat[k];
                    //Log(8)<<"\n j x y k p_hat[k] d_g "<<ix->first<<" "<<ix->second<<" "<<data.y()<<" "<<p_hat[data.y()]<<" "<<( k==data.y() ? ix->second : 0 ) - ix->second*p_hat[k];
                }
    }
    //Log(9)<<"\nLoss-only Gradient "<<grad;

    square2flat( grad, gFlat );
    if( dupcoord )
        for( unsigned i=0; i<nparams; i++ )
            gFlat[i+nparams] = gFlat[i];

    double penaltyTerm = 0;
    {//---penalty term---
    unsigned i=0;
    for( unsigned j=0; j<w.D(); j++ )
        for( unsigned k=0; k<w.C(); k++ )
            if( ! fixedParams(j,k) ) {
                double s = x[i] - priorMean(j,k);
                if( normal==bayesParam.PriorType() ) {
                    penaltyTerm += s * s * penalty[i];
                    gFlat[i] += 2 * s * penalty[i];
                }
                else if( Laplace==bayesParam.PriorType() ) {
                    if( ModelType::QN_smooth==modelType.Optimizer() ) {
                        double p = sqrt( s*s + esmooth );
                        penaltyTerm += p * penalty[i];
                        gFlat[i] += s * penalty[i] / p;
                    }else{ //2-coord
                        if( ! dupcoord ) throw logic_error("dupcoord inconsistent");
                        penaltyTerm += fabs(s) * penalty[i];
                        gFlat[i] += penalty[i];
                        gFlat[nparams+i] -= penalty[i];
                    }
                    /*/crude L1 version
                    penaltyTerm += fabs(wFlat[i] - priorMean(j,k)) * penalty[i];
                    if( wFlat[i]>priorMean(j,k) ) gFlat[i] += penalty[i];
                    else if( wFlat[i]<priorMean(j,k) ) gFlat[i] -= penalty[i];
                    else { // at prior mean
                        if( gFlat[i]+penalty[i]<0 )         gFlat[i] += penalty[i]; //right deriv <0
                        else if( gFlat[i]-penalty[i]>0 )    gFlat[i] -= penalty[i]; //left deriv >0
                        else gFlat[i] = 0;  //at min
                    }*/
                }
                i++ ;
            }
    }
    *f = lossTerm + penaltyTerm;

    // update convergence data
    cc.fgeval++;
    cc.f = *f;

    Log(7)<<"\nFGeval: loss penalty f "<<lossTerm<<" "<<penaltyTerm<<" "<<*f;
    //Log(8)<<"\nGradient"; for(unsigned i=0;i<nparams;i++) Log(8)<<" "<<gFlat[i];

    /*/suggested by S.Benson:*/
    if( ! finite(*f) ) {
        *f = numeric_limits<double>::max();
        return 1;
    }
    return 0;
}

//comments from blmvm-1.0\example-c\combustionv1.c:
/* --------------------------------------------------------------------------
 *    BLMVMConverge -- User must implement this routine that tells the solver
 *    the current solution is sufficiently accurate and the solver should 
 *    terminate.
 *
 *    Input:   ctx - Pointer to user-defined structure 
 *             residual - residuals of each variable (array of length n) (=0 at solution).
 *             n - number of variables.
 *             stepsize - change is the solution (2 norm).
 *             whystop - .
 *
 *    Output:  ctx - Pointer to user-defined structure 
 *    residual - unchanged.
 *    whystop - flag set to zero if BLMVM should continue, and nonzero if the solver should stop.
 *    return: 0 for normal return and nonzero to abort mission.
 *                                                                            */

int BLMVMConverge( void* ctx, double *residual, int n, double stepsize, int *whystop )
{
    Context* pCtx = (Context*)ctx;
    return pCtx->Converge( residual, n, stepsize, whystop );
}
int Context::Converge( double *residual, int n, double stepsize, int *whystop )
{ 
    if( n != nPseudoParams )
        throw logic_error("Wrong parameter vector size returned from BLMVMConverge");

    {
        double pgnorm=0.0;
        for( unsigned i=0; i<n; i++ )
            pgnorm += residual[i]*residual[i];
        cc.pgnorm = sqrt(pgnorm);
    }
    Log(6)<<"\nIter: "<<cc.iter<<", F: "<<cc.f<<",  pgnorm: "<<cc.pgnorm<<", Time "<<Log.time();

    int finished;
    if     ( cc.pgnorm <= cc.pgtol        ) finished=1;
    else if( cc.iter   >= cc.maxiter      ) finished=2;
    else if( cc.fgeval >= cc.maxfgeval    ) finished=3;
    else if( cc.f      <= cc.fmin         ) finished=4;
    else if( cc.iter > 0 && stepsize <= 0 ) finished=5;
    else                                    finished=0;
 
    cc.iter++;
    cc.whystop = *whystop = finished;

    return 0;
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
