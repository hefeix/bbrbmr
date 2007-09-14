// SL  Mar 07   ver3.03  allow -R classid;


#ifndef _MODEL_TYPE_PARAMETER_HPP
#define _MODEL_TYPE_PARAMETER_HPP

#include <ostream>
#include <string>
#include <stdexcept>
#ifdef USE_LEMUR
# include "Param.hpp"
#endif //USE_LEMUR

#ifdef POLYTOMOUS
    #define convergeDefault 0.001
    #define iterDefault 10000
#else
    #define convergeDefault 0.0005
    #define iterDefault 1000
#endif

class ModelType {
    friend class ReadModel;
public:
    //types
    enum link { probit=0, logistic=1, SVM=2, linkUndef };
    enum optimizer { EM=0, ZO=1, SVMlight=2,
#ifdef QUASI_NEWTON
        QN_smooth=3, QN_2coord=4, 
#endif
        optUndef };
    enum thrtune { thrNo=0, thrSumErr=1, thrT11U=2, thrF1=3, thrBER=4, thrT13U=5,  thrUndef };
    //access
    enum link Link() const { return m_link; }
    enum optimizer Optimizer() const { return m_opt; }
    enum thrtune TuneThreshold() const { return m_thr; }
    double ProbThreshold() const { return  m_probThr; }
    bool TuneEst() const { return m_tuneEst; }
    bool Standardize() const { return m_standardize; }
    string StringParam() const { return strParam; }
    double ThrConverge() const { return m_thrConverge; }
    bool HighAccuracy() const { return m_highAccuracy; } //bbr v2.04
    unsigned IterLimit() const { return m_iterLimit; } //bbr v2.04
#ifdef POLYTOMOUS
    bool ReferenceClass() const { return m_referenceClass; }
    int ReferenceClassId() const { return m_referenceClassId; }  // SL v3.03
    bool AllZeroClassMode() const { return m_allZeroClassMode; }
#endif
    //ctors
    ModelType() : m_link(logistic), m_opt(ZO), m_thr(thrSumErr), m_tuneEst(false), m_standardize(false),
         m_thrConverge(convergeDefault), m_iterLimit(iterDefault), m_highAccuracy(false)
    {}
    ModelType( enum link link_, enum optimizer opt_, enum thrtune thr_, 
        bool standardize_, 
        double thrConverge, unsigned iterLimit, bool highAccuracy,
        string strParam_="" )
        : m_link(link_), m_opt(opt_), 
        m_thr(thr_), m_probThr(0.5),
        m_tuneEst(false), m_standardize(standardize_),
        m_thrConverge(thrConverge), m_iterLimit(iterLimit), m_highAccuracy(highAccuracy),
        strParam(strParam_)  {}
    ModelType( enum link link_, enum optimizer opt_, double probThr_, 
        bool standardize_,
        double thrConverge, unsigned iterLimit, bool highAccuracy, 
#ifdef POLYTOMOUS
        bool referenceClass_,
	       int referenceClassId_,
        bool allZeroClassMode_,
#endif
        string strParam_="" )
        : m_link(link_), m_opt(opt_), 
        m_thr(thrNo), m_probThr(probThr_),
        m_tuneEst(false), m_standardize(standardize_),
        m_thrConverge(thrConverge), m_iterLimit(iterLimit), m_highAccuracy(highAccuracy),
#ifdef POLYTOMOUS
        m_referenceClass(referenceClass_),
	m_referenceClassId(referenceClassId_),
        m_allZeroClassMode(allZeroClassMode_),
#endif
        strParam(strParam_)  {}
    //input
#ifdef USE_LEMUR
    void get()
    {
        m_link = (enum link)ParamGetInt("LinkFunction", logistic);
        if( m_link>=linkUndef )
            throw runtime_error("Undefined link; Use 0 for probit, 1 for logistic.");

        m_opt = (enum optimizer)ParamGetInt("Optimizer", m_link==logistic ? ZO : EM);
        if( m_opt>=optUndef )
            throw runtime_error("Undefined optimizer; Use 0 for EM, 1 for ZO.");
        if( m_opt==EM && m_link!=probit )
            throw runtime_error("EM only supported for probit link.");
        if( m_opt==ZO && m_link!=logistic )
            throw runtime_error("ZO only supported for logistic link.");
        if( m_opt==SVMlight && m_link!=SVM )
            throw runtime_error(" should be specified for both link and optimizer.");

        m_standardize =( 1==ParamGetInt("Standardize",0 ) );

        m_thrConverge = ParamGetDouble( "eps", convergeDefault );
        m_iterLimit = ParamGetInt( "iter", iterDefault );
        m_highAccuracy =( 1==ParamGetInt("accurate",0 ) );

        strParam =  ParamGetString("SVMParam","");

        m_thr = (enum thrtune)ParamGetInt("TuneThreshold", thrNo);
        if( m_thr>=thrUndef )
            throw runtime_error("Undefined threshold tuning rule.");

        m_tuneEst = 0!=ParamGetInt("TuneEst", false);
    }
#endif //USE_LEMUR
private:
    enum link m_link;
    enum optimizer m_opt;
    enum thrtune m_thr;
    double m_probThr;
    /** If m_tuneEst==true, tuning is done by using estimated
       probabilities on the (large, unlabeled) Training Population,
       rather than by using actual labels on the (small, labeled)
       training set */
    bool m_tuneEst; 
    bool m_standardize;
    string strParam;
    double m_thrConverge;
    bool m_highAccuracy;  //bbr v2.04
    unsigned m_iterLimit; //bbr v2.04
#ifdef POLYTOMOUS
    bool m_referenceClass;
    int m_referenceClassId;
    bool m_allZeroClassMode;
#endif
};

inline std::ostream& operator<<( std::ostream& o, const ModelType& mt ) {
    o << "Model - link function: "<<( 
        mt.Link()==ModelType::probit ? "Probit "
        : mt.Link()==ModelType::logistic ? "Logistic " 
        : mt.Link()==ModelType::SVM ? "SVMlight " 
        : "<undefined> " )
     << "\tOptimizer: "<<( 
        mt.Optimizer()==ModelType::EM ? "EM"
        : mt.Optimizer()==ModelType::ZO ? "ZO" 
        : mt.Optimizer()==ModelType::SVMlight ? "SVMlight" 
#ifdef QUASI_NEWTON
        : mt.Optimizer()==ModelType::QN_smooth ? "quasi-Newton, smoothed penalty" 
        : mt.Optimizer()==ModelType::QN_2coord ? "quasi-Newton, double coordinate" 
#endif
        : "<undefined> " );
#ifndef POLYTOMOUS
    if( mt.TuneThreshold()==ModelType::thrNo )
        o<< "\tThreshold probability: "<<mt.ProbThreshold(); 
    else o<< "\tTune threshold: "<<( 
        mt.TuneThreshold()==ModelType::thrNo ? "No" //should never get here
        : mt.TuneThreshold()==ModelType::thrSumErr ? "Sum Error" 
        : mt.TuneThreshold()==ModelType::thrBER ? "Balanced error rate"
        : mt.TuneThreshold()==ModelType::thrT11U ? "T11U" 
        : mt.TuneThreshold()==ModelType::thrF1 ? "F1" 
	    : mt.TuneThreshold()==ModelType::thrT13U ? "T13U" 
        : "<undefined> " );
#endif
    o << (mt.TuneEst() ? ", on TP estimates" : "" )
     << "\nStandardize: "<<( mt.Standardize() ? "Yes" : "No" )
     << "\nConvergence threshold: "<<mt.ThrConverge()<<"\tIterations limit: "<<mt.IterLimit()
     <<( mt.HighAccuracy()? "\tHigh accuracy mode" : "" );
#ifdef POLYTOMOUS
    if( mt.ReferenceClass() )
        o<<"\nUsing Reference Class";
    if( mt.AllZeroClassMode() )
        o<<"\nAllZeroClassMode: Variable/class const zero gets zero beta";
#endif
    if( mt.StringParam().size()>0 )
        o<<"\nString parameter: '"<<mt.StringParam()<<"'";
    return o;
}

#endif //_MODEL_TYPE_PARAMETER_HPP


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
