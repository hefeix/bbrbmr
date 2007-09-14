#ifndef _TFIDF_PARAMETER_HPP
#define _TFIDF_PARAMETER_HPP

#include <exception>
#ifdef USE_LEMUR
# include "Param.hpp"
#endif //USE_LEMUR
#include "logging.h"

enum TFMethod  {RAWTF=0, LOGTF=1};
enum IDFMethod  {NOIDF=0, LOGIDF=1};

class TFIDFParameter {
    friend class ReadModel;
    TFMethod m_tf;
    IDFMethod m_idf;
    bool m_cosineNormalize;
    void set_tfMethod(TFMethod tf) { m_tf=tf; }
    void set_idfMethod(IDFMethod idf) { m_idf=idf; }
    void set_cosineNormalize(bool cn) { m_cosineNormalize=cn; }

public:
    TFIDFParameter() : m_tf(LOGTF), m_idf(LOGIDF), m_cosineNormalize(false) {} //setting defaults
    TFIDFParameter(TFMethod tf, IDFMethod idf, bool cn) 
        : m_tf(tf), m_idf(idf), m_cosineNormalize(cn) {}
#ifdef USE_LEMUR
    void get()
    {
        m_tf = (enum TFMethod)ParamGetInt("doc.tfMethod", m_tf);
        m_idf = (enum IDFMethod)ParamGetInt("doc.idfMethod", m_idf);
        m_cosineNormalize = ( 1==ParamGetInt("cosineNormalize", 0) );
    }
#endif //USE_LEMUR
    TFMethod tfMethod() const { return m_tf; }
    IDFMethod idfMethod() const { return m_idf; }
    bool cosineNormalize() const { return m_cosineNormalize; }
};

inline std::ostream& operator<<( std::ostream& o, const TFIDFParameter& p ) {
    o <<"TF parameter: "
        <<( p.tfMethod()==RAWTF ? "RAWTF" : p.tfMethod()==LOGTF ? "LogTF" : "undef" )
        <<"\t IDF parameter: "
        <<( p.idfMethod()==NOIDF ? "No IDF" : p.idfMethod()==LOGIDF ? "LogIDF" : "undef" )
        <<"\t Cosine normalization: "
        <<( p.cosineNormalize() ? "Yes" : "No" );
    return o;
}

#endif //_TFIDF_PARAMETER_HPP
