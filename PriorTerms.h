#ifndef BAYES_PRIOR_TERMS_
#define BAYES_PRIOR_TERMS_

#include <map>
#include <string>
#include <ostream>
#include <sstream>
#include <stdexcept>

#ifdef USE_LEMUR
# include "Index.hpp"
# include "Param.hpp"
#endif //USE_LEMUR

enum TIndPriorsMode { indPriorsModeAbs, indPriorsModeRel, indPriorsModeUndef };
struct TModeVarSkew {
    double mode,var,skew;
    TModeVarSkew( double mode_, double var_, double skew_ ) :  mode(mode_),var(var_),skew(skew_) {}
    TModeVarSkew() :  mode(0.0),var(1.0),skew(0.0) {}
};
typedef map< unsigned, TModeVarSkew > TIndPriorsMap;

class IndPriors {    //individual priors
    friend class PriorTermsByTopic;
    TIndPriorsMap m;
    map<int,TIndPriorsMap> m_byClass;
    TIndPriorsMode m_indPriorsMode;
    //helpers
    bool valid(const TModeVarSkew& p) const {return p.var>=0;}
    TModeVarSkew findByClass(int c, unsigned iVar) const {
        //look for class-specific first
        TModeVarSkew p(0,-1,0); //negative variance is impossible
        map<int,TIndPriorsMap>::const_iterator itr1 = m_byClass.find( c );
        if( itr1!=m_byClass.end() ) { //class found
            TIndPriorsMap::const_iterator itr2 = itr1->second.find( iVar );
            if( itr2!=itr1->second.end() )  //var found
                p = itr2->second;
        }
        if( !valid(p) ) { //not in class-specific - look in common
            TIndPriorsMap::const_iterator itr2 = m.find( iVar );
            if( itr2!=m.end() )  //var found
                p = itr2->second;
        }
        return p;
    }
public:
    bool Active() const { return m.size()>0; }
    TIndPriorsMode IndPriorsMode() const { return Active() ? m_indPriorsMode : indPriorsModeUndef; }
    //common
    bool HasIndPrior(unsigned iVar) const { return m.find(iVar)!=m.end(); }
    double PriorMode(unsigned iVar) const { 
        std::map< unsigned, TModeVarSkew >::const_iterator itr = m.find(iVar);
        if(itr==m.end()) throw logic_error("Erroneous request for an individual prior");
        return itr->second.mode;
    }
    double PriorVar(unsigned iVar) const { 
        std::map< unsigned, TModeVarSkew >::const_iterator itr = m.find(iVar);
        if(itr==m.end()) throw logic_error("Erroneous request for an individual prior");
        return itr->second.var;
    }
    double PriorSkew(unsigned iVar) const { 
        std::map< unsigned, TModeVarSkew >::const_iterator itr = m.find(iVar);
        if(itr==m.end()) throw logic_error("Erroneous request for an individual prior");
        return itr->second.skew;
    }
    //class-wise
    bool HasClassIndPrior(int c, unsigned iVar) const { 
        return findByClass(c,iVar).var >= 0; }
    double ClassPriorMode(int c, unsigned iVar) const {
        TModeVarSkew p = findByClass(c,iVar);
        if( p.var<0 ) throw logic_error("Erroneous request for an individual prior");
        return p.mode;
    }
    double ClassPriorVar(int c, unsigned iVar) const { 
        TModeVarSkew p = findByClass(c,iVar);
        if( p.var<0 ) throw logic_error("Erroneous request for an individual prior");
        return p.var;
    }
    double ClassPriorSkew(int c, unsigned iVar) const { 
        TModeVarSkew p = findByClass(c,iVar);
        if( p.var<0 ) throw logic_error("Erroneous request for an individual prior");
        return p.skew;
    }
    //ctor
    IndPriors(TIndPriorsMode indPriorsMode=indPriorsModeUndef) : m_indPriorsMode(m_indPriorsMode) {};
};

class PriorTermsByTopic {
    std::string m_indPriorsFile;
    TIndPriorsMode m_indPriorsMode;
    std::map< string, IndPriors > m_indPriorsByTopic;
    bool m_multiTopic;
    IndPriors m_emptyPriors; //to return for unknown topic
public:
    std::string IndPriorsFile() const { return m_indPriorsFile; }
    TIndPriorsMode IndPriorsMode() const { return m_indPriorsFile.size()>0 ? m_indPriorsMode : indPriorsModeUndef; }
    const IndPriors& GetForTopic( const string& topic ="") const  {
        map< string, IndPriors >::const_iterator itr;
        if( m_multiTopic ) itr = m_indPriorsByTopic.find( topic );
        else  itr = m_indPriorsByTopic.begin();
        if( itr==m_indPriorsByTopic.end() ) //topic unknown
            return m_emptyPriors;
        else
            return itr->second;
    }
#ifdef USE_LEMUR
    void get() {
        m_multiTopic = true;
        m_indPriorsFile = ParamGetString( "indPriors.file","");
        std::string strPriorsMode = ParamGetString( "indPriors.mode","");
        m_indPriorsMode = strPriorsMode=="abs" ? indPriorsModeAbs : 
                    strPriorsMode=="rel" ? indPriorsModeRel :
                    indPriorsModeUndef;
        if( m_indPriorsFile.size()>0 && m_indPriorsMode==indPriorsModeUndef )
            throw std::runtime_error("Individual priors mode undefined. Specify 'abs' or 'rel'.");

        if( m_indPriorsFile.size()>0 )
            initIndPriors();
    }
#endif //USE_LEMUR
    //ctors
    PriorTermsByTopic() : m_indPriorsMode(indPriorsModeUndef) {}
    PriorTermsByTopic( std::string indPriorsFile, TIndPriorsMode indPriorsMode, bool multiTopic=false )
        : m_indPriorsFile(indPriorsFile), m_indPriorsMode(indPriorsMode), m_multiTopic(multiTopic)
    {
        if( m_indPriorsFile.size()>0 && m_indPriorsMode==indPriorsModeUndef )
            throw std::runtime_error("Individual priors mode undefined. Specify 'abs' or 'rel'.");
        initIndPriors();
    }
private:
    void initIndPriors();
};

inline std::ostream& operator<<( std::ostream& o, const PriorTermsByTopic& pt ) {
    if( pt.IndPriorsMode() != indPriorsModeUndef )
        o << "Individual Priors file: "<< pt.IndPriorsFile()
            <<"\t Mode: "<<( pt.IndPriorsMode()==indPriorsModeAbs ? "Absolute" : "Relative" );
    return o;
}

#endif //BAYES_PRIOR_TERMS_

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
