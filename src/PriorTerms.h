// SL v3.03  keep the features non-active in training examples but having non-zero modes in ind prior file.

#ifndef BAYES_PRIOR_TERMS_
#define BAYES_PRIOR_TERMS_

#include <map>
#include <string>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#ifndef _MSC_VER
#define  stricmp(a, b)   strcasecmp(a,b) 
#endif

#ifdef USE_LEMUR
# include "Index.hpp"
# include "Param.hpp"
#endif //USE_LEMUR

using namespace std;


enum TIndPriorsMode { indPriorsModeAbs, indPriorsModeRel, indPriorsModeUndef };
struct TModeVarSkew {
    double mode,var,skew;
    TModeVarSkew( double mode_, double var_, double skew_ ) :  mode(mode_),var(var_),skew(skew_) {}
    TModeVarSkew() :  mode(0.0),var(1.0),skew(0.0) {}
};
typedef map< unsigned, TModeVarSkew > TIndPriorsMap;



struct tri{
    int feat; 
    double val; 
    int cls; 

    tri( int feat_, double val_, int cls_) : feat(feat_), val(val_), cls(cls_) {};
    tri() : feat(-1),val(0),cls(0) {};
};

// v3.03
struct IndPriorNonZeroModeFeats {
    vector<tri> nonzeros;
    unsigned count;
    IndPriorNonZeroModeFeats() {
	count=0;
    }
};

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


    //SL v3.03
    void _getNonZeroModes(vector<int>& words, IndPriorNonZeroModeFeats& nonzeromodes ) const {
	vector<int>::iterator viter;

	map<int,TIndPriorsMap>::const_iterator iter;
	nonzeromodes.count = 0;
	for(iter=m_byClass.begin();iter!=m_byClass.end();iter++){
	    viter=words.begin();
            TIndPriorsMap::const_iterator itr2;
	    for(itr2 = iter->second.begin(); itr2!=iter->second.end(); itr2++) {
		if(itr2->second.mode!=0.0) {
		    vector<int>::iterator viter2 = find(viter,words.end(),itr2->first);
		    if(viter2!=words.end())
			viter=viter2;
		    else {
			//if(iter->first==-1){	
			    nonzeromodes.nonzeros.push_back(tri(itr2->first, itr2->second.mode, iter->first));
			    nonzeromodes.count++;
			    //}
		    }
		}
	    } // end of reading one class


	}

	TIndPriorsMap::const_iterator itr2;
	viter=words.begin();
	for(itr2=m.begin();itr2!=m.end();itr2++){
	    if(itr2->second.mode!=0.0) {
		vector<int>::iterator viter2 = find(viter,words.end(),itr2->first);
		// if the feature is active in training example, skip it;
		if (viter2!=words.end())
		    viter = viter2;
		// otherwise, keep it
		else {
		    nonzeromodes.nonzeros.push_back(tri(itr2->first,itr2->second.mode,std::numeric_limits<int>::infinity()));
		}
	    }
        }
   
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

    //SL v3.03
    /*
    void checkNonZeroModes(vector<int>& words, vector<tri>& nonzeromodes) const { 
	_getNonZeroModes(words,nonzeromodes);
    }
    */

    void checkNonZeroModes(vector<int>& words, IndPriorNonZeroModeFeats& nonzeromodes) const { 
	_getNonZeroModes(words,nonzeromodes);
    }

  
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
