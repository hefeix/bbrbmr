#ifndef _FEAT_SELECT_PARAMETER_HPP
#define _FEAT_SELECT_PARAMETER_HPP

#include <fstream>
#include <string>
#include <map>
#include <stdexcept>

#ifdef USE_LEMUR
# include "Param.hpp"
#endif //USE_LEMUR

using namespace std;

class FeatSelectParameter {
    friend class FeatSelectByTopic;
public:
    enum UtilityFunc { corr=0, Yule, chisqu, BNS, undef };
    UtilityFunc utilityFunc;
    int nFeatsToSelect;

    FeatSelectParameter( UtilityFunc uf=undef, int nfeats=0 )
        : utilityFunc(uf), nFeatsToSelect(nfeats) {}

    int nFeats() const {
        return nFeatsToSelect; }
    bool isOn() const {
        return ( utilityFunc!=undef && nFeatsToSelect>0 );  }
};

class FeatSelectByTopic {
    string fsFileName;
    map<string, int > fsByTopic;
    FeatSelectParameter proto;
public:
    //access
    FeatSelectParameter GetForTopic( const string topic="" ) const {
        if( topicMode() ) {
            map<string, int >::const_iterator itr=fsByTopic.find( topic );
            if( itr==fsByTopic.end() )
                throw runtime_error(string("Unknown topic for feature selection: ")+topic);
            return FeatSelectParameter( proto.utilityFunc, itr->second );
        }
        else
            return proto;
    }
    bool topicMode() const { return 0<fsByTopic.size(); }
    const string topicFile() const { return fsFileName; }
    FeatSelectParameter::UtilityFunc utility() const { return proto.utilityFunc; }
    //ctors
    FeatSelectByTopic() {}
    FeatSelectByTopic( FeatSelectParameter::UtilityFunc uf, int nfeats ) : proto( uf, nfeats ) {}

#ifdef USE_LEMUR
    void get() {
        fsFileName = ParamGetString("FeatSelectByTopic","");
        proto.utilityFunc = (FeatSelectParameter::UtilityFunc)ParamGetInt("UtilityFunc",FeatSelectParameter::corr);
        if(proto.utilityFunc>FeatSelectParameter::undef) 
            proto.utilityFunc=FeatSelectParameter::undef;
        proto.nFeatsToSelect = ParamGetInt("nFeatsToSelect",0);

        if( fsFileName.size()>0 ) {
            ifstream f(fsFileName.c_str());
            // each row is: <topic> [label]*
            while( f ) {//by topic
                string topic;
                int nfs;
                f>>topic>>nfs;
                fsByTopic[topic] = nfs;
            }//by topic
            if( !f.eof() )
                throw runtime_error("Corrupted feature selection by topic file");
        }
    }
#endif //USE_LEMUR
};

inline std::ostream& operator<<( std::ostream& o, const FeatSelectByTopic& fs ) {
    o<<std::endl<<"Feature Utility Function: "
        <<( fs.utility()==FeatSelectParameter::corr ? "Pearson's correlation" 
            : fs.utility()==FeatSelectParameter::Yule ? "Yule's Q"
            : fs.utility()==FeatSelectParameter::chisqu ? "Chi-square"
            : fs.utility()==FeatSelectParameter::BNS ? "BNS"
            : "undef" )
            <<std::endl<<"#Features To Select: ";
    if( fs.topicMode() )
        o<<"by topic, file "<<fs.topicFile();
    else if( fs.GetForTopic().isOn() )
        o<<fs.GetForTopic().nFeats();
    else
        o<<"use all";
    return o;
}

vector<int> FeatureSelection(
                const class FeatSelectParameter& featSelectParameter,
                const vector<class SparseVector> & sparse,
                const class vector<bool>& y,
                const vector<int>& featsIn );

#endif //_FEAT_SELECT_PARAMETER_HPP

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
