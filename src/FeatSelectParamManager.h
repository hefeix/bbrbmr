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
