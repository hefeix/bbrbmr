#ifndef _SQUEEZER_PARAMETER_HPP
#define _SQUEEZER_PARAMETER_HPP

#include <fstream>
#include <string>
#include <map>

#ifdef USE_LEMUR
# include "Param.hpp"
#endif //USE_LEMUR

using namespace std;

class Squeezer {
    string fileName;
    unsigned nFeats;
    map<string, double > hpByTopic; //hyperparam to start from
public:
    //access
    bool enabled() const { return nFeats>0; }
    double HPforTopic( const string topic ) const {
        map<string, double >::const_iterator itr= hpByTopic.find( topic );
        if( itr==hpByTopic.end() )
            throw runtime_error(string("Unknown topic for squeezer: ")+topic);
        return itr->second;
    }
    const string topicFile() const { return fileName; }
    unsigned NFeats() const { return  nFeats; }
    //ctors
#ifdef USE_LEMUR
    void get() {
        fileName = ParamGetString("SqueezeHPByTopic","");
        nFeats = (unsigned)ParamGetInt("SqueezeTo",0);

        if( fileName.size()>0 ) {
            ifstream f(fileName.c_str());
            // each row is: <topic> <hp-float-value>
            string topic;
            double hp;
            while( f>>topic>>hp ) {//by topic
                hpByTopic[topic] = hp;
            }//by topic
            if( !f.eof() )
                throw runtime_error("Corrupted squeezer-by-topic file");
        }
        //for( map<string, double >::const_iterator itr=hpByTopic.begin();itr!=hpByTopic.end();itr++)
        //    cout<<endl<<"-- "<<itr->first<<" "<<itr->second;
    }
#endif //USE_LEMUR
};

inline std::ostream& operator<<( std::ostream& o, const Squeezer& s ) {
    if( s.enabled() )
        o<<std::endl<<"Squeeze to "<<s.NFeats()
            <<" Start hyperparameter values from: "<<s.topicFile();
    else
        o<<std::endl<<"Squeezer: Disabled";
    return o;
}

#endif //_SQUEEZER_PARAMETER_HPP

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
