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
