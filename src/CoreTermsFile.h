#ifndef CORE_TERMS_RESTRICTION_FILES_
#define CORE_TERMS_RESTRICTION_FILES_

#include <set>
#include <string>
#include <fstream>
#include <sstream>

#include "Param.hpp"
#include "logging.h"

class CoreTermsRestriction {
    std::string fileMask;
    std::set<int> indices;
public:
    bool isOn() const { return fileMask.size()>0; }
    bool has( int i ) const {
        if( isOn() )
            return( indices.find(i) != indices.end() );
        else
            return true; //no restriction
    }
    size_t nterms() const {
        return( isOn() ? indices.size() : 0 );   }
    void get() {
        fileMask = ParamGetString("coreFile","");
        Log(2)<<std::endl<<"Core Terms File Mask: "
            <<( isOn() ? fileMask.c_str() : "<none>");
    }
    bool setTopic( const char* topic ){
        if( !isOn() )
            return true; //--->>--

        indices.clear();

        std::string fileName=fileMask;
        size_t pos=fileName.find_first_of('?');
        if( pos<fileName.size() )
            fileName.replace(pos,1,topic);

        std::ifstream f(fileName.c_str());
        if( !f.good() )
            return false; //--->>--
        const int bufsize=1000;
        char buf[bufsize+1];
        while( f.getline( buf, bufsize ).good() ) {
            int i;
            istringstream rowbuf( buf );
            rowbuf >> i/*ignore first*/ >> i;
            if( rowbuf.fail() )
                return false; //--->>--
            indices.insert( i );
        }
        return true;
    }
};

#endif //CORE_TERMS_RESTRICTION_FILES_


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
