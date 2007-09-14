#ifndef logging_level_controlled_
#define logging_level_controlled_

#include <iostream>
#include <fstream>
#include <strstream>
#include <string>
#include <time.h>

class logging {
    unsigned int m_level;
    std::ostrstream dull;
    time_t start;
    std::ofstream logstream;
    bool isstdout;
    void init() {
        dull.freeze();
        start = ::time(NULL);
    }
public:
    logging(unsigned int level_=0) : m_level(level_) {
        isstdout = true;
        init(); }
    logging( std::string name, unsigned int level_=0) : m_level(level_), logstream((name+".lis").c_str()) {
        isstdout = false;
        init(); }
    void setLevel(unsigned int level_ ) {
        m_level = level_;     }
    std::ostream& operator()(unsigned int l=0) {
        if( l<=m_level) //return( isstdout ? std::cout : logstream );
            if( isstdout ) return std::cout;
            else return logstream;
        else return dull;
    }
    time_t time() const { return ::time(NULL) - start; }
    unsigned int level() const {  return m_level; }
};

extern logging Log;

#endif //logging_


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
