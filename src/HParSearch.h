#ifndef HYPER_PARAMETER_SEARCH_HPP_
#define HYPER_PARAMETER_SEARCH_HPP_

#include <map>
/*#include <ostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>*/
using namespace std;

/* UniModalSearch: 
 - collects loglikelihood results from previous steps
 - suggests next step
*/
class UniModalSearch {
public:
    struct MS{ //mean and stddev
        double m; double s;
        MS( double m_=0, double s_=0 ) : m(m_), s(s_) {}
    };
    double bestx() const {
        return best->first; }
    double besty() const {
        return best->second.m; }
    void tried( double x, double y, double y_stddev=0.0 ) //get to know next result (x,y)
    {
        y_by_x[x] = MS(y,y_stddev);
        if( y_by_x.size()==1 ) { //this is the first
            best = y_by_x.begin();
        }
        else {
            double improve = (y - best->second.m) / best->second.m;
            if( y > best->second.m )
                best = y_by_x.find( x );
        }
    }
    pair<bool,double> step(); // recommend: do/not next step, the next x value
    //ctor
    UniModalSearch( double stdstep=100, double stop_by_y=.01, double stop_by_x=log(1.5) ) 
        : m_stdstep(stdstep), m_stop_by_y(stop_by_y), m_stop_by_x(stop_by_x) {}
private:
    const double m_stdstep;
    const double m_stop_by_y, m_stop_by_x;
    std::map<double,MS> y_by_x;
    std::map<double,MS>::const_iterator best;
};

#endif //HYPER_PARAMETER_SEARCH_HPP_


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
