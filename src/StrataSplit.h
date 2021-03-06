#ifndef _STRATA_SPLIT_HEADER
#define _STRATA_SPLIT_HEADER

vector<int> PlanStratifiedSplit( class IRowSet & drs, unsigned nfolds );
vector< vector<unsigned> >  //returns arrays of fold ids, one array per stratum
StrataSplit(
             const vector<unsigned>& strataSizes, //sizes per strata; length defines #strata
             unsigned nfolds );
void ReportSplit( unsigned nfolds, const vector< vector<unsigned> >& split, ostream& o );

#endif //_STRATA_SPLIT_HEADER

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
