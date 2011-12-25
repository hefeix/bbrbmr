/*
 *  Feb 18, 05  bugs: 1) rand()/(randmax+1)   2) class size =0
 */

#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

#include <stdlib.h>

#include "logging.h"
#include "data.h"
#include "StrataSplit.h"

const double randmax = RAND_MAX;

static 
void rndperm( vector<unsigned>& a ) {
    unsigned n = a.size();
    if(0==n) return;
    unsigned swap;
    for( unsigned i=0; i<n; i++ )   a.at(i) = i;
    for( unsigned i=0; i+1<n; i++ ) {  
        unsigned r = rand();
        unsigned j = unsigned( i + (r/(randmax+1))*(n-i) );
        swap = a.at(i); a.at(i)=a.at(j); a.at(j)=swap;
    }
}
static 
vector<unsigned> rndperm( unsigned n ) {
    vector<unsigned> a( n );
    rndperm( a );
    return a;
}
/*
static vector< vector<unsigned> >  //returns arrays of fold ids, one array per stratum
StrataSplit(
             const vector<unsigned>& strataSizes, //sizes per strata; length defines #strata
             unsigned nfolds )
{
    unsigned nstr = strataSizes.size();
    vector< vector<unsigned> > splits( nstr );

    //permutations for each stratum
    for( unsigned istr=0; istr<nstr; istr++ ) {
        splits.at(istr).resize( strataSizes.at(istr) ); //grow each to proper size
        rndperm( splits.at(istr) ); //and create a permutation in place
    }

   //reorder strata themselves, compute shifts for reel-in
    vector<unsigned> strOrder = rndperm( nstr );
    vector<unsigned> strShifts( nstr );
    unsigned cumul = 0;
    for( unsigned i=0; i<nstr; i++ ) {
        unsigned inext = find(strOrder.begin(),strOrder.end(),i) - strOrder.begin();
        strShifts.at(inext) = cumul % nfolds;
        cumul += strataSizes.at(inext);
    }

    //do the split - emulate reel-in
    for( unsigned istr=0; istr<nstr; istr++ )
        for( unsigned j=0; j<strataSizes.at(istr); j++ ) {
           splits.at(istr).at(j) = (splits.at(istr).at(j) + strShifts.at(istr)) % nfolds;
        }

    return splits;
}*/

static void
StrataSplit(
             const vector<unsigned>& strataSizes, //sizes per strata; length defines #strata
             unsigned nfolds,
             vector< vector<unsigned> > & splits )
{
    unsigned nstr = strataSizes.size();

    //permutations for each stratum
    for( unsigned istr=0; istr<nstr; istr++ ) {
        rndperm( splits.at(istr) );
    }

    //reorder strata themselves, compute shifts for reel-in
    vector<unsigned> strOrder = rndperm( nstr );
    vector<unsigned> strShifts( nstr );
    unsigned cumul = 0;
    for( unsigned i=0; i<nstr; i++ ) {
        unsigned inext = find(strOrder.begin(),strOrder.end(),i) - strOrder.begin();
        strShifts.at(inext) = cumul % nfolds;
        cumul += strataSizes.at(inext);
    }

    //do the split - emulate reel-in
    for( unsigned istr=0; istr<nstr; istr++ )
        for( unsigned j=0; j<strataSizes.at(istr); j++ ) {
           splits.at(istr).at(j) = (splits.at(istr).at(j) + strShifts.at(istr)) % nfolds;
        }
}
static void ReportSplit( unsigned nfolds, const vector< vector<unsigned> >& split, ostream& o ) {
    unsigned nstrata = split.size();
    vector<unsigned> totals( nfolds, 0 );
    for( unsigned istr=0; istr<nstrata; istr++ ) {
        vector<unsigned> perfold( nfolds, 0 );
        for( unsigned j=0; j<split.at(istr).size(); j++ )
            perfold.at( split.at(istr).at(j) ) ++;
        o<<"\nStrata "<<istr<<" Folds:";
        for( unsigned f=0; f<nfolds; f++ ) {
            o<<" "<<perfold.at(f);
            totals.at(f) += perfold.at(f);
        }
    }
    o<<"\n   Fold totals:";
    for( unsigned f=0; f<nfolds; f++ )
        o<<" "<<totals.at(f);
}

vector<int> PlanStratifiedSplit( IRowSet & drs, unsigned nfolds )
{
    vector<int> splitindices;

    //collect class totals
    vector<unsigned> classtotals( drs.c(), 0 );
    drs.rewind();
    while( drs.next() )
        classtotals.at( drs.y() ) ++;

    //plan the split
    unsigned nstr = classtotals.size();
    vector< vector<unsigned> > split( nstr );
    for( unsigned istr=0; istr<nstr; istr++ ) {
        split.at(istr).resize( classtotals.at(istr) ); //grow each to proper size
    }
    StrataSplit( classtotals, nfolds, split );
    ReportSplit( nfolds, split, Log(6) );

    vector<unsigned> classindices( drs.c(), 0 ); //cases already met per class
    drs.rewind();
    while( drs.next() ) {
        splitindices.push_back( split .at(drs.y()) .at(classindices.at(drs.y())) );
        classindices.at(drs.y()) ++;
    }

    return splitindices;
}

#ifdef _TEST_ONLY
void main() {
    unsigned str.at()={3,5,7};
    vector<unsigned> strataSizes; //( str, str+sizeof(str) );
    strataSizes.push_back(3);strataSizes.push_back(5);strataSizes.push_back(7);
    unsigned nfolds = 4;
    for( int repeat=0; repeat<10; repeat++ ) {
        vector< vector<unsigned> >  split = StrataSplit( strataSizes, nfolds );
        ReportSplit( nfolds, split, cout );
        /*cout<<"strataSizes "<<strataSizes<<"\nnfolds "<<nfolds;
        cout<<"\n split";
        for( unsigned i=0; i<split.size(); i++ )
            cout<<"\nClass "<<i<<": "<<split.at(i);*/
    }
    /*for( unsigned ifold=0; ifold<nfolds; ifold++ ) {
        cout<<"\nFold "<<ifold<<": ";
        for( unsigned i=0; i<split.size(); i++ )*/
}
#endif //_TEST_ONLY


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
