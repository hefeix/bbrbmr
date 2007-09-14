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
vector<unsigned> rndperm( unsigned n ) {
    vector<unsigned> a( n );
    rndperm( a );
    return a;
}
vector< vector<unsigned> >  //returns arrays of fold ids, one array per stratum
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
}
void
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
void ReportSplit( unsigned nfolds, const vector< vector<unsigned> >& split, ostream& o ) {
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
