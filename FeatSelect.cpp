#include <limits>
#include <algorithm>
#include <iomanip>

#include "logging.h"
#include "data.h"
//#include "bayes.h"
#include "FeatSelectParamManager.h"

using namespace std;

vector<int> FeatureSelectionByCorr(  //by Pearson's correlation New
                const vector<SparseVector> & sparse,
                const vector<bool>& y,
                const vector<int>& featsIn, //must be sorted!
                int nFeatsSel )
{
    int nFeatsIn = featsIn.size();
    Log(3)<<"\nFeature selection, "<<nFeatsSel<<" of "<<nFeatsIn;
    assert(nFeatsSel>=1);
    int n = sparse.size();

    //normalize y
    double ymean = ntrue(y) / double(n);
    double ystd = sqrt( ymean*(1-ymean) );
    double yposnorm = (1-ymean)/ystd;
    double ynegnorm = -ymean/ystd;
    //Log(10)<<"\nYnorm "<<yposnorm<<" "<<ynegnorm;

    //prepare feature-wise arrays
    vector<double> xmean(nFeatsIn,0.0), xmeansqu(nFeatsIn,0.0), xyNorm(nFeatsIn,0.0);
    vector<unsigned> xn(nFeatsIn,0); //# non-zeroes

    // rows loop
    for( int r=0; r<n; r++ )
    {
        // ! 'featsIn' should be sorted
        //intersection of sparse row and featsIn
        SparseVector::const_iterator ix = sparse[r].begin();
        unsigned iFeat = 0;
        while( ix!=sparse[r].end() && iFeat<featsIn.size() ) { //merge
            if( ix->first < featsIn[iFeat] )          ix ++;
            else if( featsIn[iFeat] < ix->first )     iFeat ++;
            else { //equal values
                double x = ix->second; //TF value
                if( y[r] )  xyNorm[iFeat] += x * yposnorm;
                else        xyNorm[iFeat] += x * ynegnorm;
                double fNew = 1.0 /( xn[iFeat] + 1 );
                double fPrev = xn[iFeat] * fNew;
                xmean[iFeat] = xmean[iFeat] * fPrev + x * fNew;
                xmeansqu[iFeat] = xmeansqu[iFeat] * fPrev + x*x * fNew;
                xn[iFeat] ++;
                ix ++;
                iFeat ++;
            }
        }
    }

    //compute correlation
    vector< pair<double,int> > featUtility;
    //int npos = ntrue( y );
    for( int f=0; f<nFeatsIn; f++ ) {
        xmean[f] = xmean[f] * xn[f] / n;
        xmeansqu[f] = xmeansqu[f] * xn[f] / n;
        double stddev = sqrt( xmeansqu[f] - xmean[f]*xmean[f] );
        double corr = 0==stddev ? 0 
            : xyNorm[f] / n / stddev;
        featUtility.push_back( pair<double,int>(fabs(corr),featsIn[f]) );
            //Log(10)<<f<<" "<<corr<<endl;
    }

    //select 'nFeats' best
    std::sort( featUtility.begin(), featUtility.end() );
    if( nFeatsSel > nFeatsIn )
        nFeatsSel = nFeatsIn;
    Log(3)<<"\nCorrelation - Threshold for feature selection "
        <<featUtility[nFeatsIn-nFeatsSel].first<<endl;

    vector<int> featSelect;
    Log(7)<<endl<<"Correlation   ";
    for( int c=nFeatsIn-nFeatsSel; c<nFeatsIn; c++ ) //best corr at the end
    {
        featSelect.push_back( featUtility[c].second );
        Log(7)<<" "<<featUtility[c].second<<":"<<featUtility[c].first;
    }

    std::sort( featSelect.begin(), featSelect.end() );

    return featSelect;
}

vector<int> FeatureSelectionByChiSqu(
                const vector<SparseVector> & sparse,
                const vector<bool>& y,
                const vector<int>& featsIn, //must be sorted!
                int nFeatsSel )
{
    int nFeatsIn = featsIn.size();
    Log(3)<<"\nFeature selection, "<<nFeatsSel<<" of "<<nFeatsIn<<endl;
    assert(nFeatsSel>=1);
    int n = sparse.size();

    double py1 = ntrue(y) / double(n); //marginal prob y=1
    double pvar = py1*(1-py1);

    //prepare feature-wise arrays
    typedef map< double, pair<unsigned,unsigned> >  contab;
    //typedef vector< pair<unsigned,unsigned> >  contab;
    vector<contab> ct(nFeatsIn);

    // rows loop
    for( int r=0; r<n; r++ )
    {
        // ! 'featsIn' should be sorted
        //intersection of sparse row and featsIn
        SparseVector::const_iterator ix = sparse[r].begin();
        unsigned iFeat = 0;
        while( ix!=sparse[r].end() && iFeat<featsIn.size() ) { //merge
            if( ix->first < featsIn[iFeat] )          ix ++;
            else if( featsIn[iFeat] < ix->first )     iFeat ++;
            else { //equal values
                double x = ix->second; //TF value

                contab::iterator iFeatVal = ct[iFeat].find(x);
                if( iFeatVal == ct[iFeat].end() ) { //not found
                    pair<unsigned,unsigned> ab( y[r]?1:0, y[r]?0:1 );
                    ct[iFeat][x] = ab;
                }
                else //already there
                    y[r] ? iFeatVal->second.first++ : iFeatVal->second.second++ ;

                ix ++;
                iFeat ++;
            }
        }
    }

    //compute chi-squ
    vector< pair<double,int> > featUtility;
    for( int f=0; f<nFeatsIn; f++ ) {
        double chisqu = 0;
        for( contab::const_iterator iab=ct[f].begin(); iab!=ct[f].end(); iab++ ) {
            unsigned a = iab->second.first;
            unsigned b = iab->second.second;
            double c = a*(1-py1) - b*py1;
            c *= c;
            c /= py1 * (1-py1) * (a+b);
            chisqu += c;
        }
        featUtility.push_back( pair<double,int>( chisqu, featsIn[f] ) );
            //Log(10)<<f<<" "<<chisqu<<endl;
    }

    //select 'nFeats' best
    std::sort( featUtility.begin(), featUtility.end() );
    if( nFeatsSel > nFeatsIn )
        nFeatsSel = nFeatsIn;
    Log(3)<<"\nChi-square - Threshold for feature selection "
        <<featUtility[nFeatsIn-nFeatsSel].first<<endl;

    vector<int> featSelect;
    //Log(8)<<endl<<"Correlation   ";
    for( int c=nFeatsIn-nFeatsSel; c<nFeatsIn; c++ ) //best corr at the end
    {
        featSelect.push_back( featUtility[c].second );
        //Log(8)<<" "<<featUtility[c].second<<":"<<featUtility[c].first;
    }

    std::sort( featSelect.begin(), featSelect.end() );

    return featSelect;
}
/*
 * chi-square for binary data: term present/absent only
 */
vector<int> FeatureSelectionByChiSqu01(
                const vector<SparseVector> & sparse,
                const vector<bool>& y,
                const vector<int>& featsIn, //must be sorted!
                int nFeatsSel )
{
    int nFeatsIn = featsIn.size();
    Log(3)<<"\nFeature selection by Chi-Square, binary version, "<<nFeatsSel<<" of "<<nFeatsIn<<endl;
    assert(nFeatsSel>=1);
    int n = sparse.size();
    int npos = ntrue( y );

    //prepare a,b,c,d arrays
    vector<int> x1y1(nFeatsIn,0), x1y0(nFeatsIn,0);

    // rows loop
    for( int r=0; r<n; r++ )
    {
        // ! 'featsIn' should be sorted
        //intersection of sparse row and featsIn
        SparseVector::const_iterator ix = sparse[r].begin();
        unsigned iFeat = 0;
        while( ix!=sparse[r].end() && iFeat<featsIn.size() ) { //merge
            if( ix->first < featsIn[iFeat] )          ix ++;
            else if( featsIn[iFeat] < ix->first )     iFeat ++;
            else { //equal values
                if( y[r] )  x1y1[iFeat] ++;
                else        x1y0[iFeat] ++;
                ix ++;
                iFeat ++;
            }
        }
    }

    //compute chi-squ
    vector< pair<double,int> > featUtility;
    const double cntadj = 1.0/12; //continuity adjustment
    for( int f=0; f<nFeatsIn; f++ ) {
        double a=x1y1[f], b=x1y0[f], c=npos-a, d=n-a-b-c;
            //?? a+=cntadj; b+=cntadj; c+=cntadj; d+=cntadj;
        double chisqu = n*(a*d - b*c)*(a*d - b*c);
        double denom = (a+b)*(a+c)*(b+d)*(c+d);
        if( 0.0==denom ) chisqu = 0.0;
        else chisqu /= denom;
        featUtility.push_back( pair<double,int>( chisqu, featsIn[f] ) );
            //Log(10)<<"feat/a/b/c/d/chisqu "<<featsIn[f]<<" "<<a<<" "<<b<<" "<<c<<" "<<d<<" "<<chisqu<<endl;
    }

    //select 'nFeats' best
    std::sort( featUtility.begin(), featUtility.end() );
    if( nFeatsSel > nFeatsIn )
        nFeatsSel = nFeatsIn;
    Log(3)<<"\nChi-square - Threshold for feature selection "
        <<featUtility[nFeatsIn-nFeatsSel].first<<endl;

    vector<int> featSelect;
    Log(7)<<endl<<"Chi-square   ";
    for( int c=nFeatsIn-nFeatsSel; c<nFeatsIn; c++ ) //best corr at the end
    {
        featSelect.push_back( featUtility[c].second );
        Log(7)<<" "<<featUtility[c].second<<":"<<featUtility[c].first;
    }

    std::sort( featSelect.begin(), featSelect.end() );

    return featSelect;
}

/*
 * Yule's Q is based on 2-by-2 contigency table, so TF numeric value is ignored
 *  Q = (ad-bc) / (ad+bc)
 */
vector<int> FeatureSelectionByYule(
                const vector<SparseVector> & sparse,
                const vector<bool>& y,
                const vector<int>& featsIn, //must be sorted!
                int nFeatsSel )
{
    int nFeatsIn = featsIn.size();
    Log(3)<<"\nFeature selection, "<<nFeatsSel<<" of "<<nFeatsIn<<endl;
    assert(nFeatsSel>=1);
    int n = sparse.size();

    //prepare a,b,c,d arrays
    vector<int> x1y1(nFeatsIn,0), x1y0(nFeatsIn,0);

    // rows loop
    for( int r=0; r<n; r++ )
    {
        // ! 'featsIn' should be sorted
        //intersection of sparse row and featsIn
        SparseVector::const_iterator ix = sparse[r].begin();
        unsigned iFeat = 0;
        while( ix!=sparse[r].end() && iFeat<featsIn.size() ) { //merge
            if( ix->first < featsIn[iFeat] )          ix ++;
            else if( featsIn[iFeat] < ix->first )     iFeat ++;
            else { //equal values
                if( y[r] )  x1y1[iFeat] ++;
                else        x1y0[iFeat] ++;
                ix ++;
                iFeat ++;
            }
        }
    }

    //compute Q
    const double cntadj = 1.0/12; //continuity adjustment
    vector< pair<double,int> > featUtility;
    int npos = ntrue( y );
    for( int f=0; f<nFeatsIn; f++ ) {
        double a=x1y1[f], b=x1y0[f], c=npos-a, d=n-a-b-c;
        a+=cntadj; b+=cntadj; c+=cntadj; d+=cntadj;
        double denom = a*d + b*c;
        double Q = 0==denom ? 0 : (a*d - b*c)/denom;
        featUtility.push_back( pair<double,int>(fabs(Q),featsIn[f]) );
        //Log(10)<<"feat/a/b/c/d/Q "<<featsIn[f]<<" "<<a<<" "<<b<<" "<<c<<" "<<d<<" "<<Q<<endl;
    }

    //select 'nFeats' best
    std::sort( featUtility.begin(), featUtility.end() );
    if( nFeatsSel > nFeatsIn )
        nFeatsSel = nFeatsIn;
    Log(3)<<"\nYule's Q - Threshold for feature selection "
        <<featUtility[nFeatsIn-nFeatsSel].first<<endl;

    //BoolVector featSelect( true, data.dim() );
    vector<int> featSelect;
    //Log(10)<<endl<<"Yule's Q   ";
    for( int c=nFeatsIn-nFeatsSel; c<nFeatsIn; c++ ) //best corr at the end
    {
        featSelect.push_back( featUtility[c].second );
        //Log(10)<<" "<<featUtility[c].second<<"/"<<featUtility[c].first;
    }
    //Log(10)<<endl;

    std::sort( featSelect.begin(), featSelect.end() );

    return featSelect;
}

/*
 * Bi-Normal separation by Forman:  http://www.jmlr.org/papers/volume3/forman03a/forman03a.pdf
 *  BNS = | \Phi^{-1}(tp/(tp+fn)) -  \Phi^{-1}(fp/(tn+fp)) |
 *  \Phi is standard normal c.d.f.
 *  Function arguments are really sensitivity and 1-specificity
 */
vector<int> FeatureSelectionByBNS(
                const vector<SparseVector> & sparse,
                const vector<bool>& y,
                const vector<int>& featsIn, //must be sorted!
                int nFeatsSel )
{
    int nFeatsIn = featsIn.size();
    Log(3)<<"\nFeature selection, "<<nFeatsSel<<" of "<<nFeatsIn<<endl;
    assert(nFeatsSel>=1);
    int n = sparse.size();

    //prepare a,b,c,d arrays
    vector<int> x1y1(nFeatsIn,0), x1y0(nFeatsIn,0);

    // rows loop
    for( int r=0; r<n; r++ )
    {
        // ! 'featsIn' should be sorted
        //intersection of sparse row and featsIn
        SparseVector::const_iterator ix = sparse[r].begin();
        unsigned iFeat = 0;
        while( ix!=sparse[r].end() && iFeat<featsIn.size() ) { //merge
            if( ix->first < featsIn[iFeat] )          ix ++;
            else if( featsIn[iFeat] < ix->first )     iFeat ++;
            else { //equal values
                if( y[r] )  x1y1[iFeat] ++;
                else        x1y0[iFeat] ++;
                ix ++;
                iFeat ++;
            }
        }
    }

    //compute BNS
    vector< pair<double,int> > featUtility;
    int npos = ntrue( y );
    const double cutoff=0.0005;
    for( int f=0; f<nFeatsIn; f++ ) {
        double tp=x1y1[f], fp=x1y0[f], fn=npos-tp, tn=n-tp-fp-fn;
        double sensty = tp/npos;
        if( sensty<cutoff )  sensty=cutoff;
        if( sensty>1-cutoff )  sensty=1-cutoff;
        double compl_specty = fp/(tn+fp);
        if( compl_specty<cutoff )  compl_specty=cutoff;
        if( compl_specty>1-cutoff )  compl_specty=1-cutoff;
        double BNS = fabs( normsinv(sensty) - normsinv(compl_specty) );
        featUtility.push_back( pair<double,int>(BNS,featsIn[f]) );
        //Log(10)<<"feat/sensty/compl_specty/BNS "<<featsIn[f]<<" "<<sensty<<" "<<compl_specty<<" "<<BNS<<endl;
    }

    //select 'nFeats' best
    std::sort( featUtility.begin(), featUtility.end() );
    if( nFeatsSel > nFeatsIn )
        nFeatsSel = nFeatsIn;
    Log(3)<<"\nBNS - Threshold for feature selection "
        <<featUtility[nFeatsIn-nFeatsSel].first<<endl;

    //BoolVector featSelect( true, data.dim() );
    vector<int> featSelect;
    //Log(8)<<endl<<"BNS   ";
    for( int c=nFeatsIn-nFeatsSel; c<nFeatsIn; c++ ) //best at the end
    {
        featSelect.push_back( featUtility[c].second );
        //Log(8)<<" "<<featUtility[c].second<<"/"<<featUtility[c].first;
    }
    //Log(8)<<endl;

    std::sort( featSelect.begin(), featSelect.end() );

    return featSelect;
}

vector<int> FeatureSelection(
                const FeatSelectParameter& featSelectParameter,
                const vector<SparseVector> & sparse,
                const vector<bool>& y,
                const vector<int>& featsIn )
{
    vector<int> featSelect;
    if( ! featSelectParameter.isOn() )
        throw logic_error("Feature selection parameter invalid");
    if( featSelectParameter.corr==featSelectParameter.utilityFunc )
        featSelect = FeatureSelectionByCorr( sparse, y, featsIn, featSelectParameter.nFeatsToSelect );
    else if( featSelectParameter.Yule==featSelectParameter.utilityFunc )
        featSelect = FeatureSelectionByYule( sparse, y, featsIn, featSelectParameter.nFeatsToSelect );
    else if( featSelectParameter.chisqu==featSelectParameter.utilityFunc )
        featSelect = FeatureSelectionByChiSqu01( sparse, y, featsIn, featSelectParameter.nFeatsToSelect );
    else if( featSelectParameter.BNS==featSelectParameter.utilityFunc )
        featSelect = FeatureSelectionByBNS( sparse, y, featsIn, featSelectParameter.nFeatsToSelect );
    else
        featSelect = vector<int>(); //never gets here, for compilation only

    Log(3)<<endl<<"Features selected, Time   "<<Log.time();
    Log(7)<<endl;
    for(unsigned i=0;i<featSelect.size();i++ )
        Log(7)<<" "<<featSelect[i];
    Log(3)<<endl;
    return featSelect;
}

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
