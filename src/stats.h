//  v.2.04a     May 30, 05  stddev is zero when it seems that meansqu < mean^2

#ifndef STATISTICS_FOR_ROWSET_
#define STATISTICS_FOR_ROWSET_

#include "data.h"

class Stats {
    IRowSet & m_source;
    mutable Vector m_stddevs;
    mutable Vector m_means;
    mutable double m_avgSquNorm;
    mutable unsigned m_nrows;
    mutable bool m_ready; //laziness
public:
    Stats(  IRowSet & drs_ )  // VM -- added for testing
    : m_source(drs_), m_ready(false) {}
    Stats(  DesignRowSet & drs_ )
    : m_source(drs_), m_ready(false) {}
    const unsigned NRows() const { 
        if( !m_ready ) getready();
        return m_nrows; 
    }
    const Vector& Means() const { 
        if( !m_ready ) getready();
        return m_means; 
    }
    const Vector& Stddevs() const { 
        if( !m_ready ) getready();
        return m_stddevs; 
    }
    double AvgSquNorm() const {  //for default bayes parameter: avg x*x
        if( !m_ready ) getready();
        return m_avgSquNorm; 
    }
private:
    void getready() const { //laziness
        m_means.resize( m_source.dim(), 0.0 );
        m_avgSquNorm = 0.0;
        vector<double> meansqu( m_source.dim(), 0.0 );
        vector<unsigned> xn( m_source.dim(), 0 ); //non-zeroes

        Log(8)<<"\nCompute stddevs start Time "<<Log.time();

        // compute m_means and stddevs for non-zeroes only first
        m_source.rewind();
        unsigned r;
        for( r=0; m_source.next(); r++ )
        {
            const SparseVector& x=m_source.xsparse();
            double squNorm = 0;
            for( SparseVector::const_iterator xitr=x.begin(); xitr!=x.end(); xitr++ )
            {
                unsigned iFeat = m_source.colNumById( xitr->first );
                double val = xitr->second;
                double fNew = 1.0 /( xn[iFeat] + 1 );
                double fPrev = xn[iFeat] * fNew;
                m_means[iFeat] = m_means[iFeat] * fPrev + val * fNew;
                meansqu[iFeat] = meansqu[iFeat] * fPrev + val*val * fNew;
                xn[iFeat] ++;
                squNorm += val*val;
                //Log(8)<<"\n- "<<iFeat<<" "<<val<<" "<<m_means[iFeat]<<" "<<meansqu[iFeat]<<" "<<squNorm;
            }
            m_avgSquNorm = m_avgSquNorm*r/(r+1) + squNorm/(r+1);
            //Log(6)<<"\nm_avgSquNorm "<<m_avgSquNorm;
        }
        //Log(5)<<"\nr "<<r;

        //now adjust for zeroes
        m_stddevs.resize( m_source.dim(), 0.0 );
            //Log(5)<<"\ni xn mean meansqu stddev";
        for( unsigned f=0; f<m_source.dim(); f++ ) {
            double adjust = double(xn[f]) / double(r);
            //Log(9)<<"\nstats "<<f<<" "<<xn[f]<<" "<<m_means[f]<<" "<<meansqu[f]<<" "<<adjust;
            m_means[f] *= adjust;
            meansqu[f] *= adjust;
            double stddev_squ = meansqu[f] - m_means[f]*m_means[f];
            m_stddevs[f] = stddev_squ<=0 ? 0 : sqrt( stddev_squ );
                //most people do it this way: sqrt( ( meansqu[f] - m_means[f]*m_means[f] )*r/(r-1) );
            //Log(9)<<" "<<m_stddevs[f];
        }
        //Log(8)<<"\nCompute stddevs end Time "<<Log.time();

        m_nrows = r;
        m_ready = true;
    }
};

#endif //STATISTICS_FOR_ROWSET_

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
