//  v.2.04a     May 30, 05  StandardizedRowSet: check if stddev is zero

#ifndef Data_And_Labels_
#define Data_And_Labels_

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string.h>

#ifndef _MSC_VER
# define  stricmp(a, b)   strcasecmp(a,b) 
# include <strings.h>
#endif

#include "Matrix.h"
#include "logging.h"
#include "TFIDFParamManager.h"

using namespace std;

inline bool readFromStdin( const char* fname ) {
    return 0==stricmp( "-", fname ) || 0==stricmp( "=", fname );    }

inline string int2string(int i) {
    std::ostringstream buf; 
    buf<<i;
    return buf.str();
}
typedef pair<int,double> SparseItem;
class SparseVector : private vector< SparseItem > {
    friend class TopicRowSet;
    friend class RowSetMem;
    friend class StandardizedRowSet;
    friend class LemurTopicRowSet;
    friend class DesignRowSet;
    friend class PlainDesign;
    friend class InteractionsDesign;
    friend SparseVector docToTFvector( int docId, class Index *ind, TFMethod tf );
    void sort() {
        std::sort( begin(), end() );  }
    void insert( int i, double d ) {
        push_back( SparseItem( i, d ) ); }
public:
	typedef vector< SparseItem >::const_iterator const_iterator;
	typedef vector< SparseItem >::iterator iterator;
    const_iterator begin() const { return vector<SparseItem>::begin(); }
    const_iterator end() const { return vector<SparseItem>::end(); }
    iterator begin() { return vector<SparseItem>::begin(); }
    iterator end() { return vector<SparseItem>::end(); }
    const_iterator find( int i ) const { 
        SparseItem pattern( i, - numeric_limits<double>::max() );
        vector< SparseItem >::const_iterator found = lower_bound( begin(), end(), pattern );
        if( found != end() && found->first==i )
            return found;
        else
            return end();
    }
    //ctor
    SparseVector() {}
    //ctor
    SparseVector( const vector<SparseItem>& v ) : vector<SparseItem>( v ) {
        std::sort( begin(), end() );
    }
};
/*class SparseVector : private map<int,double> {};*/
inline std::ostream& operator<<( std::ostream& o, const SparseVector& sv ) {
    for( SparseVector::const_iterator isv=sv.begin(); isv!=sv.end(); isv++ )
        o << isv->first <<":"<<isv->second<<"  ";
    return o;
}

inline double TFWeight(double rawTF, TFMethod tf ) {
    if (tf == RAWTF) {
        return (rawTF);
    } else if (tf == LOGTF) {
        return (rawTF > 0) ?  log(rawTF) +1 : 0;
    } else {  // default to raw TF
        throw logic_error("unknown TF method");
    }
}

SparseVector docToTFvector( int docId, class Index *ind, TFMethod tf = RAWTF );
vector<SparseVector> MakeSparseMatrix( const vector<int>& docs, Index *ind, TFMethod tfMethod );

#ifdef POLYTOMOUS
    typedef unsigned YType; //should be unsigned; not the class id in file but number returned by data object
#else
    typedef bool YType;
#endif

class INameResolver {
public:
    virtual ~INameResolver() {};
    virtual unsigned dim() const =0;
    virtual string rowName( unsigned r ) const=0;
    virtual string colName( unsigned c ) const=0;
//    virtual int rowId( int r ) const=0;
    virtual int colId( unsigned c ) const=0;
    virtual unsigned colNumById( unsigned c ) const=0;
#ifdef POLYTOMOUS
    virtual unsigned c() const =0; //#classes
    virtual int classId( unsigned c ) const=0;
    virtual unsigned classNumById( int c ) const=0;
#else
    unsigned c() const { return 2; } // #classes
#endif //POLYTOMOUS
};

class IDenseData : public INameResolver {
public:
    virtual const Matrix& x() const =0;
    virtual SparseVector xsparse( unsigned row )  const =0;
    virtual const vector<YType>& y() const =0;
    //virtual void Normalize() =0;
    virtual ~IDenseData() {};
    unsigned dim() const { return x().nCols(); }
    int n() const { return x().nRows(); }
    int nRel() const { return ntrue(y()); }
    int nNonRel() const { return n() - ntrue(y()); }
    //virtual const char* rowName( unsigned r ) const=0;
//    virtual int rowId( int r ) const=0;
};

class IScores{
public:
    virtual unsigned n() const =0;
    virtual bool next() =0;
    virtual void rewind() =0;
    virtual double val(unsigned i) =0;
    virtual ~IScores() {}
};

class DenseData : public IDenseData {
    Matrix m_x;
    vector<YType> m_y;
    INameResolver* pNames;
    bool cosineNormalize;
    const vector<int>& docs;  //to resolve row id
    //auxiliary for creation time only
    int m_dim, m_nr;
    valarray<double> m_v;
    //void addRow( valarray<double> x, bool y, int id );
    void addRow( valarray<double> x, bool y );
public:
    //ctor's
    //DenseData( const Matrix& x, const BoolVector& y) : m_x(x), m_y(y) {}; 
    //from plain file; assumes labels in the last column
    //DenseData( const char* file); 
    //from text data
    //DenseData( const vector< pair<SparseVector,bool> >&, const std::hash_map<int,int>& );
    DenseData( const vector<SparseVector>& x, const vector<YType> y, 
        const vector<int>& terms, const vector<int>& docs, Index* index );
    ~DenseData() { delete pNames; }

    //access
    const Matrix& x() const { return m_x; }
    SparseVector xsparse( unsigned row ) const {
        if( row>=m_x.nRows() )
            throw logic_error("Dense data: row out of bound");
        vector<SparseItem> x;
        for( unsigned c=0; c<m_x.nCols(); c++ )
            if( 0.0!=m_x.val( row, c ) )
                x.push_back(SparseItem( colId(c), m_x.val( row, c ) )); //x[ colId(c) ] = m_x.val( row, c );
        return SparseVector( x );
    }
    const vector<YType>& y() const { return m_y; }
    //virtual void Normalize() { ::Normalize(m_x); }
    //virtual void Centralize() { ::Centralize(m_x); }

    string rowName( unsigned r ) const { return pNames->rowName(r); }
    string colName( unsigned c ) const { return pNames->colName(c); }
//    int rowId( int r ) const { return docs[r]; }    //{ return pNames->rowId(r); }
    int colId( unsigned c ) const { return pNames->colId(c); }
    unsigned colNumById( unsigned id ) const { return pNames->colNumById(id); }
};

class IRowSet : public INameResolver {
public:
    virtual bool next() =0;
    virtual void rewind() =0;
    virtual const SparseVector& xsparse() const =0;
    virtual YType y() const =0;
#ifdef POLYTOMOUS
    virtual bool ygood() const =0; //allowed label value
#else
    virtual const vector<bool>& groups() const =0;
    virtual bool group( unsigned g ) const =0;
    virtual unsigned ngroups() const =0;
#endif //POLYTOMOUS
    virtual ~IRowSet() {};
    virtual string currRowName() const=0;
    virtual unsigned n() const { throw logic_error("Unsupported feature: Size of rowset"); return unsigned(-1); }
};

class TopicRowSet : public IRowSet {
protected:
#ifdef POLYTOMOUS
    const vector<int>& classes;
#else
    const char* topic;
#endif //POLYTOMOUS
    const unsigned m_ngroups;
    const vector<int>& featSelect;
    bool cosineNormalize;
    //tfidf
    const vector<int>& wordRestrict;
    const vector<double>& globalIdf;
    TFMethod tfMethod;
    //current
    SparseVector m_sparse;
    YType m_y;
    vector<bool> m_groups;

public:
    TopicRowSet( const char* topic_,
#ifdef POLYTOMOUS
        const vector<int>& classes_,
#endif //POLYTOMOUS
        const vector<int>& featSelect_,
        TFMethod tfMethod_, const vector<int>& wordRestrict_, const vector<double>& globalIdf_,
        bool cosineNormalize_,
        unsigned ngroups_ )
    :
#ifdef POLYTOMOUS
      classes(classes_),
#else
      topic(topic_), 
#endif //POLYTOMOUS
      m_ngroups(ngroups_),
      featSelect(featSelect_),
      tfMethod(tfMethod_), wordRestrict(wordRestrict_), globalIdf(globalIdf_),
      cosineNormalize(cosineNormalize_)
    {
    }
    virtual bool next() =0;
    virtual void rewind() { throw logic_error("Rewind capability not supported for TopicRowSet"); }
    const SparseVector& xsparse() const { return m_sparse; }
    YType y() const { return m_y; }
    virtual const vector<bool>& groups() const { return m_groups; }
    virtual bool group( unsigned g ) const  { return m_groups.at(g); }
    virtual unsigned ngroups() const { return m_ngroups; }
#ifdef POLYTOMOUS
    bool ygood() const  { return 0<=m_y && m_y<classes.size(); };
    unsigned c() const { return classes.size(); }
    virtual int classId( unsigned c ) const { 
        if(c<classes.size()) return classes[c];
        else throw logic_error(string("Illegal class number: ")+int2string(c));
    }
    unsigned classNumById( int id ) const {
        vector<int>::const_iterator itc=lower_bound( classes.begin(), classes.end(), id );
        if( itc==classes.end() || *itc!=id )
            throw logic_error(string("Unknown class id: ")+int2string(id));
        return itc - classes.begin();
    }
    virtual const vector<int>& OddClasses() const =0;
    virtual unsigned NOddRows() const =0;
#endif //POLYTOMOUS

protected:
    void RestrictRecode( const SparseVector& tf ) //converts to m_sparse
    {
        m_sparse.clear();
        double norm = 0.0;
        //Log(8)<<"\nTEST Sparse vector  "<<tf;
        /**from sparse to 'featSelect'*/
        //intersection of sparse vector and terms
        //TODO: go through WordRestrict only when cosNorm; consider case when wordRestrict is all
        for( SparseVector::const_iterator ix = tf.begin(); ix!=tf.end(); ix++ )
        {
            //find in WordRestrict first
            vector<int>::const_iterator iWR = lower_bound(
                wordRestrict.begin(), 
                wordRestrict.end(), 
                ix->first );
            if( iWR!=wordRestrict.end() && *iWR==ix->first )
            {
                unsigned iTermWR = iWR - wordRestrict.begin();
                double tfidf = ix->second * globalIdf[iTermWR];
                norm += tfidf * tfidf;

                vector<int>::const_iterator iFS = lower_bound(
                    featSelect.begin(), 
                    featSelect.end(), 
                    ix->first );
                if( iFS!=featSelect.end() && *iFS==ix->first ) {
                    unsigned iTermFS = iFS - featSelect.begin();
                    m_sparse.push_back(SparseItem( featSelect[iTermFS], tfidf ));  //m_sparse[ featSelect[iTermFS] ] = tfidf
                    //Log(10)<<"  "<<featSelect[iTermFS]<<":"<<ix->second<<":"<<tfidf;
                }
            }
        }

        if( cosineNormalize && norm>0 ) {
            norm = 1.0/sqrt(norm);
            for( SparseVector::iterator ix=m_sparse.begin(); ix!=m_sparse.end(); ix++ )
                ix->second *= norm;
        }
    }
};

class SamplingRowSet : public IRowSet {
    IRowSet& source;
    const vector<int>& sampler;
    int from, to;
    bool in;
    bool InSample( unsigned i ) {
        bool inInterval =( from<=sampler[i] && sampler[i]<to );
        return in==inInterval;
    }
    void Init() {
        curr = -1;
        nRows = 0;
        for( unsigned i=0; i<sampler.size(); i++ )
            if( InSample(i) )
                nRows ++;
    }
    int curr;
    unsigned nRows;
public:
    SamplingRowSet( IRowSet& source_, const vector<int>& sampler_,
        int from_, int to_, bool in_)
        : source(source_), sampler(sampler_), from(from_), to(to_), in(in_)
    { Init(); }
    bool next() {
        while(true){
            //Log(10)<<"\nSampling curr "<<curr+1<<" randind "<<sampler[curr+1];
            if( !source.next() ) return false;
            if( InSample(++curr) ) return true;
        }
    }
    void rewind() {
        source.rewind();
        curr = -1;
    };
    unsigned n() const { return nRows; }
    //the rest just delegates
    const SparseVector& xsparse() const { return source.xsparse(); }
    YType y() const { return source.y(); }
    string currRowName() const { return source.currRowName(); }
    unsigned dim() const { return source.dim(); }
    string rowName( unsigned r ) const { return source.rowName(r); }
    string colName( unsigned c ) const { return source.colName(c); }
    int colId( unsigned c ) const { return source.colId(c); }
    unsigned colNumById( unsigned id ) const { return source.colNumById(id); }
#ifdef POLYTOMOUS
    bool ygood() const { return source.ygood(); }
    unsigned c() const { return source.c(); }
    int classId( unsigned c ) const { return source.classId(c); }
    unsigned classNumById( int c ) const { return source.classNumById(c); }
#else
    const vector<bool>& groups() const  { return source.groups(); }
    bool group( unsigned g ) const  { return source.group(g); }
    unsigned ngroups() const { return source.ngroups(); }
#endif //POLYTOMOUS
};

class StandardizedRowSet : public IRowSet {
    IRowSet& source;
    Vector means;
    Vector stddevs;
    SparseVector m_x;
public:
    StandardizedRowSet( IRowSet& source_, const Vector& means_, const Vector& stddevs_ )
        : source(source_), means(means_), stddevs(stddevs_)
    { }
    bool next() {  
        if( source.next() ) {
            m_x.clear();
            for( unsigned i=0; i<source.dim(); i++ ) 
            {
                double val, newval;
                //?? that does not work for some reason... int colid = source.colId( i );
                SparseVector::const_iterator xitr=source.xsparse().find( i ); //?colid
                if( xitr==source.xsparse().end() ) //not found - assume 0
                    val = 0;
                else
                    val = xitr->second;
                newval = val - means[i];
                if( stddevs[i] > 0 ) newval /= stddevs[i];
                m_x.insert( i, newval );   //? m_x.insert( colid,...
            }
            /*for( SparseVector::const_iterator xitr=source.xsparse().begin(); xitr!=source.xsparse().end(); xitr++ )
            {
                unsigned iFeat = source.colNumById( xitr->first );
                double val = xitr->second;
                m_x.insert( xitr->first, (val - means[iFeat])/stddevs[iFeat] );
                Log(5)<<"\n-- "<<xitr->first<<":"<<xitr->second<<" "<<source.colId( iFeat )<<" "<<val<<" "<<(val - means[iFeat])/stddevs[iFeat];
            }*/
            m_x.sort();
            //Log(10)<<"\nstd "<<m_x;
            return true;
        }
        else return false;
    }
    const SparseVector& xsparse() const { return m_x; }
    //the rest just delegates
    void rewind() { source.rewind(); };
    unsigned n() const { return source.n(); }
    YType y() const { return source.y(); }
    string currRowName() const { return source.currRowName(); }
    unsigned dim() const { return source.dim(); }
    string rowName( unsigned r ) const { return source.rowName(r); }
    string colName( unsigned c ) const { return source.colName(c); }
    int colId( unsigned c ) const { return source.colId(c); }
    unsigned colNumById( unsigned id ) const { return source.colNumById(id); }
#ifdef POLYTOMOUS
    bool ygood() const { return source.ygood(); }
    unsigned c() const { return source.c(); }
    int classId( unsigned c ) const { return source.classId(c); }
    unsigned classNumById( int c ) const { return source.classNumById(c); }
#else
    const vector<bool>& groups() const { return source.groups(); }
    bool group( unsigned g ) const  { return source.group(g); }
    unsigned ngroups() const { return source.ngroups(); }
#endif //POLYTOMOUS
};

#endif //Data_And_Labels_

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
