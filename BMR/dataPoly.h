#ifndef Data_And_Labels_Polytomous_case_
#define Data_And_Labels_Polytomous_case_

#include "data.h"

class PlainYRowSet : public TopicRowSet { //single topic - labels in the data file
    //to generate data
    istream *p_ifs;
    bool m_readFromStdin;
    //external scores
    class IScores* scores;
    //current
    bool valid;
    unsigned currRow1;
    vector<int> oddclasses;
    unsigned oddrows;
public:
    ~PlainYRowSet() { 
        delete scores; 
        if(!m_readFromStdin) delete p_ifs;
    }
    PlainYRowSet( const char* dataFName,
        const vector<int>& classes_,
        const vector<int>& featSelect_,
        TFMethod tfMethod_, const vector<int>& wordRestrict_, const vector<double>& globalIdf_,
        class IScores* scores_, bool cosineNormalize_ )
    : TopicRowSet( "<class>",  classes_, featSelect_, tfMethod_, wordRestrict_, globalIdf_,
        cosineNormalize_, 0/*groups not supported*/ ),
    scores(scores_),
    oddrows(0)
    {
        if( readFromStdin(dataFName) ) {
            p_ifs = &cin;
            m_readFromStdin = true;
        }else{
            p_ifs = new ifstream(dataFName);
            m_readFromStdin = false;
        }
        currRow1 = 0;
        valid = true;
    }

    bool next() {
        vector<SparseItem> tf;
        valid = ReadRowY( tf ); //sets m_y
        if( valid ) {
            // reduce to selected 'terms' and multiply by idf; compute norm;
            RestrictRecode( SparseVector( tf ) );

            //external scores // !! do not sit well with cosnorm!
            //scores->next();
            //for( unsigned i=0; i<scores->n(); i++ )
              //  m_sparse[ pNames->colId( featSelect.size()+i ) ] = scores->val(i);
        }
        return valid;
    }
    void rewind() {  throw logic_error("Rewind capability not supported for TopicRowSet"); }
        /*p_ifs->seekg(0); //rewind
        currRow1 = 0;
        valid = true; }*/

    //INameResolver
    unsigned dim() const { return featSelect.size(); }
    virtual string rowName( unsigned r ) const { 
        return int2string(r);
    }
//    int rowId( int r ) const { return pNames->rowId(r); }  //!! not the id in Oracle
    string colName( unsigned c ) const { return int2string(featSelect[c]); }
    int colId( unsigned c ) const { return featSelect[c]; }
    unsigned colNumById( unsigned id ) const {
        vector<int>::const_iterator iterm=lower_bound( featSelect.begin(), featSelect.end(), (int)id );
        if( iterm==featSelect.end() || *iterm!=id )
            throw logic_error("PlainNameResolver: Term id not in the term set");
        return iterm - featSelect.begin();
    }
    // IRowSet
    string currRowName() const { return int2string(currRow1); }
    int currRowId() const { return currRow1; }
    const vector<int>& OddClasses() const { return oddclasses; }
    unsigned NOddRows() const { return oddrows; }
    virtual int classId( unsigned c ) const { 
        if(c<classes.size()) return classes[c];
        else if(c<classes.size()+oddclasses.size()) return oddclasses[c-classes.size()];
        else throw logic_error(string("Illegal class number: ")+int2string(c));
    }
private:
    bool ReadRowY( vector<SparseItem>& tf ) {
        string buf;
        while(1) {//look for not-a-comment line
            getline( *p_ifs, buf );
            if( p_ifs->fail() )
                return false;
            else
                if( buf[0]=='#' )
                    continue; //look for not-a-comment line
            else {
                istringstream rowbuf( buf );
                int i; double d; char delim; int y;
                rowbuf>>y;
                if( rowbuf.fail() )  
                    return ReadRowY( tf ); //empty line - recurse to read the next one

                vector<int>::const_iterator itc=lower_bound( classes.begin(), classes.end(), y );
                if( itc!=classes.end() && *itc==y )
                    m_y = itc - classes.begin();
                else{ // class unknown
                    oddrows++;
                    vector<int>::const_iterator itrodd=locate_in_odd( y );
                    if( itrodd==oddclasses.end() ) { //first time met
                        oddclasses.push_back( y );
                        m_y = classes.size() + oddclasses.size()-1;
                    }
                    else
                        m_y = classes.size() +( itrodd - oddclasses.begin() );
                }

                while( rowbuf>>i>>delim>>d ) { //rowbuf.good()
                    if( delim!=':' )
                        throw runtime_error(string("Wrong delimiter, should be semicolon, is ")+delim);
                    if( !rowbuf.fail() )
                        tf.push_back(SparseItem( i, TFWeight( d, tfMethod ) )); //tf[i] = TFWeight( d, tfMethod )
                }
                if( rowbuf.eof() ) { //the only legal reason to end the loop
                    currRow1 ++;
                    return true;
                }
                else
                    throw runtime_error(string("Corrupt test file line: ") + buf);
            }
        } //look for not-a-comment line
    }
    vector<int>::const_iterator locate_in_odd( int y ) const {
        vector<int>::const_iterator itrodd;
        for( itrodd=oddclasses.begin(); itrodd!=oddclasses.end(); itrodd++ )
            if( *itrodd==y )
                break;
        return itrodd;
    }
};

class RowSetMem : public IRowSet { //rowset explicit in memory
    const vector<SparseVector>& m_x;
    const vector<YType>& m_y;
    vector<int> featSelect;

    INameResolver* pNames;
    bool ownNameResolver;

    //current
    bool valid;
    unsigned currow;
    SparseVector currx;

    //external scores
    class IScores* scores;

public:
    const vector<YType>& Y() const { return m_y; } //not to override
    //ctors
    RowSetMem(  const vector<SparseVector>& x_, const vector<YType>& y_,
        const vector<int>& featSelect_, INameResolver* pNames_,
        class IScores* scores_ )
        : m_x(x_), m_y(y_), featSelect(featSelect_),
        pNames(pNames_),
        scores(scores_)
    {
        if( scores->n()>0 )
            throw logic_error("External scores not supported with plain data file!");
        valid = false;
        ownNameResolver = true;

        //DetectZeroClassVars( c(), m_y, featSelect, m_x );
    }
    ~RowSetMem() { 
        if(ownNameResolver) delete pNames; 
        delete scores; }
    unsigned n() const { return m_x.size(); }
    bool next() {
        if(valid) currow++;
        else currow = 0; //rewind
        valid = currow < m_x.size();
        if( !valid )  return false;   //--->>--

        //restrict source data to featSelect
        //TODO: skip if featSelect is all features
        vector<SparseItem> tf; //currx.clear();
        double norm = 0;
        vector<int>::const_iterator fsBeg = featSelect.begin();
        vector<int>::const_iterator fsEnd = featSelect.end();
        for( SparseVector::const_iterator ix=m_x[currow].begin(); ix!=m_x[currow].end(); ix++ )
        {
            //locate in featSelect
            vector<int>::const_iterator i=lower_bound( fsBeg, fsEnd, ix->first );
            if( i==fsEnd )
                break; //term beyond the last in 'featSelect', so are all the rest - sorted
            else{
                if( *i==ix->first ) {//othw term missing from 'featSelect'
                     tf.push_back(SparseItem( ix->first, ix->second )); //currx[ix->first] = ix->second;
                }
                fsBeg = i;
            }
        }

        //external scores
        scores->next();
        for( unsigned i=0; i<scores->n(); i++ ) {
            tf.push_back(SparseItem( pNames->colId( featSelect.size()+i ), scores->val(i) )); //currx[ pNames->colId( featSelect.size()+i ) ] = scores->val(i);
        }

        currx = SparseVector( tf );
        //Log(10)<<"\nTrain sparse vector "<<currx;

        return true;
    }
    void rewind() { 
        valid=false;
        scores->rewind();
    }
    const SparseVector& xsparse() const { return currx; }
    YType y() const { return m_y[currow]; }
    bool ygood() const { return true; } //sic!
    //INameResolver
    unsigned dim() const { return featSelect.size()+scores->n(); }
    string rowName( unsigned r ) const { return pNames->rowName(r); }
    string colName( unsigned c ) const { return pNames->colName(c); }
    int colId( unsigned c ) const { return pNames->colId(c); }
    unsigned colNumById( unsigned id ) const { return pNames->colNumById(id); }
    // IRowSet
    string currRowName() const { return rowName(currow); }
    unsigned c() const { return pNames->c(); }
    int classId( unsigned c ) const { return pNames->classId(c); }
    unsigned classNumById( int c ) const { return pNames->classNumById(c); }
    vector< vector<bool> > allzeroes() const //{ return  m_allzeroes; }
    {
        unsigned d = featSelect.size();
        vector< vector<bool> >  zeroes( d, vector<bool>( c(), true ) );
        vector<bool> nonneg( d, true );
        for( unsigned i=0; i<m_x.size(); i++ )
            for( SparseVector::const_iterator iTF=m_x.at(i).begin(); iTF!=m_x.at(i).end(); iTF++ ) 
            if( iTF->second!=0 )
            {
                vector<int>::const_iterator v=lower_bound( featSelect.begin(), featSelect.end(), iTF->first );
                if( v!=featSelect.end() ) {
                    zeroes.at( v-featSelect.begin() ).at( m_y.at(i) ) = false;
                    if( 0.0 > iTF->second )
                        nonneg [v-featSelect.begin()] = false;
                }
            }

        // remove those which can be negative
        for( unsigned j=0; j<d; j++ )
            if( !nonneg[j] )
                zeroes.at(j) = vector<bool>( c(), false );

        // report
        for( unsigned k=0; k<c(); k++ ) {
            Log(3)<<"\nZero vars for class "<<k<<":";
            unsigned nzeroes=0;
            for( unsigned j=0; j<d; j++ )
                if( zeroes.at(j).at(k) ) {
                    nzeroes++; 
                    Log(7)<<" "<<featSelect[j];
                }
            Log(3)<<" total "<<nzeroes;
        }
        return zeroes;
    }
};

#endif //Data_And_Labels_Polytomous_case_

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
