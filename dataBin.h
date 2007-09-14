#ifndef Data_And_Labels_Binary_case_
#define Data_And_Labels_Binary_case_

#include "data.h"

#ifdef USE_LEMUR

#include "Oracle.hpp" //Filter

class PlainTopicRowSet : public TopicRowSet {
    //to generate data
    istream *p_ifs;
    bool m_readFromStdin;
    PlainOracle* qrel;
    //external scores
    class IScores* scores;
    //for INameResolver
    vector<string> docNames;
    //current
    bool valid;
    unsigned currRow1;
    string currRName;
public:
    ~PlainTopicRowSet() {
        delete scores; 
        if(!m_readFromStdin) delete p_ifs;
    }
    PlainTopicRowSet( const char* topic_, const char* dataFName, PlainOracle* qrel_,
        const vector<int>& featSelect_,
        //const vector<string>& docNames_,
        TFMethod tfMethod_, const vector<int>& wordRestrict_, const vector<double>& globalIdf_,
        class IScores* scores_, bool cosineNormalize_ )
    : TopicRowSet( topic_, featSelect_, tfMethod_, wordRestrict_, globalIdf_,
        cosineNormalize_, 0 ), //groups not supported
    //ifs(dataFName),
    qrel(qrel_),  
    //docNames(docNames_),
    scores(scores_)
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
        //pNames = new CombinedNameResolver(featSelect, docs, ind, scores );
    }

    bool next() {
        vector<SparseItem> tf;
        valid = ReadRow( tf, currRName, tfMethod );
        if( valid ) {

            docNames.push_back(currRName);

            m_y = (REL==qrel->ask( topic, currRName.c_str() )); //oracleNRelDefault

            // reduce to selected 'terms' and multiply by idf; compute norm;
            RestrictRecode( SparseVector( tf ) );

            //external scores // !! do not sit well with cosnorm!
            //scores->next();
            //for( unsigned i=0; i<scores->n(); i++ )
              //  m_sparse[ pNames->colId( featSelect.size()+i ) ] = scores->val(i);
        }
        return valid;
    }
    virtual void rewind() { 
        throw logic_error("PlainTopicRowSet: Cannot rewind");
        /*p_ifs->seekg(0); //rewind
        currRow1 = 0;
        valid = true; 
        docNames.clear();   */
    }

    //INameResolver
    unsigned dim() const { return featSelect.size(); }
    virtual string rowName( unsigned r ) const { 
        if( r>=docNames.size() )
            throw runtime_error(string("Name unknown for row number ")+int2string(r));
        return docNames[r];
    }
//    int rowId( int r ) const { return pNames->rowId(r); }  //!! not the id in Oracle
    string colName( unsigned c ) const { return int2string(featSelect[c]); }
    int colId( unsigned c ) const { return featSelect[c]; }
    virtual unsigned colNumById( unsigned id ) const {
        vector<int>::const_iterator iterm=lower_bound( featSelect.begin(), featSelect.end(), (int)id );
        if( iterm==featSelect.end() || *iterm!=id )
            throw logic_error("PlainNameResolver: Term id not in the term set");
        return iterm - featSelect.begin();
    }
    // IRowSet
    string currRowName() const { return currRName; }
private:
    bool ReadRow( vector<SparseItem>& tf, string& rowName, TFMethod tfMethod ) {
        string buf;
        getline( *p_ifs, buf );
            //Log(10)<<"\nrow "<<currRow1+1<<" `"<<buf<<"`";//.str()
        if( p_ifs->fail() )
            return false;
        else {
            istringstream rowbuf( buf );
            int i; double d; char delim;
            rowbuf>>rowName;
            while( rowbuf>>i>>delim>>d ) { //rowbuf.good() 
                if( delim!=':' )
                    throw runtime_error(string("Wrong delimiter, should be semicolon, is ")+delim);
                if( !rowbuf.fail() )
                    tf.push_back(SparseItem( i, TFWeight( d, tfMethod ) )); //tf[i] = TFWeight( d, tfMethod )
            }
            //Log(12)<<"\nparsed "<<rowName<<" - "<<tf;
            if( rowbuf.eof() ) {//the only legal reason to end the loop 
                currRow1 ++;
                return true;
            }
            else
                throw runtime_error(string("Corrupt test file line: ") + buf);
        }
    }
};
#endif //USE_LEMUR

class PlainYRowSet : public TopicRowSet { //single topic - labels in the data file
    //to generate data
    istream *p_ifs;
    bool m_readFromStdin;
    bool rowIdMode;
    //external scores
    class IScores* scores;
    //current
    bool valid;
    unsigned currRow1;
    string m_currRowName;
public:
    ~PlainYRowSet() { 
        delete scores; 
        if(!m_readFromStdin) delete p_ifs;
    }
    PlainYRowSet( const char* topic_, const char* dataFName, bool rowIdMode_,
        const vector<int>& featSelect_,
        TFMethod tfMethod_, const vector<int>& wordRestrict_, const vector<double>& globalIdf_,
        class IScores* scores_, bool cosineNormalize_, unsigned ngroups_ )
    : TopicRowSet( topic_, featSelect_, tfMethod_, wordRestrict_, globalIdf_,
        cosineNormalize_, ngroups_ ),
    //ifs(dataFName),
    rowIdMode(rowIdMode_),
    scores(scores_)
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
    virtual void rewind() { 
        p_ifs->seekg(0); //rewind
        currRow1 = 0;
        valid = true; }

    //INameResolver
    unsigned dim() const { return featSelect.size(); }
    virtual string rowName( unsigned r ) const { 
        return int2string(r);
    }
    //    int rowId( int r ) const { return pNames->rowId(r); }  //!! not the id in Oracle
    string colName( unsigned c ) const { return int2string(featSelect[c]); }
    int colId( unsigned c ) const { return featSelect[c]; }
    virtual unsigned colNumById( unsigned id ) const {
        vector<int>::const_iterator iterm=lower_bound( featSelect.begin(), featSelect.end(), (int)id );
        if( iterm==featSelect.end() || *iterm!=id )
            throw logic_error("PlainNameResolver: Term id not in the term set");
        return iterm - featSelect.begin();
    }
    // IRowSet
    string currRowName() const { 
        if(rowIdMode) return m_currRowName; 
        else return int2string(currRow1);
    }
private:
    bool ReadRowY( vector<SparseItem>& tf ) {
        string buf;
        m_currRowName = ""; //for rowId
        while(1) {//look for not-a-comment line
            getline( *p_ifs, buf );
                //Log(10)<<"\nrow "<<currRow1+1<<" `"<<buf<<"`";//.str()
            if( p_ifs->fail() )
                return false;
            else if( buf[0]=='#' ) {
               if( rowIdMode ) {
                    istringstream rowbuf( buf );
                    string hash, uidkw, id;
                    rowbuf>>hash>>uidkw>>id;
                    if( 0==stricmp( uidkw.c_str(), "UID:" ) )
                        m_currRowName = id;
                    else
                        continue;
                }
                else
                   continue; //look for not-a-comment line
            }
            else { //essential data row

                istringstream rowbuf( buf );
                int i; double d; char delim; int y;
                rowbuf>>y;
                if( rowbuf.fail() )  
                    return ReadRowY( tf ); //empty line - recurse to read the next one
                if( rowIdMode && 0==m_currRowName.size() )
                    throw runtime_error("Row Id absent ");
                m_y = (1==y); //either 1/0 or 1/-1

                vector<bool> groups;
                int ig;
                for( unsigned g=0; g<m_ngroups; g++ ) {
                    rowbuf>>ig;
                    if( 1==ig ) groups.push_back(true);
                    else if( 0==ig ) groups.push_back(false);
                    else throw runtime_error(string("Wrong group indicator in line ")+int2string(currRow1));
                }
                m_groups = groups;

                while( rowbuf>>i>>delim>>d ) { //rowbuf.good()
                    if( i<=0 )
                        throw runtime_error(string("Non-positive variable id in row ")+int2string(currRow1));
                    if( delim!=':' )
                        throw runtime_error(string("Wrong delimiter, should be semicolon, is ")+delim);
                    if( !rowbuf.fail() )
                        tf.push_back(SparseItem( i, TFWeight( d, tfMethod ) ));
                    //cout<<endl<<"i delim d "<<i<<" "<<delim<<" "<<d;cout.flush();
                }
                if( rowbuf.eof() ) { //the only legal reason to end the loop
                    currRow1 ++;
                    return true;
                }
                else
                    throw runtime_error(string("Corrupt test file line: ") + buf);
            }
        }//look for not-a-comment line
    }
};

template<class TRowHandle> class IDirectAccessRowSet : public IRowSet {
public:
    virtual TRowHandle RowHandle() const =0;
    virtual YType y(TRowHandle r) const =0;
    virtual const SparseVector& xsparse(TRowHandle r) const =0;
    virtual bool group(TRowHandle r, unsigned g) const =0;
    virtual string name(TRowHandle r) const =0;
};

class RowMem { 
    friend class RowSetMem;
    unsigned i0;
//public:
    explicit RowMem( unsigned i0_ ) : i0(i0_) {}
    operator unsigned() const { return i0; }
};
class RowSetMem : public IDirectAccessRowSet<RowMem> { //rowset explicit in memory
    const vector<SparseVector>& m_x;
    const vector< vector<bool> >& m_groups;
    const vector<YType>& m_y;
    vector<int> featSelect;

    INameResolver* pNames;
    bool ownNameResolver;
    const unsigned m_ngroups;

    //current
    bool valid;
    unsigned currow;
    SparseVector currx;

    //external scores
    class IScores* scores;

public:
    // IDirectAccessRowSet
    RowMem RowHandle() const {return RowMem(currow);}
    YType y(RowMem r) const {return m_y[unsigned(r)];}
    const SparseVector& xsparse(RowMem r) const {return m_x.at(r);}
    bool group(RowMem r, unsigned g) const {return m_groups.at(r).at(g);}
    string name(RowMem r) const  {return pNames->rowName(r);}

    //ctors
#ifdef USE_LEMUR
    RowSetMem(  const vector<SparseVector>& x_, const vector<YType>& y_,
        const vector<int>& featSelect_, const vector<int>& docs_, Index* index,
        class IScores* scores_ );
#endif
    RowSetMem(  const vector<SparseVector>& x_, 
        unsigned ngroups_, const vector< vector<bool> >& groups_,
        const vector<YType>& y_,
        const vector<int>& featSelect_, INameResolver* pNames_,
        class IScores* scores_ )
        : m_x(x_), 
        m_ngroups(ngroups_), m_groups(groups_),
        m_y(y_),
        featSelect(featSelect_), pNames(pNames_),
        scores(scores_)
    {
        if( scores->n()>0 )
            throw logic_error("External scores not supported with plain data file!");
        valid = false;
        ownNameResolver = true;
    }
    ~RowSetMem() { 
        if(ownNameResolver) delete pNames; 
        delete scores; }
    unsigned n() const { return m_x.size(); }
    unsigned npos() const { return ntrue(m_y); }
    const vector<bool>& groups() const { return m_groups.at(currow); }
    bool group( unsigned g ) const  { return m_groups.at(currow).at(g); }
    unsigned ngroups() const { return m_ngroups; }
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
    bool y() const { return m_y[currow]; }
    //INameResolver
    unsigned dim() const { return featSelect.size()+scores->n(); }
    string rowName( unsigned r ) const { return pNames->rowName(r); }
    string colName( unsigned c ) const { return pNames->colName(c); }
    int colId( unsigned c ) const { return pNames->colId(c); }
    unsigned colNumById( unsigned id ) const { return pNames->colNumById(id); }
    // IRowSet
    string currRowName() const { return rowName(currow); }
};

#endif //Data_And_Labels_Binary_case_

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
