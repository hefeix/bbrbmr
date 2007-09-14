#ifndef Text_Data_Lemur_Format_
#define Text_Data_Lemur_Format_

#include <map>
#include <ostream>
using std::map;

#ifdef USE_LEMUR
# include "Index.hpp" //Lemur
# include "Oracle.hpp" //Filter
#endif //USE_LEMUR
#include "TFIDFParamManager.h"
#include "dataBin.h"
#include "DataFactory.h"

class LemurNameResolver : public INameResolver {
protected:
    Index* ind;
    const vector<int>& terms;
    const vector<int>& docs;
public:
    LemurNameResolver(const vector<int>& terms_, const vector<int>& docs_, Index* ind_)
        : terms(terms_), docs(docs_), ind(ind_) {}
    virtual string rowName( unsigned r ) const { return ind->document( docs[r] ); }
    virtual string colName( unsigned c ) const { return ind->term( terms[c] ); }
    unsigned dim() const { return terms.size(); }
//    virtual int rowId( int r ) const { return docs[r]; }
    virtual int colId( unsigned c ) const { return terms[c]; }
    virtual unsigned colNumById( unsigned id ) const {
        vector<int>::const_iterator iterm=lower_bound( terms.begin(), terms.end(), (int)id );
        if( iterm==terms.end() || *iterm!=id )
            throw logic_error("LemurNameResolver: Term id not in the term set");
        return iterm - terms.begin();
    }
};

class CombinedNameResolver : public LemurNameResolver {
    const class IScores* scores;
    unsigned scoreId;
public:
    CombinedNameResolver(const vector<int>& terms_, const vector<int>& docs_, Index* ind_, 
        const class IScores* scores_ )
        : LemurNameResolver(terms_, docs_, ind_),  scores(scores_)  
    {
        scoreId = ind->termCountUnique() + 1;
    }
    unsigned dim() const { return LemurNameResolver::dim() + scores->n(); }
    string colName( unsigned c ) const { 
        if( c<terms.size() )
            return ind->term( terms[c] ); 
        else {  //return scores->name(i);
            std::ostringstream buf; buf<<"<score_" << c-terms.size()+1 <<">";
            return buf.str();
        }
    }
    int colId( unsigned c ) const { 
        if( c<terms.size() )
            return terms[c];
        else
            return scoreId + c-terms.size();
    }
    unsigned colNumById( unsigned id ) const {
        if( id>=scoreId )
            return id-scoreId+terms.size();
        else
            return LemurNameResolver::colNumById( id );
    }
};

class LemurTopicRowSet : public TopicRowSet {
    INameResolver* pNames;
    //to generate data
    Index *ind;
    Oracle* qrel;
    vector<int> docs;
    //current
    bool valid;
    unsigned currow;
    //external scores
    class IScores* scores;

public:
    LemurTopicRowSet( const char* topic_,
        const vector<int>& featSelect_, const vector<int>& docs_, Index* ind_, Oracle* qrel_,
        TFMethod tfMethod_, const vector<int>& wordRestrict_, const vector<double>& globalIdf_,
        class IScores* scores_, bool cosineNormalize_ )
    : TopicRowSet( topic_, featSelect_, tfMethod_, wordRestrict_, globalIdf_,
        cosineNormalize_, 0/*groups not supported*/),
    docs(docs_), ind(ind_), qrel(qrel_),  
    scores(scores_)
    {
        valid = false;
        pNames = new CombinedNameResolver(featSelect, docs, ind, scores );
    }
    ~LemurTopicRowSet() { delete pNames; delete scores; }
    bool next() {
        if(valid) currow++;
        else currow = 0; //rewind
        valid = currow<docs.size();
        if( !valid )
            return false;

        m_y = (REL==qrel->ask( topic, docs[currow] )); //oracleNRelDefault

        SparseVector tf = docToTFvector( docs[currow], ind, tfMethod );

        // reduce to selected 'terms' and multiply by idf; compute norm;
        RestrictRecode( tf );

        //external scores
        // !! do not sit well with cosnorm!
        scores->next();
        for( unsigned i=0; i<scores->n(); i++ ) {
            m_sparse.insert( pNames->colId( featSelect.size()+i ), scores->val(i) ); //m_sparse[ pNames->colId( featSelect.size()+i ) ] = scores->val(i)
            //that would be a mistake: norm computes on 'wordRestrict' set
            //-- norm += scores->val(i) * scores->val(i);
        }
        m_sparse.sort();
        Log(10)<<"\n te "<<m_sparse;

        return true;
    }
    virtual void rewind() { valid = false; }
    //INameResolver
    unsigned dim() const { return pNames->dim(); }
    string rowName( unsigned r ) const { return pNames->rowName(r); }
    string colName( unsigned c ) const { return pNames->colName(c); }
//    int rowId( int r ) const { return pNames->rowId(r); }
    int colId( unsigned c ) const { return pNames->colId(c); }
    unsigned colNumById( unsigned id ) const { return pNames->colNumById(id); }
    // IRowSet
    string currRowName() const { return ind->document( docs[currow] ); }  //rowName(currow); }
//    int currRowId() const { return rowId(currow); }
};

class LemurDataFactory : public DataFactory {
    Index *allInd;
    Oracle *trainQrel;
    Oracle *testQrel;
    vector<int> trainDocs, testDocs;
    LemurDocIndex docIndexWrapper;
public:
    LemurDataFactory( Index *ind_, //single index, separate qrels, separate doc lists
        const char* trainDocsFile,
        const char* testDocsFile,
        const char* trainQrelFile,
        const char* testQrelFile,
        const char* bagOfWordsFile,
        TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_,
        const string idfDocsFile );
    LemurDataFactory( //separate train index, single qrel, doc list for test only
        Index *ind_, 
        Index *trainInd, 
        const char* testDocsFile,
        const char* qrelFile,
        const char* bagOfWordsFile,
        TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_,
        const string idfDocsFile );
    LemurDataFactory( //test only
        Index *ind_, 
        const char* testDocsFile,
        const char* qrelFile,
        TFMethod tfMethod_,
        IDFMethod idfMethod_,
        bool cosineNormalize_,
        vector<int> wordRestrict = vector<int>(),
        vector<double> idf  = vector<double>() );
    DenseData* NewTrainData( const char* topic, 
        const FeatSelectParameter& featSelectParameter,
        const IndPriors& priorTerms );
    RowSetMem* NewTrainRowSet( const char* topic, 
        const FeatSelectParameter& featSelectParameter,
        const IndPriors& priorTerms,
        const class ExtScoreParam& extScoreParam );
    TopicRowSet* NewTestRowSet( const char* topic, const class ExtScoreParam& extScoreParam );
    TopicRowSet* NewTestRowSet( const char* topic, const vector<int>& featSelect_, //test only
                                              const class ExtScoreParam& extScoreParam );
    TopicRowSet* NewTrainPopRowSet( const char* topic, 
				    //const vector<int>& featSelect_, 
				    const class ExtScoreParam& extScoreParam,
				    const string & idfDocsFile); // VM
    ~LemurDataFactory() {
        delete trainQrel;
        if( testQrel!=trainQrel )
            delete testQrel;
    }
private:
    void YTrain( const char* topic ) {
        yTrain = BoolVector( false, Mtrain.size() );
        for( unsigned i=0; i<Mtrain.size(); i++ )
            yTrain[i] =( REL==trainQrel->ask( topic, trainDocs[i] ) ); //oracleNRelDefault
    }
    void SetWordRestrict( const char* bagOfWordsFile, Index *trainInd );
    void ComputeIDF( Index *trainInd, const string idfDocsFile );
    vector<int> CollectFromFile( const char* docsFile, Index* ind );
    vector<int> CollectFromIndex( Index* indOfDocs, Index* ind );
    void Zscores();
};

#endif //Text_Data_Lemur_Format_

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
