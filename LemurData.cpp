#include <stdexcept>
#include "IndexManager.hpp"
#include "LemurData.h" 
#include "ExtScore.h" 
#include "logging.h" 

SparseVector docToTFvector( int docId, Index *ind, TFMethod tf ) {
    SparseVector tfv;
    TermInfoList* terms = ind->termInfoList(docId);
    terms->startIteration();

    //Log(12)<<endl<<"Doc "<<docId<<" "<<ind->document(docId)<<" Terms id/count/tfMethod ";
    while (terms->hasMore())
    {
        TermInfo* term = terms->nextEntry();
        //Log(12)<<" "<<ind->term(term->id())<<" "<<term->id()<<"/"<<term->count()<<"/"<<tf;
        tfv.insert( term->id(), TFWeight( term->count(), tf ) );  //tfv[term->id()] = TFWeight( term->count(), tf )
    }
    delete terms;
    tfv.sort();
    return tfv;
}

vector<SparseVector> MakeSparseMatrix( const vector<int>& docs, Index *ind, TFMethod tfMethod ) {
    vector<SparseVector> MSparse( docs.size() );
    for( unsigned r=0; r<docs.size(); r++ )
        MSparse[r] = docToTFvector( docs[r], ind, tfMethod );
    return MSparse;
}

//ctor
RowSetMem::RowSetMem(  const vector<SparseVector>& x_, const vector<bool>& y_,
    const vector<int>& featSelect_, const vector<int>& docs_, Index* index,
    class IScores* scores_ )
    : m_x(x_), m_y(y_), featSelect(featSelect_),  
    m_ngroups(0), m_groups( vector< vector<bool> >() ),//groups not supported
    scores(scores_)
{
    valid = false;
    pNames = new CombinedNameResolver(featSelect, docs_, index, scores );
        Log(10)<<"\nCombinedNameResolver, #scores="<<scores->n();
    ownNameResolver = true;
}

DenseData::DenseData( 
              const vector<SparseVector>& x, const vector<bool> y,
              const vector<int>& terms, const vector<int>& docs_, Index* ind )
              : docs(docs_)
{
    pNames = new LemurNameResolver(terms, docs_, ind );

    m_dim = terms.size();
    m_nr = x.size();
    if( y.size()!=m_nr )
        throw DimensionConflict(__FILE__,__LINE__);
    m_x = Matrix( m_nr, m_dim, 0.0 );
    m_y = y;

    for( int r=0; r<m_nr; r++ ) //--rows--
    {
        //TODO: 'terms' should be sorted
        //intersection of x[r] and terms
        SparseVector::const_iterator ix = x[r].begin();
        vector<int>::const_iterator iInd = terms.begin();
        while( ix!=x[r].end() && iInd!=terms.end() ) { //merge
            if( ix->first < *iInd )          ix ++;
            else if( *iInd < ix->first )     iInd ++;
            else { //equal values
                m_x.val( r, iInd-terms.begin() ) = ix->second; // * topicIDF[ iInd-terms.begin() ];
                ix ++;
                iInd ++;
            }
        }
    } //--rows--
}

/*DenseData::DenseData( 
              const vector<SparseVector>& x, const BoolVector y,
              const vector<int>& terms, const vector<int>& docs_, Index* ind )
              : docs(docs_)
{
    pNames = new LemurNameResolver(terms, docs_, ind );

    m_dim = terms.size();
    m_nr = x.size();
    if( y.size()!=m_nr )
        throw DimensionConflict(__FILE__,__LINE__);
    m_x = Matrix( m_nr, m_dim, 0.0 );
    m_y = y;

    for( int r=0; r<m_nr; r++ ) //--rows--
    {
        //TODO: 'terms' should be sorted
        //intersection of x[r] and terms
        SparseVector::const_iterator ix = x[r].begin();
        vector<int>::const_iterator iInd = terms.begin();
        while( ix!=x[r].end() && iInd!=terms.end() ) { //merge
            if( ix->first < *iInd )          ix ++;
            else if( *iInd < ix->first )     iInd ++;
            else { //equal values
                m_x.val( r, iInd-terms.begin() ) = ix->second; // * topicIDF[ iInd-terms.begin() ];
                ix ++;
                iInd ++;
            }
        }
    } //--rows--
}*/
#ifdef USE_LEMUR
LemurDataFactory ::LemurDataFactory( Index *ind_, //single index, separate qrels, separate doc lists
    const char* trainDocsFile,
    const char* testDocsFile,
    const char* trainQrelFile,
    const char* testQrelFile,
    const char* bagOfWordsFile,
    TFMethod tfMethod_,
    IDFMethod idfMethod_,
    bool cosineNormalize_,
    const string idfDocsFile )
: allInd(ind_), docIndexWrapper(*allInd),
DataFactory( tfMethod_, idfMethod_, cosineNormalize_ )
{
    trainQrel = new Oracle( trainQrelFile, docIndexWrapper, true);  //( trainQrelFile, *allInd, true)
    //Log(8)<<"\nQrel for training, Time "<<Log.time();
    testQrel = new Oracle( testQrelFile, docIndexWrapper, false);    //( testQrelFile, *allInd, true)
    //Log(8)<<"\nQrel for testing, Time "<<Log.time();

    // apriori word restriction
    SetWordRestrict( bagOfWordsFile, 0 );
    //Log(8)<<"\nSetWordRestrict, Time "<<Log.time();

    //collect doc ids from index
    trainDocs = CollectFromFile(trainDocsFile,allInd);
    //Log(8)<<"\nCollectFromFile trainDocs, Time "<<Log.time();
    //form a sparse matrix
    Mtrain = MakeSparseMatrix( trainDocs, allInd, tfMethod );
    //Log(8)<<"\nMakeSparseMatrix, Time "<<Log.time();

    ComputeIDF( 0, idfDocsFile );
    //Log(8)<<"\nComputeIDF, Time "<<Log.time();
    ApplyIDFtoTrain();
    //Log(8)<<"\nApplyIDFtoTrain, Time "<<Log.time();

    //collect doc ids from file
    testDocs = CollectFromFile(testDocsFile,allInd);
    //Log(8)<<"\nCollectFromFile testDocs, Time "<<Log.time();
}
LemurDataFactory ::LemurDataFactory( //separate train index, single qrel, doc list for test only
    Index *ind_, 
    Index *trainInd, 
    const char* testDocsFile,
    const char* qrelFile,
    const char* bagOfWordsFile,
    TFMethod tfMethod_,
    IDFMethod idfMethod_,
    bool cosineNormalize_,
    const string idfDocsFile
    )
: allInd(ind_), docIndexWrapper(*allInd),
DataFactory( tfMethod_, idfMethod_, cosineNormalize_ )
{
    trainQrel = new Oracle( qrelFile, docIndexWrapper, true);   //( qrelFile, *allInd, true)
    testQrel = trainQrel;

    // apriori word restriction
    SetWordRestrict( bagOfWordsFile, trainInd );

    //collect doc ids from Lemur index
    trainDocs = CollectFromIndex(trainInd,allInd);
    //form a sparse matrix
    Mtrain = MakeSparseMatrix( trainDocs, allInd, tfMethod );

    ComputeIDF( trainInd, idfDocsFile );
    ApplyIDFtoTrain();

    //collect doc ids from file
    testDocs = CollectFromFile(testDocsFile,allInd);
    Log(4)<<"\nTesting docs list, Time "<<Log.time()<<endl;
}
LemurDataFactory::LemurDataFactory( //test only
    Index *ind_, 
    const char* testDocsFile,
    const char* qrelFile,
    TFMethod tfMethod_,
    IDFMethod idfMethod_,
    bool cosineNormalize_,
    vector<int> wordRestrict_,
    vector<double> idf_
    )
: allInd(ind_), docIndexWrapper(*allInd),
DataFactory( tfMethod_, idfMethod_, cosineNormalize_, wordRestrict_, idf_ )
{
    trainQrel = 0;
    testQrel = new Oracle( qrelFile, docIndexWrapper, true);

    //collect doc ids from file
    testDocs = CollectFromFile(testDocsFile,allInd);
    Log(4)<<"\nTesting docs list, Time "<<Log.time()<<endl;
}

#if !defined(TEST_ONLY)
DenseData* LemurDataFactory:: NewTrainData( const char* topic, 
    const FeatSelectParameter& featSelectParameter,
    const IndPriors& priorTerms )
{
    SetTopic( topic, featSelectParameter, priorTerms );
    return new DenseData( Mtrain, yTrain, featSelect, trainDocs, allInd );
}
RowSetMem* LemurDataFactory:: NewTrainRowSet( const char* topic, 
    const FeatSelectParameter& featSelectParameter,
    const IndPriors& priorTerms,
    const class ExtScoreParam& extScoreParam )
{
    SetTopic( topic, featSelectParameter, priorTerms );
    ExtScores* pES  = new ExtScores( extScoreParam, topic, true);// 'true' means training
    return new RowSetMem( Mtrain, yTrain, featSelect, trainDocs, allInd, pES );
}
#endif //!defined(TEST_ONLY)

TopicRowSet* LemurDataFactory:: NewTestRowSet( const char* topic, const class ExtScoreParam& extScoreParam ){
    if( strcmp(topic,currtopic.c_str()) )
        throw logic_error("Topic does not match training");

    //external scores
    ExtScores* pES  = new ExtScores( extScoreParam, topic, false); //false means test*

    //Log(8)<<"\nTopicRowSet, Dim "<<featSelect.size()<<" Time "<<Log.time();
    return new LemurTopicRowSet( topic,
        featSelect, 
        testDocs, 
        allInd, 
        testQrel,
        tfMethod,
        wordRestrict,
        idf,
        pES,
        cosineNormalize );
}

TopicRowSet* LemurDataFactory:: NewTestRowSet //test only
         ( const char* topic, const vector<int>& featSelect_, const class ExtScoreParam& extScoreParam )
{
    currtopic = topic;
    featSelect = featSelect_;

    ExtScores* pES  = new ExtScores( extScoreParam, topic,  false); //false means test

    //Log(8)<<"\nTopicRowSet, Dim "<<featSelect.size()<<" Time "<<Log.time();
    return new LemurTopicRowSet( topic,
        featSelect, 
        testDocs, 
        allInd, 
        testQrel,
        tfMethod,
        wordRestrict,
        idf,
        pES,
        cosineNormalize );
    Log(6)<<"\nTopicRowSet created; Dim "<<featSelect.size()<<" Time "<<Log.time();
}

// Added by Vladimir Menkov, for thresholding option (B.2). 
// Based on NewTestRowSet(), but uses the list of docs from
// idfDocsFile to compose the population.
// 2004-08-12
TopicRowSet* LemurDataFactory:: NewTrainPopRowSet 
         ( const char* topic, 
	   //const vector<int>& featSelect_, 
	   const class ExtScoreParam& extScoreParam,
	   const string & idfDocsFile)
{
    if( strcmp(topic,currtopic.c_str()) )
        throw logic_error("Topic does not match training");

    //external scores
    ExtScores* pES  = new ExtScores( extScoreParam, topic, false); //false means test*

     if( 0==idfDocsFile.size() ) {
      throw logic_error("We don't know what the wider Training Population is (no idfDocsFile given)");
    }
    vector<int> idfDocs = CollectFromFile(idfDocsFile.c_str(),allInd);

    //Log(8)<<"\nTopicRowSet, Dim "<<featSelect.size()<<" Time "<<Log.time();
    return new LemurTopicRowSet( topic,
        featSelect, 
        idfDocs, 
        allInd, 
        testQrel,
        tfMethod,
        wordRestrict,
        idf,
        pES,
        cosineNormalize );
    Log(6)<<"\nTopicRowSet created; Dim "<<featSelect.size()<<" Time "<<Log.time();
}




void LemurDataFactory:: SetWordRestrict( const char* bagOfWordsFile, Index *trainInd )
{
    wordRestrict.clear();
    // apriori word restriction
    if( 0!=bagOfWordsFile && '\0'!=bagOfWordsFile[0] )
    {
        std::ifstream f(bagOfWordsFile);
        if( !f.good() )
            throw runtime_error("Bad file: bagOfWordsFile"); //--->>--
        const int bufsize=1000;
        char buf[bufsize+1];
        while( f>>buf ) {
            int termId = allInd->term( buf );
            if( 0==termId )
                Log(3)<<"\n***Word from the bag is not in the index: "<<buf;
                //throw runtime_error(string("Restricting word not in the index: ")+buf);
            else
                wordRestrict.push_back( termId );
        }
        Log(3)<<"\nBag of words: "<<wordRestrict.size();
    }
    else
    {
        if( trainInd ) //use terms from training only
        {
            wordRestrict.resize( trainInd->termCountUnique() );
            for( int i=0; i<trainInd->termCountUnique(); i++ )
            {
                const char* spelling = trainInd->term(i+1); //Lemur word indices start from 1
                int iAll = allInd->term(spelling);
                if( 0>=iAll )
                    throw runtime_error(string("Word from training not in the index: ")+spelling);
                wordRestrict[i] = iAll;
            }
        }
        else //use all terms
            // NB: could do better, scan training docs for terms that are there
        {
            wordRestrict.resize( allInd->termCountUnique() );
            for( int i=0; i<allInd->termCountUnique(); i++ )
                wordRestrict[i] = i+1; //Lemur word indices start from 1
        }
    }

    std::sort( wordRestrict.begin(),  wordRestrict.end() );
}
void LemurDataFactory ::ComputeIDF( Index *trainInd, const string idfDocsFile )
{
    // Definition borrowed from SumFilterMethod ctor:
    //  "Pre-compute IDF values using the "Corpus-based feature weight"
    //  specified by David Lewis in July 2002."
    if(NOIDF==idfMethod)
        idf.resize( wordRestrict.size(), 1.0 ); // default value is 1.0 - no idf

    else
    // compute IDF basing on training data, 
    //  while term id's in wordRestrict refer to global index
    if( 0==trainInd && 0==idfDocsFile.size() ) // no training Index, use training sample
    {
        int countAll = Mtrain.size();
        // go from sparse vectors to wordRestrict
        idf.resize( wordRestrict.size(), 0 ); // use as counter first
        for( vector<SparseVector>::const_iterator idoc=Mtrain.begin(); idoc!=Mtrain.end(); idoc++ ) {
            for( SparseVector::const_iterator iw = idoc->begin(); iw!=idoc->end(); iw++ ) {
                //find in WordRestrict first
                vector<int>::const_iterator iWR = lower_bound(
                    wordRestrict.begin(), 
                    wordRestrict.end(), 
                    iw->first );
                if( iWR!=wordRestrict.end() && *iWR==iw->first ) //found
                    idf[ iWR-wordRestrict.begin() ]  ++ ;
            }
            //Log(12)<<"\nDoc "<<idoc-Mtrain.begin()+1<<" Time "<<Log.time();
        }           
        // now recalculate to log-idf
        unsigned nTermsInTrain = 0;
        for( unsigned i=0; i<wordRestrict.size(); i++ ) {
            if( idf[i]>0 )   nTermsInTrain++;
            idf[i] = log( (countAll+1.0)/(idf[i] + 1.0) );
        }
        Log(3)<<"\nTerms in Training "<<nTermsInTrain;
    }
    else if( 0==idfDocsFile.size() ) //have explicit training Index, still use training sample
    {
        idf.resize( wordRestrict.size() );
        int countAll = trainInd->docCount();
        for( unsigned i=0; i<wordRestrict.size(); i++ ) {
            const char* spelling = allInd->term( wordRestrict[i] );
            int trainTermID = trainInd->term( spelling );
            int count = trainTermID>0 ? trainInd->docCount(trainTermID) : 0;
            idf[i] = log( (countAll+1.0)/(count + 1.0) );
        }
    }
    else // use idfDocsFile, ignore training Index even if present
    {
        idf.resize( wordRestrict.size(), 0 ); // use as counter first
        int countAll = 0; //docs

        char buf[1000];
        ifstream fDocs( idfDocsFile.c_str() );
        while( fDocs>>buf && !fDocs.fail() ) {
            int docIndex = allInd->document(buf);
            if( docIndex<=0 )
                throw runtime_error(string("Document in 'for IDF' list not in the Lemur Index: ")+buf);
            countAll ++;
            SparseVector docTF = docToTFvector( docIndex, allInd, RAWTF );
            for( SparseVector::const_iterator iw = docTF.begin(); iw!=docTF.end(); iw++ ) {
                //find in WordRestrict first
                vector<int>::const_iterator iWR = lower_bound(
                    wordRestrict.begin(), 
                    wordRestrict.end(), 
                    iw->first );
                if( iWR!=wordRestrict.end() && *iWR==iw->first ) //found
                    idf[ iWR-wordRestrict.begin() ]  ++ ;
            }
        }
        // now recalculate to log-idf
        unsigned nTermsInIDF = 0;
        for( unsigned i=0; i<wordRestrict.size(); i++ ) {
            if( idf[i]>0 )   nTermsInIDF++;
            idf[i] = log( (countAll+1.0)/(idf[i] + 1.0) );
        }
        Log(3)<<"\nDocs for IDF calculation: "<<countAll;
    }
    //Log(12)<<"\nIDF ";
    //for( unsigned i=0; i<wordRestrict.size(); i++ )
        //Log(12)<<wordRestrict[i]<<":"<<idf[i]<<"  ";
}

vector<int> LemurDataFactory ::CollectFromFile( const char* docsFile, Index* ind ){
    vector<int> tDocs;
    char buf[1000];
    ifstream fDocs(docsFile);
    while( fDocs>>buf && !fDocs.fail() ) {
        int docIndex = allInd->document(buf);
        if( docIndex<=0 )
            throw runtime_error(string("Document not indexed: ")+buf);
        tDocs.push_back(docIndex);
    }
    if( tDocs.empty() )
            throw runtime_error("Empty training set");
    return tDocs;
}
vector<int> LemurDataFactory ::CollectFromIndex( Index* indOfDocs, Index* ind ){
    vector<int> tDocs;
    for( int i=1; i<=indOfDocs->docCount(); i++ ) {
        const char* spelling = indOfDocs->document(i);
        int docIndex = ind->document( spelling );
        if( docIndex<=0 )
            throw runtime_error(string("Document not indexed: ")+spelling);
        tDocs.push_back(docIndex);
    }
    if( tDocs.empty() )
            throw runtime_error("Empty training set");
    return tDocs;
}
#endif //USE_LEMUR

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
