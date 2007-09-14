// 2.10     Jun 20, 05  bug with ind.priors file read - fixed
// SL 3.02  May 29, 07  calc the total number of features (including repeated ones) in CollectWordRestrict
//                      pass totalFeats to RowSetMem in NewTrainRowSet
// SL 3.03  Jun 18, 07  nonzero mode features which are not active in training examples

#include "dataBin.h"
#include "DataFactory.h"
#include "FeatSelectParamManager.h"
//#include "bayes.h"
#include "ExtScore.h"

DataFactory::DataFactory(   //test only ctor
        TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_,
        vector<int>& wordRestrict_, vector<double>& idf_, unsigned ngroups_ )
:   tfMethod(tfMethod_), idfMethod(idfMethod_), cosineNormalize(cosineNormalize_),
    wordRestrict(wordRestrict_), idf(idf_), ngroups(ngroups_)
{
    if( idfMethod==NOIDF )
        idf.resize( wordRestrict.size(), 1.0 );
}
void DataFactory::ComputeIDF()
{
    // Definition borrowed from SumFilterMethod ctor:
    //  "Pre-compute IDF values using the "Corpus-based feature weight"
    //  specified by David Lewis in July 2002."
    idf.resize( wordRestrict.size(), 1.0 ); // default value is 1.0 - no idf
    if(NOIDF==idfMethod)
        return;                     //---->>---

    // compute IDF basing on training data, 
    int countAll = Mtrain.size();
    for( unsigned i=0; i<wordRestrict.size(); i++ ) {
        unsigned count = 0;
        for( vector<SparseVector>::const_iterator idoc=Mtrain.begin(); idoc!=Mtrain.end(); idoc++ )
            if( idoc->find(wordRestrict[i]) != idoc->end() )
                count ++;
        idf[i] = log( (countAll+1.0)/(count + 1.0) );
    }            
}
void DataFactory::ApplyIDFtoTrain() {
    if(NOIDF==idfMethod  && !cosineNormalize )
        return;                     //-nothing to worry about--->>---
    for( vector<SparseVector>::iterator iTrain=Mtrain.begin(); iTrain!=Mtrain.end(); iTrain++ )
    {
        vector<int>::const_iterator iWr = wordRestrict.begin();
        vector<int>::const_iterator iWend = wordRestrict.end();
        //Log(12)<<"\nTrain Sparse vector  ";
        //idf and compute norm
        double norm = 0.0;
        for( vector<SparseItem>::iterator iTF=iTrain->begin(); iTF!=iTrain->end(); iTF++ )
        {
            //Log(12)<<"  "<<iTF->first<<":"<<iTF->second;
            //locate in wordRestrict
            vector<int>::const_iterator i=lower_bound( iWr, iWend, iTF->first );
            if( i==iWend )
                break; //term out of 'wordRestrict', so are all the rest - sorted
            else{
                if( *i==iTF->first ) {//othw term missing from 'wordRestrict'
                    if( idfMethod!=NOIDF )
                        iTF->second *= idf[i-wordRestrict.begin()];
                    norm += iTF->second * iTF->second;
                }
                //Log(12)<<":"<<*i<<":"<<iTF->second;
                iWr = i;
            }
        }
        //normalize
        if( cosineNormalize && norm>0.0) {
            norm = 1.0/sqrt(norm);
            for( vector<SparseItem>::iterator iTF=iTrain->begin(); iTF!=iTrain->end(); iTF++ )
                iTF->second *= norm; //including those out of wordRestrict, we don't care
        }
    }
}

#if !defined(TEST_ONLY)
void DataFactory::SetTopic( const char* topic, 
    const FeatSelectParameter& featSelectParameter,
    const IndPriors& priorTerms )
{
    currtopic = topic; //save for test

    //specialize qrels
    YTrain( topic );
    //feature selection
    if( featSelectParameter.isOn() )
    {
        //select by correlation or Yule's Q
        featSelect = FeatureSelection( featSelectParameter, 
            Mtrain, yTrain, wordRestrict );
        std::sort( featSelect.begin(),  featSelect.end() );

        //make sure prior terms are there
        for( vector<int>::const_iterator is=wordRestrict.begin(); is!=wordRestrict.end(); is++ )
            if( priorTerms.HasIndPrior(*is) ) {
                unsigned i;
                for( i=0; i<featSelect.size(); i++ )
                    if( *is == featSelect[i] ) //already there
                        break;
                if( i>=featSelect.size() ) {  //not there
                    featSelect.push_back( *is );
                    //TODO:  Log(3)<<"\nEnter prior term: "<<allInd->term(*is);
                }
            }

        //OBSOLETE  for( set<int>::const_iterator is=priorTerms.begin(); is!=priorTerms.end(); is++ ) {
    }
    else
        featSelect = wordRestrict;

    std::sort( featSelect.begin(),  featSelect.end() );

    //-- RestrictIDF --
    idfTopic.resize( featSelect.size() );
    unsigned iWR=0; //trace wordRestrict
    //Log(8)<<"\nTopicIDF  ";
    for( unsigned i=0; i<featSelect.size(); i++ )// featSelect and wordRestrict supposed to be sorted
    {
        while( iWR<wordRestrict.size() && wordRestrict[iWR] != featSelect[i] )
            iWR++;
        if( iWR>=wordRestrict.size() )
            throw logic_error("Selected feature out of word set");
        idfTopic[i] = idf[iWR];
        //Log(8)<<featSelect[i]<<":"<<idfTopic[i]<<"  ";
    }
    Log(2)<<endl<<"Class sizes in training, 1/0: "<<ntrue(yTrain)
        <<" / "<<yTrain.size()-ntrue(yTrain)<<endl;
}
#endif //!defined(TEST_ONLY)

void DataFactory::CollectWordRestrict() {
    wordRestrict.clear();
    set<int> words;
    for( vector<SparseVector>::const_iterator mitr=Mtrain.begin(); mitr!=Mtrain.end(); mitr++ )
        for( SparseVector::const_iterator xitr=mitr->begin(); xitr!=mitr->end(); xitr++ ) {
            words.insert( xitr->first );
	    totalFeats++;  // SL., ver 3.02; calculate the total number of features
	}

    for( set<int>::const_iterator sitr=words.begin(); sitr!=words.end(); sitr++ )
        wordRestrict.push_back( *sitr );
}

#ifdef USE_LEMUR
PlainFileDataFactory::PlainFileDataFactory( 
    const char* trainFile_,
    const char* testFile_,
    const char* trainQrelFile,
    const char* testQrelFile,
    TFMethod tfMethod_,
    IDFMethod idfMethod_,
    bool cosineNormalize_ )
: trainFName(trainFile_), testFName(testFile_), 
DataFactory( tfMethod_, idfMethod_, cosineNormalize_ )
{
    trainQrel = new PlainOracle( trainQrelFile, true/*isTraining*/ );
    MakeSparseMatrix( Mtrain, trainDocNames, trainFName );
    Log(3)<<"\nTraining data "<<Mtrain.size()<<" rows";

    CollectWordRestrict();
    Log(3)<<"\nFeatures in training data: "<<wordRestrict.size();

    ComputeIDF();
    ApplyIDFtoTrain();

    testQrel = new PlainOracle( testQrelFile, false/*isTraining*/ );
}
PlainFileDataFactory::PlainFileDataFactory( //test only
    const char* testFile_,
    const char* testQrelFile,
    TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_,
    vector<int> wordRestrict_,
    vector<double> idf_ )
: trainFName(""), testFName(testFile_), 
DataFactory( tfMethod_, idfMethod_, cosineNormalize_, wordRestrict_, idf_ )
{
    trainQrel = 0;
    testQrel = new PlainOracle( testQrelFile, false/*isTraining*/ );
}
void PlainFileDataFactory::MakeSparseMatrix( vector<SparseVector>& matrix, vector<string>& docNames, string fName ) {
    matrix.clear();
    docNames.clear();
    istream* p_ifs;
    if( readFromStdin( fName.c_str() ) )
        p_ifs = &cin;
    else
        p_ifs = new ifstream(fName.c_str());
    if( !*p_ifs )
        throw runtime_error(string("Cannot open training file ")+fName);
    int nrows = 0;
    
    while(  p_ifs->good() )
    {
        string buf;
        getline( *p_ifs, buf );
            //Log(10)<<"\nrow "<<nrows+1<<" `"<<buf<<"`";//.str()
        istringstream rowbuf( buf);//.str() );
        int i; double d; char delim; string docName;
        rowbuf>>docName;
        if(0==docName.size()) //empty line
            continue;
        vector<SparseItem> x;
        while( rowbuf>>i>>delim>>d ) { //rowbuf.good()
            if( delim != ':' )
                throw runtime_error(string("Wrong delimiter, should be semicolon, is ")+delim);
            if( !rowbuf.fail() && d!=0.0 )
                x.push_back(SparseItem( i, TFWeight( d, tfMethod ) )); //x[i] = TFWeight( d, tfMethod );
        }
            //Log(10)<<"  parsed "<<docName<<" - "<<x;
        if( rowbuf.eof() ) { //othw corrupt
            matrix.push_back( SparseVector( x )  );
            docNames.push_back( docName );   //add( docName, nrows );
            nrows ++;
        }
        else throw runtime_error(string("Corrupt train file line: ") + buf);//.str()
    }
    if( !readFromStdin( fName.c_str() ) )
        delete p_ifs;
}

#if !defined(TEST_ONLY)
DenseData* PlainFileDataFactory:: NewTrainData( const char* topic, 
    const FeatSelectParameter& featSelectParameter,
    const IndPriors& priorTerms )
{
    throw runtime_error("Dense data not supported with plain file data");
    //SetTopic( topic, featSelectParameter, priorTerms );
    //return new DenseData( Mtrain, yTrain, featSelect, trainDocs, allInd );
}
RowSetMem* PlainFileDataFactory:: NewTrainRowSet( const char* topic, 
    const FeatSelectParameter& featSelectParameter,
    const IndPriors& priorTerms,
    const class ExtScoreParam& extScoreParam )
{
    SetTopic( topic, featSelectParameter, priorTerms );
    if( extScoreParam.isValid() )
        throw runtime_error("External scores not supported with plain data file!");
    ExtScores* pES  = new ExtScores( extScoreParam, topic, true/*train*/);
    PlainNameResolver* pNameResolver = new PlainNameResolver( featSelect, trainDocNames );
    RowSetMem* TrainRowSet = new RowSetMem( Mtrain, ngroups/*should be 0*/, groupsTrain, 
        yTrain, featSelect, pNameResolver, pES );
    return TrainRowSet;
}
#endif //!defined(TEST_ONLY)

TopicRowSet* PlainFileDataFactory:: NewTestRowSet( const char* topic, const class ExtScoreParam& extScoreParam ){
    if( strcmp(topic,currtopic.c_str()) )
        throw logic_error("Topic does not match training");

    //external scores
    if(extScoreParam.isValid()) throw runtime_error("External score not supported with plain file data");
    ExtScores* pES  = new ExtScores( extScoreParam, topic, false/*test*/);

    return new PlainTopicRowSet( topic,
        testFName.c_str(),
        testQrel,
        featSelect, 
        tfMethod,
        wordRestrict,
        idf,
        pES,
        cosineNormalize );
    Log(6)<<"\nTopicRowSet created; Dim "<<featSelect.size()<<" Time "<<Log.time();
}
TopicRowSet* PlainFileDataFactory:: NewTestRowSet //test only
         ( const char* topic, const vector<int>& featSelect_, const class ExtScoreParam& extScoreParam )
{
    currtopic = topic;
    featSelect = featSelect_;

    //external scores
    if(extScoreParam.isValid()) throw runtime_error("External score not supported with plain file data");
    ExtScores* pES  = new ExtScores( extScoreParam, topic,  false); //false means test

    return new PlainTopicRowSet( topic,
        testFName.c_str(),
        testQrel,
        featSelect, 
        tfMethod,
        wordRestrict,
        idf,
        pES,
        cosineNormalize );
    Log(6)<<"\nTopicRowSet created; Dim "<<featSelect.size()<<" Time "<<Log.time();
}
#endif //USE_LEMUR

/*
 * PlainFileYDataFactory
 */
PlainFileYDataFactory::PlainFileYDataFactory( 
    const char* trainFile_,
    const char* testFile_,
    bool rowIdMode_,
    TFMethod tfMethod_,
    IDFMethod idfMethod_,
    bool cosineNormalize_, 
    unsigned ngroups_ )
: trainFName(trainFile_), testFName(testFile_),  rowIdMode(rowIdMode_),
DataFactory( tfMethod_, idfMethod_, cosineNormalize_, ngroups_ )
{
    MakeSparseMatrix( trainFName );
    Log(3)<<"\nTraining data "<<Mtrain.size()<<" rows";

    CollectWordRestrict();
    Log(3)<<"\nFeatures in training data: "<<wordRestrict.size();

    ComputeIDF();
    ApplyIDFtoTrain();
}
PlainFileYDataFactory::PlainFileYDataFactory( //test only
    const char* testFile_,
    bool rowIdMode_,
    TFMethod tfMethod_,
    IDFMethod idfMethod_,
    bool cosineNormalize_,
    vector<int> wordRestrict_,
    vector<double> idf_,
    unsigned ngroups  )
: trainFName(""), testFName(testFile_), rowIdMode(rowIdMode_),
DataFactory( tfMethod_, idfMethod_, cosineNormalize_, wordRestrict_, idf_, ngroups )
{}

void PlainFileYDataFactory::MakeSparseMatrix( string fName ) { //collect both X and Y and groups
    Mtrain.clear();
    vector<bool> Y;//temp for collection
    istream* p_ifs;
    if( readFromStdin( fName.c_str() ) )
        p_ifs = &cin;
    else
        p_ifs = new ifstream(fName.c_str());
    if( !*p_ifs )
        throw runtime_error(string("Cannot open training file ")+fName);
    int nrows = 0;
    int nfeatures=0; //SL

    string nextRowId="";

    while(  p_ifs->good() )
    {
        string buf;
        getline( *p_ifs, buf );
            //Log(10)<<"\nrow "<<nrows+1<<" `"<<buf<<"`";//.str()
        if( buf[0]=='#' ) {
                if( rowIdMode ) {
                    istringstream rowbuf( buf );
                    string hash, uidkw, id;
                    rowbuf>>hash>>uidkw>>id;
                    if( 0==stricmp( uidkw.c_str(), "UID:" ) )
                        nextRowId = id;
                }
                continue; //look for not-a-comment line
        }

        //essential data row
        istringstream rowbuf( buf);//.str() );
        int i; double d; char delim; int y;
        
        rowbuf>>y;
        if( rowbuf.fail() ) //empty line
            continue;
        if( rowIdMode && 0==nextRowId.size() )
            throw runtime_error("Row Id absent ");

        vector<bool> groups;
        int ig;
        for( unsigned g=0; g<ngroups; g++ ) {
            rowbuf>>ig;
            if( 1==ig ) groups.push_back(true);
            else if( 0==ig ) groups.push_back(false);
            else throw runtime_error(string("Wrong group indicator in line ")+int2string(nrows+1));
        }
        groupsTrain.push_back( groups );

        vector<SparseItem> x;
        while( rowbuf>>i>>delim>>d ) { //rowbuf.good()
            if( i<=0 )
                throw runtime_error(string("Non-positive variable id in line: ")+int2string(nrows+1)
                                    +"  sparse pair "+int2string(x.size()+1));
            if( delim != ':' )
                throw runtime_error(string("Wrong delimiter, should be semicolon, is ")+delim);
            if( !rowbuf.fail() && d!=0.0) { // added by SL
                x.push_back(SparseItem( i, TFWeight( d, tfMethod ) )); //x[i] = TFWeight( d, tfMethod );
		nfeatures++;
	    }
        }
        if( rowbuf.eof() ) { //othw corrupt
            Mtrain.push_back( SparseVector( x ) );
            Y.push_back( (1==y ? true : false ) );
            nrows ++;
            if( rowIdMode ) {
                rowIds.push_back( nextRowId );
                nextRowId.clear();
            }
        }
        else throw runtime_error(string("Corrupt train file line: ") + buf);//.str()
    }
    if( !readFromStdin( fName.c_str() ) )
        delete p_ifs;

    // Y - once and for all
    yTrain = BoolVector( false, Y.size() );
    for( unsigned i=0; i<Y.size(); i++ )
        yTrain[i] = Y[i];

}

#if !defined(TEST_ONLY)
DenseData* PlainFileYDataFactory:: NewTrainData( const char* topic, 
    const FeatSelectParameter& featSelectParameter,
    const IndPriors& priorTerms )
{
    throw runtime_error("Dense data not supported with plain file data");
    //SetTopic( topic, featSelectParameter, priorTerms );
    //return new DenseData( Mtrain, yTrain, featSelect, trainDocs, allInd );
}


RowSetMem* PlainFileYDataFactory:: NewTrainRowSet( const char* topic, 
    const FeatSelectParameter& featSelectParameter,
    const IndPriors& priorTerms,
    const class ExtScoreParam& extScoreParam )
{

    // get the non-zero mode feature information
    priorTerms.checkNonZeroModes(wordRestrict, nonzeromodefeats);  // v3.03

    SetTopic( topic, featSelectParameter, priorTerms );
    if( extScoreParam.isValid() )
        throw runtime_error("External scores not supported with plain data file!");
    ExtScores* pES  = new ExtScores( extScoreParam, topic, true/*train*/);
    PlainNameResolver* pNameResolver =  rowIdMode ? 
        new PlainNameResolver( featSelect, rowIds ) :
        new PlainNameResolver( featSelect );
    RowSetMem* TrainRowSet = new RowSetMem( Mtrain, ngroups, groupsTrain, yTrain, nonzeromodefeats, // v3.03 
					    featSelect, pNameResolver, pES);
    TrainRowSet->setTotalFeats(totalFeats); //ver3.02 SL  
    return TrainRowSet;
}
#endif //!defined(TEST_ONLY)



TopicRowSet* PlainFileYDataFactory:: NewTestRowSet( const char* topic, const class ExtScoreParam& extScoreParam ){
    if( strcmp(topic,currtopic.c_str()) )
        throw logic_error("Topic does not match training");

    //external scores
    if(extScoreParam.isValid()) throw runtime_error("External score not supported with plain file data");
    ExtScores* pES  = new ExtScores( extScoreParam, topic, false/*test*/);

    return new PlainYRowSet( topic,
        testFName.c_str(), rowIdMode,
        featSelect, 
        tfMethod,
        wordRestrict,
        idf,
        pES,
        cosineNormalize,
        ngroups );
    Log(6)<<"\nTopicRowSet created; Dim "<<featSelect.size()<<" Time "<<Log.time();
}
TopicRowSet* PlainFileYDataFactory:: NewTestRowSet //test only
         ( const char* topic, const vector<int>& featSelect_, const class ExtScoreParam& extScoreParam )
{
    currtopic = topic;
    featSelect = featSelect_;

    //external scores
    if(extScoreParam.isValid()) throw runtime_error("External score not supported with plain file data");
    ExtScores* pES  = new ExtScores( extScoreParam, topic,  false); //false means test

    return new PlainYRowSet( topic,
        testFName.c_str(), rowIdMode,
        featSelect, 
        tfMethod,
        wordRestrict,
        idf,
        pES,
        cosineNormalize,
        ngroups );
    Log(6)<<"\nTopicRowSet created; Dim "<<featSelect.size()<<" Time "<<Log.time();
}


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
