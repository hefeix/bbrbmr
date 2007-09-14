/*
SL 3.02  May 30, 07   calculate total number of features in collectWordRestrict; pass info to RowSetMem in newtrainrowset.
SL 3.03  Jun 12, 07   non-zero mode feature not active in training examples, the final beta should equal to the mode.
 */

#include "DataFactory.h"
#include "dataPoly.h"
#include "ExtScore.h"
#include "PriorTerms.h" // v3.03

DataFactory::DataFactory(   //test only ctor
        vector<int>& classes_,
        TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_,
        vector<int>& wordRestrict_, vector<double>& idf_ )
:   classes(classes_),
    tfMethod(tfMethod_), idfMethod(idfMethod_), cosineNormalize(cosineNormalize_),
    wordRestrict(wordRestrict_), idf(idf_)  
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
void DataFactory::SetTopic( const char* topic, 
    const FeatSelectParameter& featSelectParameter,
    const IndPriors& priorTerms ) //prior terms not used yet
{
    currtopic = topic; //save for test

    //specialize qrels
    YTrain( topic );
    /*/feature selection
    if( featSelectParameter.isOn() )
    {
        //select by correlation or Yule's Q
        featSelect = FeatureSelection( featSelectParameter, 
            Mtrain, yTrain, wordRestrict );

        //make sure prior terms are there
        for( set<int>::const_iterator is=priorTerms.begin(); is!=priorTerms.end(); is++ ) {
            unsigned i;
            for( i=0; i<featSelect.size(); i++ )
                if( *is == featSelect[i] ) //already there
                    break;
            if( i>=featSelect.size() ) //not there
            {
                featSelect.push_back( *is );
                //TODO:  Log(3)<<"\nEnter prior term: "<<allInd->term(*is);
            }
        }
    }
    else*/
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
    //Log(2)<<endl<<"Documents in Train: "<<yTrain.size()<<endl;
}
void DataFactory::CollectWordRestrict() {
    wordRestrict.clear();
    set<int> words;
    totalFeats=0;
    for( vector<SparseVector>::const_iterator mitr=Mtrain.begin(); mitr!=Mtrain.end(); mitr++ )
        for( SparseVector::const_iterator xitr=mitr->begin(); xitr!=mitr->end(); xitr++ ) {
            words.insert( xitr->first );
	    totalFeats++;  // SL, ver 3.02
	}

    for( set<int>::const_iterator sitr=words.begin(); sitr!=words.end(); sitr++ )
        wordRestrict.push_back( *sitr );
}
void DataFactory::CollectRecodeClasses( const vector<int>& clids ) {
    classes.clear();
    set<int> cset;
    //collect: clids is what occurs in data
    for( vector<int>::const_iterator yitr=clids.begin(); yitr!=clids.end(); yitr++ )
            cset.insert( *yitr );

    for( set<int>::const_iterator sitr=cset.begin(); sitr!=cset.end(); sitr++ )
        classes.push_back( *sitr );
//  cout<<"\n!classes "<<classes;cout.flush();

    yTrain.clear();
    //yTrain is class numbers [0,c)
    for( vector<int>::const_iterator yitr=clids.begin(); yitr!=clids.end(); yitr++ ) //recode
        yTrain.push_back( lower_bound( classes.begin(), classes.end(), *yitr ) - classes.begin() );
//  cout<<"\n! yTrain recoded "<<yTrain;cout.flush();
}

/*
 * PlainFileYDataFactory
 */
PlainFileYDataFactory::PlainFileYDataFactory( 
    const char* trainFile_,
    const char* testFile_,
    //unsigned c, #classes
    TFMethod tfMethod_,
    IDFMethod idfMethod_,
    bool cosineNormalize_ )
: trainFName(trainFile_), testFName(testFile_), 
DataFactory( tfMethod_, idfMethod_, cosineNormalize_ )
{
    MakeSparseMatrix( trainFName );
    Log(3)<<"\nCases in training data: "<<Mtrain.size();

    CollectWordRestrict();
    Log(3)<<"\nFeatures in training data: "<<wordRestrict.size();

    //CollectRecodeClasses();
    Log(3)<<"\nClasses in training data: "<<classes.size();

    ComputeIDF();
    ApplyIDFtoTrain();
}
PlainFileYDataFactory::PlainFileYDataFactory( //test only
    const char* testFile_,
    vector<int> classes_, //TODO REF?
    TFMethod tfMethod_,
    IDFMethod idfMethod_,
    bool cosineNormalize_,
    vector<int> wordRestrict_, //TODO REF?
    vector<double> idf_ //TODO REF?
    )
: trainFName(""), testFName(testFile_),
DataFactory( classes_, tfMethod_, idfMethod_, cosineNormalize_, wordRestrict_, idf_ )
{}

void PlainFileYDataFactory::MakeSparseMatrix( string fName ) { //collect both X and Y
    Mtrain.clear();
    vector<int> Y;//temp for collection
    istream *p_ifs;
    if( readFromStdin( fName.c_str() ) )
        p_ifs = &cin;
    else
        p_ifs = new ifstream(fName.c_str());
    if( !*p_ifs )
        throw runtime_error(string("Cannot open training file ")+fName);
    int nrows = 0;
    set<unsigned> illegalLabels;
    //unsigned ignoredRows =0;

    int nfeatures=0; // SL v3.02

    while( p_ifs->good() )
    {
        string buf;
        getline( *p_ifs, buf );
            //Log(10)<<"\nrow "<<nrows+1<<" `"<<buf<<"`";//.str()
        if( buf[0]=='#' )
            continue; //look for not-a-comment line
        istringstream rowbuf( buf);//.str() );
        int i; double d; char delim; 
        int y;
        
        rowbuf>>y;
        if( rowbuf.fail() ) //empty line
            continue;
        //allow all: if( y<0 || y>=m_c )
            //throw runtime_error( string("Illegal class label: ")+int2string(y)
              //  +"\tRow "+int2string(nrows+1)+" Classes "+int2string(c())) ;
        //if( y>=c() ) {
          //  illegalLabels.insert(y);
            //ignoredRows++;
            //continue;
        //}

        vector<SparseItem> x;
        while( rowbuf>>i>>delim>>d ) { //rowbuf.good()
            if( i<0 )
                throw runtime_error(string("Negative variable id in row ")+int2string(nrows+1));
            /** when featex fixed
            if( i<=0 )
                throw runtime_error(string("Non-positive variable id in row ")+int2string(nrows+1));
                */
            if( delim != ':' )
                throw runtime_error(string("Wrong delimiter, should be semicolon, is ")+delim);
            if( !rowbuf.fail() && d!=0.0 ) {
                x.push_back(SparseItem( i, TFWeight( d, tfMethod ) )); //x[i] = TFWeight( d, tfMethod );
		nfeatures++;
	    }
        }
        if( rowbuf.eof() ) { //othw corrupt
            Mtrain.push_back( SparseVector( x ) );
            Y.push_back( y );
            nrows ++;
        }
        else throw runtime_error(string("Corrupt train file line: ") + buf);//.str()
    }
    if( !readFromStdin( fName.c_str() ) ) //bug fixed - ver 1.06
        delete p_ifs;

    // Y - once and for all
    CollectRecodeClasses( Y );
    /*yTrain = vector<YType>( Y.size(), false );
    for( unsigned i=0; i<Y.size(); i++ )
        yTrain[i] = Y[i];*/
//  cout<<"\n! yTrain "<<yTrain;cout.flush();
}


DenseData* PlainFileYDataFactory:: NewTrainData( const char* topic, 
    //unsigned c, #classes
    const FeatSelectParameter& featSelectParameter,
    const IndPriors& priorTerms )
{
    throw runtime_error("Dense data not supported with plain file data");
    //SetTopic( topic, featSelectParameter, priorTerms );
    //return new DenseData( Mtrain, yTrain, featSelect, trainDocs, allInd );
}
RowSetMem* PlainFileYDataFactory:: NewTrainRowSet( const char* topic, 
    //unsigned c, #classes
    const FeatSelectParameter& featSelectParameter,
    const IndPriors& priorTerms,
    const class ExtScoreParam& extScoreParam )
{
    priorTerms.checkNonZeroModes(wordRestrict, nonzeromodefeats);  // v3.03
    SetTopic( topic, featSelectParameter, priorTerms );   // priorTerms not used yet
    if( extScoreParam.isValid() )
        throw runtime_error("External scores not supported with plain data file!");
    ExtScores* pES  = new ExtScores( extScoreParam, topic, true/*train*/);
    PlainNameResolver* pNameResolver = new PlainNameResolver( featSelect, classes );
    RowSetMem* TrainRowSet = new RowSetMem( Mtrain, yTrain, nonzeromodefeats, featSelect, pNameResolver, pES ); //v3.03 add nonzeroModes
    TrainRowSet->setTotalFeats(totalFeats);  // SL ver3.02
    return TrainRowSet;
}

TopicRowSet* PlainFileYDataFactory:: NewTestRowSet(
    const char* topic, 
    //unsigned c, #classes
    const class ExtScoreParam& extScoreParam )
{
    if( strcmp(topic,currtopic.c_str()) )
        throw logic_error("Topic does not match training");

    //external scores
    if(extScoreParam.isValid()) throw runtime_error("External score not supported with plain file data");
    ExtScores* pES  = new ExtScores( extScoreParam, topic, false/*test*/);

    return new PlainYRowSet(
        testFName.c_str(),
        classes,
        featSelect, 
        tfMethod,
        wordRestrict,
        idf,
        pES,
        cosineNormalize );
    Log(8)<<"\nTopicRowSet created; Dim "<<featSelect.size()<<" Time "<<Log.time();
}
TopicRowSet* PlainFileYDataFactory:: NewTestRowSet //test only
    ( const char* topic, 
    //unsigned c, #classes
    const vector<int>& featSelect_, const class ExtScoreParam& extScoreParam )
{
    currtopic = topic;
    featSelect = featSelect_;

    //external scores
    if(extScoreParam.isValid()) throw runtime_error("External score not supported with plain file data");
    ExtScores* pES  = new ExtScores( extScoreParam, topic,  false); //false means test

    return new PlainYRowSet(
        testFName.c_str(),
        classes,
        featSelect, 
        tfMethod,
        wordRestrict,
        idf,
        pES,
        cosineNormalize );
    Log(8)<<"\nTopicRowSet created; Dim "<<featSelect.size()<<" Time "<<Log.time();
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
