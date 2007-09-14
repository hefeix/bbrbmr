#ifndef DATA_FACTORY_
#define DATA_FACTORY_

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include <stdlib.h>

using namespace std;

#ifdef USE_LEMUR
# include "IndexManager.hpp" 
# include "Oracle.hpp"
#endif //USE_LEMUR
#include "TFIDFParamManager.h"
#include "data.h"
#include "PriorTerms.h"

class FeatSelectParameter;

class DataFactory {
public:
#if !defined(TEST_ONLY)
    virtual DenseData* NewTrainData( const char* topic,
        const FeatSelectParameter& featSelectParameter,
        const IndPriors& priorTerms
        ) =0;
    virtual RowSetMem* NewTrainRowSet( const char* topic,
        const FeatSelectParameter& featSelectParameter,
        const IndPriors& priorTerms,
        const class ExtScoreParam& extScoreParam
        ) =0;
#endif //!defined(TEST_ONLY)
    //virtual DenseData* NewTestData( const char* topic ) =0;
    virtual IRowSet* NewTestRowSet( const char* topic, const class ExtScoreParam& extScoreParam ) =0;
    virtual IRowSet* NewTestRowSet( const char* topic, const vector<int>& featSelect_, //test only
                        const class ExtScoreParam& extScoreParam ) =0;
    //access - primarily to store model file
#ifdef GROUPS
    unsigned NGroups() const { return ngroups; }
#endif
#ifdef POLYTOMOUS
    unsigned c() const { return classes.size(); }
    const vector<int>& Classes() const { return classes; }
#endif
    const vector<int>& Feats() const { return wordRestrict; }
    const vector<double>& Idf() const { return idf; }
    const vector<int>& TopicFeats() const { return featSelect; }
    TFIDFParameter tfidfParam() const { return TFIDFParameter(tfMethod,idfMethod,cosineNormalize); }
    //dtor
    virtual ~DataFactory() {}
protected:
    //--data representation--//
    TFMethod tfMethod;
    IDFMethod idfMethod;
    bool cosineNormalize;
#ifdef GROUPS
    unsigned ngroups;
#endif
    //--data storage - training--//
    vector<SparseVector> Mtrain;
#ifdef GROUPS
    vector< vector<bool> > groupsTrain;
#endif
    vector<int> wordRestrict;
    vector<double> idf;
    //-- topic related --//
    string currtopic;
    vector<int> featSelect;
    vector<double> idfTopic;
    vector<YType> yTrain; //parallel to Mtrain  was BoolVector
    vector<int> classes; //dictionary of class ids

    //--ctors--//
    DataFactory( TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_, unsigned ngroups_=0 )
        : tfMethod(tfMethod_), idfMethod(idfMethod_), cosineNormalize(cosineNormalize_)
#ifdef GROUPS
        , ngroups(ngroups_)
#endif
    {}
    DataFactory(  //test only ctor
#ifdef POLYTOMOUS
        vector<int>& classes_,
#endif
        TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_, 
        vector<int>& wordRestrict_, vector<double>& idf_
#ifdef GROUPS
        , unsigned ngroups_=0 
#endif
        );
    //--methods--//
    void CollectWordRestrict();
#ifdef POLYTOMOUS
    void CollectRecodeClasses( const vector<int>& );
#endif
    void ApplyIDFtoTrain();
    void ComputeIDF();
    void SetTopic( const char* topic, 
        const FeatSelectParameter& featSelectParameter,
        const IndPriors& priorTerms
        );
    virtual void YTrain( const char* topic ) =0;
};

class PlainNameResolver : public INameResolver {
protected:
    const vector<int>& terms;
#ifdef POLYTOMOUS
    const vector<int>& classes;
#endif
    const vector<string>& docNames; //can be active or not
    bool docsActive;
public:
    PlainNameResolver( const vector<int>& terms_,
#ifdef POLYTOMOUS
        const vector<int>& classes_, 
#endif
        const vector<string>& docNames_ )
        : terms(terms_), 
#ifdef POLYTOMOUS
        classes(classes_), 
#endif
        docNames(docNames_) 
    {docsActive=true;}
    PlainNameResolver( const vector<int>& terms_
#ifdef POLYTOMOUS
        ,const vector<int>& classes_
#endif
        )
        : terms(terms_), 
#ifdef POLYTOMOUS
        classes(classes_), 
#endif
        docNames(vector<string>()) 
    {docsActive=false;} //no docs
    //INameResolver
    unsigned dim() const { return terms.size(); }
    virtual string rowName( unsigned r ) const { 
        if( docsActive ) {//active doc names
            if( r>docNames.size() )
                throw runtime_error(string("Name unknown for row number ")+int2string(r));
            return docNames[r]; 
        }
        else
            return int2string( r );
    }
    virtual string colName( unsigned c ) const { return int2string(terms[c]); }
//    virtual int rowId( int r ) const { return r; }
    virtual int colId( unsigned c ) const { return terms[c]; }
    virtual unsigned colNumById( unsigned id ) const {
        vector<int>::const_iterator iterm=lower_bound( terms.begin(), terms.end(), (int)id );
        if( iterm==terms.end() || *iterm!=id )
            throw logic_error("PlainNameResolver: Term id not in the term set");
        return iterm - terms.begin();
    }
#ifdef POLYTOMOUS
    unsigned c() const { return classes.size(); }
    int classId( unsigned c ) const { return classes[c]; }
    unsigned classNumById( int id ) const {
        vector<int>::const_iterator itc=lower_bound( classes.begin(), classes.end(), id );
        if( itc==classes.end() || *itc!=id )
            throw logic_error(string("PlainNameResolver: Unknown class id: ")+int2string(id));
        return itc - classes.begin();
    }
#endif
};

class PlainFileYDataFactory : public DataFactory{
    string trainFName;
    string testFName;
    bool rowIdMode;
    vector<string> rowIds;
public:
    PlainFileYDataFactory( 
        const char* trainFile_,
        const char* testFile_, 
#ifndef POLYTOMOUS
        bool rowIdMode_,
#endif
        TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_
#ifdef GROUPS
        , unsigned ngroups 
#endif
        );
    PlainFileYDataFactory( //test only
        const char* testFile_, 
#ifdef POLYTOMOUS
        vector<int> classes_,
#else
        bool rowIdMode_,
#endif
        TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_,
        vector<int> wordRestrict_,
        vector<double> idf_
#ifdef GROUPS
        , unsigned ngroups
#endif
        );
    DenseData* NewTrainData( const char* topic, 
        const FeatSelectParameter& featSelectParameter,
        const IndPriors& priorTerms );
    RowSetMem* NewTrainRowSet( const char* topic, 
        const FeatSelectParameter& featSelectParameter,
        const IndPriors& priorTerms,
        const class ExtScoreParam& extScoreParam );
    TopicRowSet* NewTestRowSet( const char* topic, const class ExtScoreParam& extScoreParam );
    TopicRowSet* NewTestRowSet //test only
         ( const char* topic, const vector<int>& featSelect_, const class ExtScoreParam& extScoreParam );
private:
    void YTrain( const char* topic ) {} //nothing to do here
    void MakeSparseMatrix( string fName );
};

#ifdef USE_LEMUR
class PlainFileDataFactory : public DataFactory{
    string trainFName;
    string testFName;
    PlainOracle *trainQrel;
    PlainOracle *testQrel;
    vector<string> trainDocNames;
public:
    PlainFileDataFactory( 
        const char* trainFile_,
        const char* testFile_,
        const char* trainQrelFile,
        const char* testQrelFile,
        TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_ );
    PlainFileDataFactory( //test only
        const char* testFile_,
        const char* testQrelFile,
        TFMethod tfMethod_, IDFMethod idfMethod_, bool cosineNormalize_,
        vector<int> wordRestrict_,
        vector<double> idf_ );
    ~PlainFileDataFactory() {
        delete trainQrel;
        delete testQrel;
    }
    DenseData* NewTrainData( const char* topic, 
        const FeatSelectParameter& featSelectParameter,
        const IndPriors& priorTerms );
    RowSetMem* NewTrainRowSet( const char* topic, 
        const FeatSelectParameter& featSelectParameter,
        const IndPriors& priorTerms,
        const class ExtScoreParam& extScoreParam );
    TopicRowSet* NewTestRowSet( const char* topic, const class ExtScoreParam& extScoreParam );
    TopicRowSet* NewTestRowSet //test only
         ( const char* topic, const vector<int>& featSelect_, const class ExtScoreParam& extScoreParam );
private:
    void YTrain( const char* topic ) {
        yTrain = BoolVector( false, Mtrain.size() );
        for( unsigned i=0; i<Mtrain.size(); i++ )
            yTrain[i] =( REL==trainQrel->ask( topic, trainDocNames[i].c_str() ) ); //oracleNRelDefault
    }
    void MakeSparseMatrix( vector<SparseVector>& matrix, vector<string>& docNames, string fName );
};
#endif // USE_LEMUR

#endif //DATA_FACTORY_

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
