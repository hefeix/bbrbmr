#ifndef STORED_MODEL_
#define STORED_MODEL_

#include <string>
#include <fstream>
#include <vector>
using namespace std;

#include "poly.h"
#include "TFIDFParamManager.h"
#include "DataFactory.h"
#include "ModelTypeParam.h"
#include "Design.h"

#define TITLE "Bayesian Polytomous Regression model file ver "
static const double ver = 2.5; //beta: 1)sparse 2) classes and feats by input id's

// key words
#define KW_endofheader "endofheader"
#define KW_endofbeta "endofbeta"

#define KW_nclasses "nclasses"
#define KW_classes "classes"
#define KW_nDesignParams "nDesignParams"
#define KW_featRestrict "featRestrict"
#define KW_featRestrictRange "featRestrictRange"
#define KW_idf "idf"
#define KW_tfMethod "tfMethod"
#define KW_idfMethod "idfMethod"
#define KW_cosNorm "cosineNormalize"

#define KW_topic "topic"
#define KW_modelType "modelType"
#define KW_design "design"
#define KW_topicFeats "topicFeats"
#define KW_betaDense "betaDense"
#define KW_betaClassSparse "betaClassSparse"
#define KW_intercept "intercept"
#define KW_threshold "threshold"

class ReadModel {
    bool active;
    string filename;
    ifstream file;
    unsigned c;
    unsigned nDesignParams;
    //data factory
    vector<int> featRestrict;
    vector<double> idf;
    TFIDFParameter tfidfParameter;
    vector<int> topicFeats;
    vector<int> classes;
    //vector<double> beta;
        //double intercept;
    ModelType modelType;
    DesignParameter design;
    //double threshold;
public:
    //access
    bool Active() const { return active; }
    const vector<int>& Feats() const { return featRestrict; }
    const vector<int>& Classes() const { return classes; }
    unsigned NClasses() const { return classes.size(); }
    const vector<double>& Idf() const { return idf; }
    const TFIDFParameter& tfidfParam() const { return tfidfParameter; }
    //access - topic related
    const vector<int>& TopicFeats() const { return topicFeats; }
    //const vector<double>& Beta() const { return beta; }
    //double Threshold() const { return threshold; }
    const ModelType& getModelType() const { return modelType; }
    const DesignParameter& getDesignParameter() const { return design; }
    bool ReadParamMatrix( const IDesign* pDesign, IParamMatrix& beta ) {
        if(!active) return true;         //--->>--
        beta.reset( nDesignParams, c );
        string buf;
        char dbuf[100];
        //read all lines until KW_endofbeta or eof
        while( getline( file, buf ) ) { //---line-wise---
            istringstream rowbuf( buf );
            string kw;
            rowbuf>>kw;
            int k;
            double v;
            if( kw==KW_endofbeta ) break;
            else if( kw==KW_betaDense ) {
                rowbuf>>k;
                 //workaround for Microsoft bug: fails to read "...e-320"
                for( unsigned j=0; rowbuf>>dbuf; j++ ) {
                    sscanf(dbuf,"%le",&v);
                    beta( j, k ) = v;
                }
                if( !rowbuf.eof() ) //the only good reason to end the loop 
                    throw runtime_error("Corrupt model file: 'beta'");
            }
            else if( kw==KW_betaClassSparse ) {// ver 2.5
                rowbuf>>k;
                //copied from PlainNameResolver::classNumById( int id )
                vector<int>::const_iterator itc=lower_bound( classes.begin(), classes.end(), k );
                if( itc==classes.end() || *itc!=k )
                    throw runtime_error(string("Unknown class id in model file: ")+int2string(k));
                unsigned classNum = itc - classes.begin();

                int j;  char delim;
                while( rowbuf>>j>>delim>>dbuf ) {
                    sscanf(dbuf,"%le",&v); //workaround for Microsoft bug: fails to read "...e-320"
                    if( j<0 )
                        throw runtime_error(string("ReadModel: Negative variable id for class ")+int2string(k));
                    if( delim != ':' )
                        throw runtime_error(string("Wrong delimiter, should be semicolon, is ")+delim);
                    if( !rowbuf.fail() ) {
                        //cout<<"\n-- "<<j<<" "<<pDesign->colNumById(j)<<" "<<k<<" "<<classNum;
                        beta( pDesign->colNumById(j), classNum ) = v;
                    }
                }
                if( !rowbuf.eof() ) //the only good reason to end the loop 
                    throw runtime_error("Corrupt model file: 'beta'");
            }

            if( !rowbuf.good() && !rowbuf.eof() )
                throw runtime_error(string("Corrupt model file, line: ")+buf);

        }//---line-wise---

        //if( !file.eof() && !file.good() )
        //    throw runtime_error(string("Corrupt model file after following line: ")+buf);

        return true;
    }
    //ctor
    ReadModel(string filename_) : filename(filename_) {
        active = 0<filename.size();
        if(!active) return;         //--->>--

        string buf;
        file.open(filename.c_str());
        if( !file.good() )
            throw runtime_error(string("Cannot open model file: ")+filename);
        getline( file, buf ); //title line

        //read header - all lines until KW_endofheader
        while( getline( file, buf ) ) { //line-wise
            istringstream rowbuf( buf );
            string kw;
            rowbuf>>kw;
            if( kw==KW_endofheader ) break;
            else if( kw==KW_nclasses ) {
                int c_;
                rowbuf  >>c_;
                if( c_<2 )
                    throw runtime_error("Corrupt model file: less than 2 classes");
                c = (unsigned)c_;
            }
            else if( kw==KW_tfMethod ) {
                int i;  rowbuf>>i;
                tfidfParameter.set_tfMethod( TFMethod(i) );
            }
            else if( kw==KW_idfMethod ) {
                int i;  rowbuf>>i;
                tfidfParameter.set_idfMethod(IDFMethod(i));
            }
            else if( kw==KW_cosNorm ) {
                bool b; rowbuf>>b;
                tfidfParameter.set_cosineNormalize(b);
            }
            else if( kw==KW_featRestrict ) {
                int i;
                while( rowbuf.good() ) {
                    rowbuf>>i;
                    featRestrict.push_back(i);
                }
                if( !rowbuf.eof() ) //the only good reason to end the loop 
                    throw runtime_error("Corrupt model file header: 'featRestrict'");
                //should be already sorted! sort( featRestrict.begin(), featRestrict.end() );
            }
            else if( kw==KW_featRestrictRange ) {
                int from, to;
                rowbuf>>from>>to;
                for( int i=from; i<=to; i++ )
                    featRestrict.push_back(i);
            }
            else if( kw==KW_idf ) {
                double d;
                char dbuf[100];
                while( rowbuf.good() ) {
                //workaround for Microsoft bug: fails to read "...e-320"
                    rowbuf>>dbuf;
                    sscanf( dbuf, "%le", &d );
                    idf.push_back( d );
                }
                if( !rowbuf.eof() ) //the only good reason to end the loop 
                    throw runtime_error("Corrupt model file header: 'idf'");
            }
            else if( kw==KW_nDesignParams ) {
                int c_;
                rowbuf  >>c_;
                if( c_<1 )
                    throw runtime_error("Corrupt model file: less than 1 design params");
                nDesignParams = (unsigned)c_;
            }
            else if( kw==KW_modelType )
                rowbuf  >>(int&)modelType.m_link
                        >>(int&)modelType.m_opt
                        >>(int&)modelType.m_thr
                        >>modelType.m_standardize
                        >>modelType.m_referenceClass
                        >>modelType.strParam;
            else if( kw==KW_design ) {
                DesignType d;
                rowbuf  >>(int&)d;
                design = DesignParameter::DesignParameter( d );
            }
            else if( kw==KW_topicFeats ) {
                int i;
                while( rowbuf>>i )
                    topicFeats.push_back( i );
                if( !rowbuf.eof() ) //the only good reason to end the loop 
                    throw runtime_error("Corrupt model file: '" KW_topicFeats "'");
            }
            else if( kw==KW_classes ) {
                int i;
                while( rowbuf>>i )
                    classes.push_back( i );
                c = classes.size();
                if( !rowbuf.eof() ) //the only good reason to end the loop 
                    throw runtime_error("Corrupt model file: '" KW_classes "'");
            }
        } //line-wise

        if( !file.good() )
            throw runtime_error("Corrupt model file: header");

        //back-compatibility: no classes list
        if( classes.size()==0 ) {
            for( unsigned i=0; i<c; i++ )
                classes.push_back( i );
        }
    }
private:
    string GetString( string key, string defaultval );
    int GetInt( string key, int defaultval );
    double GetDouble( string key, double defaultval );
};
    
class WriteModel {
    string filename;
    const DataFactory& df;
    ofstream file;
    unsigned ntopics; //state
    bool active;
public:
    void WriteParams(
        const ModelType& modelType,
        const DesignParameter& design,
        const IDesign* pDesign,
        const vector<int>& feats,
        const IParamMatrix& beta )
        //double threshold
    {
        if(!active) return;         //--->>--

        file<<KW_nDesignParams<<" "<<beta.D()<<endl;
        file<<KW_modelType 
            <<" "<<modelType.Link()
            <<" "<<modelType.Optimizer()
            <<" "<<modelType.TuneThreshold()
            <<" "<<modelType.Standardize()
            <<" "<<modelType.ReferenceClass()
            <<" "<<modelType.StringParam()
            <<endl;
        file<<KW_design<<" "<<design.DesignType()<<endl;
        file<<KW_topicFeats;
        for( unsigned i=0; i<feats.size(); i++ )
            file<<" "<<feats[i];
        file<<endl;

        file<<KW_endofheader<<endl;

        for( unsigned k=0; k<beta.C(); k++ ) { // ver 2.5
            file<<KW_betaClassSparse<<" "<<df.Classes().at( k )
                <<setiosflags(ios_base::scientific)<<setprecision(12);
            for( unsigned j=0; j<beta.D(); j++ )
                if( 0!=beta(j,k) )
                    file<<" "<<pDesign->colId(j)<<":"<<beta(j,k);

            /*obsolete file<<KW_betaDense <<" "<<k<<setiosflags(ios_base::scientific) <<setprecision(12);
            for( unsigned j=0; j<beta.D(); j++ )
                file<<" "<<beta(j,k);*/

            file<<endl;
        }

        // ver 2.5 file<<KW_endofbeta<<endl;
        file.flush();
    }

    // ctor
    WriteModel(string filename_, const DataFactory& df_ )
        : filename(filename_), df(df_)
    {
        active = 0<filename.size();
        if(!active) return;         //--->>--
        file.open(filename.c_str());
        file<<TITLE<<ver<<endl;

        //file<<KW_nclasses<<" "<<df.c()<<endl;
        writeClasses( df.Classes() );
        writeTFIDFparameter( df.tfidfParam() );
        writeFeats( df.Feats() );
        if( df.tfidfParam().idfMethod() != NOIDF )
            writeIDF( df.Idf() );

        file.flush();
        if( !file.good() )
            throw runtime_error(string("Error creating model file '")+filename+"'");
    }
private:
    void writeClasses( const vector<int>& classes )
    {
        if(!active) return;         //--->>--
        file<<KW_classes;
        for( unsigned i=0; i<classes.size(); i++ )
            file<<" "<<classes[i];
        file<<endl;
        file.flush();
    }
    void writeTFIDFparameter( const TFIDFParameter& tfidf )
    {
        if(!active) return;         //--->>--
        file<<KW_tfMethod<<" "<<tfidf.tfMethod()<<endl;
        file<<KW_idfMethod<<" "<<tfidf.idfMethod()<<endl;
        file<<KW_cosNorm<<" "<<tfidf.cosineNormalize()<<endl;
        file.flush();
    }
    void writeFeats( const vector<int>& feats )
    {
        if(!active) return;         //--->>--
        file<<KW_featRestrict;
        for( unsigned i=0; i<feats.size(); i++ )
            file<<" "<<feats[i];
        file<<endl;
        file.flush();
    }
    void writeFeats( int from, int to )  {
        if(!active) return;         //--->>--
        file<<KW_featRestrictRange<<" "<<from<<" "<<to<<endl;
        file.flush();
    }
    void writeIDF( const vector<double>& idf )
    {
        if(!active) return;         //--->>--
        file<<KW_idf <<setiosflags(ios_base::scientific)<<setprecision(12);
        for( unsigned i=0; i<idf.size(); i++ )
            file<<" "<<idf[i];
        file<<endl;
        file.flush();
    }
};

#endif //STORED_MODEL_

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
