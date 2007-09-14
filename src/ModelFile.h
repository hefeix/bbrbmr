// 2.58     Sep 06, 05  ModelFile: token-wise file read for very large data; end-of-line ignored!
// 3.03     Jun 12, 07  non zero mode feats in WriteModel.


#ifndef STORED_MODEL_
#define STORED_MODEL_

#include <string>
#include <fstream>
#include <vector>
using namespace std;

#include "global.h"
#include "TFIDFParamManager.h"
#include "DataFactory.h"
#include "ModelTypeParam.h"
#include "Design.h"

#define TITLE "Bayesian Binary Regression ver "
static const double ver = 2.0; // groups added
/*static const double ver = 2.5; 
 ver = 2.5:
    - easy for manual preparation
    - comments allowed, line starts with '#'
    - unknown keywords are errors
 */

// key words
#define KW_endofheader "endofheader"
#define KW_endoftopic "endoftopic"

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
#define KW_beta "beta"
#define KW_betaSparse "betaSparse"
#define KW_intercept "intercept"
#define KW_threshold "threshold"
#define KW_ngroups "ngroups"
#define KW_betaHierSparse "betaHierSparse"
#define KW_modelname "modelname"


typedef map< unsigned, vector< pair<unsigned,double> > >TWHierSparse;
class ReadModel {
    bool active;
    string filename;
    ifstream file;
    //data factory
    vector<int> featRestrict;
    vector<double> idf;
    TFIDFParameter tfidfParameter;
    unsigned ngroups;
    //topic related
    string topic;
    bool  isHier;
    vector<int> feats;
    vector<double> beta;
    TWHierSparse wHier;
        //double intercept;
    ModelType modelType;
    DesignParameter design;
    double threshold;
    string lasttoken;
public:
    //access
    bool Active() const { return active; }
    const vector<int>& Feats() const { return featRestrict; }
    const vector<double>& Idf() const { return idf; }
    const TFIDFParameter& tfidfParam() const { return tfidfParameter; }
    unsigned NGroups() const { return ngroups; }
    //access - topic related
    string Topic() const { return topic; }
    bool IsHier()  const { return isHier; }
    const vector<int>& TopicFeats() const { return feats; }
    const vector<double>& Beta() const { return beta; }
    const TWHierSparse& WHier() const { return  wHier; }
        //double Intercept() const { return intercept; }
    double Threshold() const { return threshold; }
    const ModelType& getModelType() const { return modelType; }
    const DesignParameter& getDesignParameter() const { return design; }
    bool NextTopic() {
        if(!active) return true;         //--->>--
        topic.clear();
        isHier = false;
        feats.clear();
        beta.clear();
        threshold = 0;  //intercept = 0;
        string buf;
        //char dbuf[100];
        //read all lines until KW_endoftopic
        while( file.good() ) { //---line-wise--- getline( file, buf )
            //istringstream rowbuf( buf );
            string kw;
            if(lasttoken.size()==0)
                file>>kw;
            else{
                kw=lasttoken;
                lasttoken.clear();
            }
            Log(8)<<"\nModel keyword "<<kw<<", Time "<<Log.time(); Log(8).flush();
            if( kw==KW_endoftopic ) break;
            else if( kw==KW_topic )
                file>>topic;
            else if( kw==KW_topicFeats ) {
                int i;
                while( 1 ) { //rowbuf>>i
                    file>>buf;
                    istringstream rowbuf( buf );
                    if( rowbuf>>i )
                        feats.push_back(i);
                    else{
                        lasttoken=buf;
                        break;
                    }
                }
                //if( !rowbuf.eof() ) //the only good reason to end the loop 
                  //  throw runtime_error("Corrupt model file: '" KW_topicFeats "'");
            }
            else if( kw==KW_beta ) {
                double d;
                while( file>>buf ) {
                    if( 1==sscanf(buf.c_str(),"%le",&d) )//workaround for Microsoft bug: fails to read "...e-320"
                        beta.push_back( d );
                    else{
                        lasttoken=buf;
                        break;
                    }
                }
                //if( !rowbuf.eof() ) //the only good reason to end the loop 
                  //  throw runtime_error("Corrupt model file: 'beta'");
            }
            else if( kw==KW_betaHierSparse ) {
                isHier = true;
                unsigned feat; file>>feat;
                double d; char smcln; unsigned g;
                vector< pair<unsigned,double> >   sparseW;
                char dbuf[100];
                while( file>>buf ) { //!!! <var>:<val> does not allow spaces inside! 
                    istringstream rowbuf( buf );
                    if( rowbuf>>g>>smcln>>dbuf ) {
                        sscanf(dbuf,"%le",&d); //workaround for Microsoft bug: fails to read "...e-320"
                        sparseW.push_back( pair<unsigned,double>(g,d) );
                    }
                    else{
                        lasttoken=buf;
                        break;
                    }
                }
                //if( !rowbuf.eof() ) //the only good reason to end the loop 
                    //throw runtime_error("Corrupt model file: sparse hierarchical beta");
                wHier[feat] = sparseW;
            }
            else if( kw==KW_threshold ) {
                //workaround for Microsoft bug: fails to read "...e-320"
                file>>buf;
                sscanf( buf.c_str(), "%le", &threshold );
            }
            else if( kw==KW_modelType )
                file >>(int&)modelType.m_link
                        >>(int&)modelType.m_opt
                        >>(int&)modelType.m_thr
                        >>modelType.m_standardize
                        >>modelType.strParam;
            else if( kw==KW_design ) {
                DesignType d;
                file  >>(int&)d;
                design = DesignParameter::DesignParameter( d );
            }

            //if( !rowbuf.good() && !rowbuf.eof() )
              //  throw runtime_error(string("Corrupt model file, line: ")+buf);

        }//---line-wise---

        //if( !file.eof() && !file.good() )
          //  throw runtime_error(string("Corrupt model file after following line: ")+buf);

        return topic.size()>0; //we think topic's been read if name's been read
    }
    //ctor
    ReadModel(string filename_="") : filename(filename_), ngroups(0) {
        active = 0<filename.size();
        if(!active) return;         //--->>--

        //defaults: fill out
        tfidfParameter = TFIDFParameter( TFMethod(0), IDFMethod(0), false );

        string buf;
        file.open(filename.c_str());
        getline( file, buf ); //title line

        //read header - all lines until KW_endofheader
        while( file.good() ) { //line-wise getline( file.peek, buf )
            //istringstream rowbuf( buf );
            string kw;
            if(lasttoken.size()==0)
                file>>kw;
            else{
                kw=lasttoken;
                lasttoken.clear();
            }
            Log(8)<<"\nModel header keyword "<<kw<<", Time "<<Log.time(); Log(8).flush();
            if( kw==KW_endofheader ) break;
            else if( kw==KW_featRestrict ) {
                int i;
                while( 1 ) {  //rowbuf.good()
                    file>>buf;
                    istringstream rowbuf( buf );
                    if( rowbuf>>i )
                        featRestrict.push_back(i);
                    else{
                        lasttoken=buf;
                        break;
                    }
                }
                //if( !rowbuf.eof() ) //the only good reason to end the loop 
                  //  throw runtime_error("Corrupt model file header: 'featRestrict'");
                //should be already sorted! sort( featRestrict.begin(), featRestrict.end() );
            }
            /*else if( kw==KW_featRestrictRange ) {
                int from, to;
                rowbuf>>from>>to;
                for( int i=from; i<=to; i++ )
                    featRestrict.push_back(i);
            }*/
            else if( kw==KW_idf ) {
                double d;
                while( file>>buf ) {
                    if( 1==sscanf(buf.c_str(),"%le",&d) )//workaround for Microsoft bug: fails to read "...e-320"
                        idf.push_back( d );
                    else{
                        lasttoken=buf;
                        break;
                    }
                }
                //if( !rowbuf.eof() ) //the only good reason to end the loop 
                  //  throw runtime_error("Corrupt model file header: 'idf'");
            }
            else if( kw==KW_tfMethod ) {
                int i;  file>>i;
                tfidfParameter.set_tfMethod( TFMethod(i) );
            }
            else if( kw==KW_idfMethod ) {
                int i;  file>>i;
                tfidfParameter.set_idfMethod(IDFMethod(i));
            }
            else if( kw==KW_cosNorm ) {
                bool b; file>>b;
                tfidfParameter.set_cosineNormalize(b);
            }
            else if( kw==KW_ngroups ) {
                file>>ngroups;
            }
        } //line-wise

        //if( !file.good() )
          //  throw runtime_error("Corrupt model file: header");
        Log(8)<<"\nModel file header read, Time "<<Log.time(); Log(8).flush();
    }
};
    
class WriteModel {
    string filename;
    ofstream file;
    unsigned ntopics; //state
    bool active;

public:
    void WriteTopic( const string& topic, 
		     const string& modelname,
		     const ModelType& modelType,
		     const DesignParameter& design,
		     const vector<int>& feats,
		     const Vector& beta,
		     double threshold,
		     IRowSet& drs )
    {
        if(!active) return;         //--->>--

	// [added by Shenzhi] for Task 2, task 3
	/*
	  a sample BBRtrain model file:

	  Bayesian Binary Regression ver 2
	  tfMethod 0
	  idfMethod 0
	  cosineNormalize 0
	  featRestrict 75 98 115   
	  endofheader
	  modelname MODELNAME // added in task 3
	  topic <class>
	  modelType 1 1 0 0
	  design 0
	  topicFeats 75 98 115
	  beta 0.3 0.2 2.2 -11.0
	  threshold 0.000000000000e+00
	  endoftopic

	  Note1: the topicFeats should include 1. active features in training examples ;
	  2. features not-active in training examples, but have non-zero modes in indprior file;

	  Note2: For the features like Note1.(2), they do not need to participate in fitting. 

	 */

	// write out the featRestrict line; SL v3.03
	// get the struct which keeps the non-zero mode feature info;
	const IndPriorNonZeroModeFeats& nonzeroModes = drs.getNonzeroModes();
	// get the number of features declared using coefficient-level (priority level 6 in defaults)
	unsigned count = nonzeroModes.count;
	// to record the unique non-zero mode features; one may appear in both level 6 (coefficient) and level 4 (feature)
	map<int,double> mfeats;

	for(unsigned i=0; i<nonzeroModes.nonzeros.size(); i++) {
	    if(i<count && ( nonzeroModes.nonzeros.at(i).cls==-1 || nonzeroModes.nonzeros.at(i).cls==0) )
		mfeats[nonzeroModes.nonzeros.at(i).feat] = nonzeroModes.nonzeros.at(i).val;
	    else if( i>=count && mfeats.find(nonzeroModes.nonzeros.at(i).feat)==mfeats.end())
		mfeats[nonzeroModes.nonzeros.at(i).feat] = nonzeroModes.nonzeros.at(i).val;
	}
	vector<int>* featrestrict = new vector<int>();
	for(map<int,double>::iterator mitert=mfeats.begin();mitert!=mfeats.end();mitert++)
	    featrestrict->push_back(mitert->first);
	featrestrict->insert(featrestrict->end(), feats.begin(), feats.end());
	sort(featrestrict->begin(),featrestrict->end());
	WriteFeats(*featrestrict);

	file<<KW_endofheader<<endl;

	// for task 3
	file<<KW_modelname<<" "<<modelname<<endl;

	// write the "topic" line  
        file<<KW_topic<<" "<<topic<<endl;
	// write the "modelType" line
        file<<KW_modelType 
            <<" "<<modelType.Link()
            <<" "<<modelType.Optimizer()
            <<" "<<modelType.TuneThreshold()
            <<" "<<modelType.Standardize()
            <<" "<<modelType.StringParam()
            <<endl;
	// write "design" line
        file<<KW_design<<" "<<design.DesignType()<<endl;

	// write out the topicFeats and beta lines, only outputting the nonzeroes feats and betas
	vector< pair<int,double> > sparseFeatureBeta;
	// for each feature, if the corresponding beta is close enough to 0.0, discard it; otherwise, keep it;
	map<int,double>::iterator miter = mfeats.begin();
	for(unsigned i=0; i<feats.size(); i++) {
	    // v3.03
	    while(miter!=mfeats.end() && miter->first < feats.at(i)) {
		sparseFeatureBeta.push_back(pair<int,double>(miter->first, miter->second));
		miter++;
	    }
	    if(beta.at(i)!=0.0) // abandon the AlmostEqual strategy;  Shenzhi 12Mar07
		sparseFeatureBeta.push_back(pair<int,double>(feats.at(i), beta.at(i)));
	}
	while(miter!=mfeats.end()) {
	    sparseFeatureBeta.push_back(pair<int,double>(miter->first,miter->second));
	    miter++;
	}
				       

	// write out the SPARSE topicFeats line
        file<<KW_topicFeats;
        for( unsigned i=0; i<sparseFeatureBeta.size(); i++ )
            file<<" "<<sparseFeatureBeta.at(i).first;
        file<<endl;

	// write the SPARSE beta line 
        file<<KW_beta <<setiosflags(ios_base::scientific) <<setprecision(12);
	for (unsigned i=0; i<sparseFeatureBeta.size(); i++)
	    file<<" "<<sparseFeatureBeta.at(i).second;
	// beta line should have one more @constant value than topicFeats line
        file<<" "<<beta.at(beta.size()-1)<<endl;

	// [end of adding]

        file<<KW_threshold <<setiosflags(ios_base::scientific)<<setprecision(12)<<" "<<threshold<<endl;

        EOTopic();
    }

    void WriteTopicHierSparse( const string& topic, 
			       const string& modelname,
			       const ModelType& modelType,
			       const DesignParameter& design,
			       const vector<int>& feats,
			       const vector< vector<double> >& beta,
			       double threshold,
			       IRowSet& drs )
    {
        if(!active) return;         //--->>--


	// commented by Shenzhi for task 2
	/*
        WriteModelHeader( topic, modelType, design, feats );

        file<<setiosflags(ios_base::scientific) <<setprecision(12);

        for( unsigned j=0; j<beta.size(); j++ ) { //--feats--
            vector< pair<unsigned,double> > sparseBeta;
            for( unsigned g=0; g<beta.at(j).size(); g++ ) //build sparse for the feature
                if( beta.at(j).at(g) != 0 )
                    sparseBeta.push_back( pair<unsigned,double>(g,beta.at(j).at(g)) );
            if( sparseBeta.size() > 0 ) {  //save all about the feature to file
                file<<KW_betaHierSparse<<" "<<j;
                for( unsigned gsp=0; gsp<sparseBeta.size(); gsp++ )
                    file<<" "<<sparseBeta.at(gsp).first<<":"<<sparseBeta.at(gsp).second;
                file<<endl;
            }
        } //--feats--
	*/


	// write out the featRestrict line; SL v3.03
	const IndPriorNonZeroModeFeats& nonzeroModes = drs.getNonzeroModes();
	set<int> sfeats;
	vector<int> vpos;
	unsigned count = nonzeroModes.count;
	for(unsigned i=1; i<nonzeroModes.nonzeros.size(); i++) {
	    pair<set<int>::iterator, bool> retval;
	    retval =  sfeats.insert(nonzeroModes.nonzeros.at(i).feat);
	    if ( i<count || (i>=count && retval.second) )
		vpos.push_back(i); 
	}
	vector<int>* featrestrict = new vector<int>();
	for(set<int>::iterator siter=sfeats.begin();siter!=sfeats.end();siter++)
	    featrestrict->push_back(*siter);
	featrestrict->insert(featrestrict->end(), feats.begin(), feats.end());
	sort(featrestrict->begin(),featrestrict->end());
	WriteFeats(*featrestrict);


	file<<KW_endofheader<<endl;

	// [added by shenzhi for task 2 & 3]
	// for task 3
	file<<KW_modelname<<" "<<modelname<<endl;

	// write the "topic" line  
        file<<KW_topic<<" "<<topic<<endl;
	// write the "modelType" line
        file<<KW_modelType 
            <<" "<<modelType.Link()
            <<" "<<modelType.Optimizer()
            <<" "<<modelType.TuneThreshold()
            <<" "<<modelType.Standardize()
            <<" "<<modelType.StringParam()
            <<endl;
	// write "design" line
        file<<KW_design<<" "<<design.DesignType()<<endl;


        file<<setiosflags(ios_base::scientific) <<setprecision(12);

	set<int> sparseFeats;
	ostringstream betastr;
        for( unsigned j=0; j<beta.size(); j++ ) { //--feats--
            vector< pair<unsigned,double> > sparseBeta;
	    bool sparsefeat = false;
            for( unsigned g=0; g<beta.at(j).size(); g++ )  //build sparse for the feature 
                if( beta.at(j).at(g) != 0 ) {
                    sparseBeta.push_back( pair<unsigned,double>(g,beta.at(j).at(g)) );
		    if(!sparsefeat && j!=beta.size()-1) {
			sparseFeats.insert(feats.at(j));
			sparsefeat = true;
		    }
		}
            if( sparseBeta.size() > 0 ) {  //save all about the feature to file
                betastr<<KW_betaHierSparse<<" "<<j;
                for( unsigned gsp=0; gsp<sparseBeta.size(); gsp++ )
                    betastr<<" "<<sparseBeta.at(gsp).first<<":"<<sparseBeta.at(gsp).second;
                betastr<<endl;
            }
        } 

	// write out the SPARSE topicFeats line
        file<<KW_topicFeats;
        for( set<int>::iterator siter=sparseFeats.begin(); siter!=sparseFeats.end(); siter++)
            file<<" "<<*siter;
	for(unsigned i=0; i<vpos.size(); i++)
	    file<<" "<<nonzeroModes.nonzeros.at(vpos.at(i)).feat;
        file<<endl;

	file<<betastr;

	// [end of adding]

        file<<KW_threshold <<setiosflags(ios_base::scientific)<<setprecision(12)<<" "<<threshold<<endl;

        EOTopic();
    }

    // a better ctor
    WriteModel(string filename_, const DataFactory& df )
        : filename(filename_), ntopics(0)
    {
        active = 0<filename.size();
        if(!active) return;         //--->>--
        file.open(filename.c_str());
        file<<TITLE<<ver<<endl;
        if( !file.good() )
            throw runtime_error(string("Error creating model file '")+filename+"'");

        if( df.NGroups() > 0 )
            file<<KW_ngroups<<" "<<df.NGroups()<<endl;
        WriteTFIDFparameter( df.tfidfParam() );

        //WriteFeats( df.Feats() );   // SL v3.03; moved to modelfile.writetopic()

        if( df.tfidfParam().idfMethod() != NOIDF )
            WriteIDF( df.Idf() );

        // file<<KW_endofheader<<endl;  // v3.03, moved to modelfile.writetopic()
        file.flush();
    }
private:


    void WriteModelHeader( const string& topic, 
        const ModelType& modelType,
        const DesignParameter& design,
        const vector<int>& feats )
    {
        if(!active) return;         //--->>--

        file<<KW_topic<<" "<<topic<<endl;
        file<<KW_modelType 
            <<" "<<modelType.Link()
            <<" "<<modelType.Optimizer()
            <<" "<<modelType.TuneThreshold()
            <<" "<<modelType.Standardize()
            <<" "<<modelType.StringParam()
            <<endl;
        file<<KW_design<<" "<<design.DesignType()<<endl;

        file<<KW_topicFeats;
        for( unsigned i=0; i<feats.size(); i++ )
            file<<" "<<feats[i];
        file<<endl;
    }
    void EOTopic() {
        file<<KW_endoftopic<<endl;
        file.flush();
        ntopics ++;
    }
    void WriteTFIDFparameter( const TFIDFParameter& tfidf )
    {
        if(!active) return;         //--->>--
        file<<KW_tfMethod<<" "<<tfidf.tfMethod()<<endl;
        file<<KW_idfMethod<<" "<<tfidf.idfMethod()<<endl;
        file<<KW_cosNorm<<" "<<tfidf.cosineNormalize()<<endl;
        file.flush();
    }
    void WriteFeats( const vector<int>& feats )
    {
        if(!active) return;         //--->>--
        file<<KW_featRestrict;
        for( unsigned i=0; i<feats.size(); i++ )
            file<<" "<<feats[i];
        file<<endl;
        file.flush();
    }
    void WriteFeats( int from, int to )  {
        if(!active) return;         //--->>--
        file<<KW_featRestrictRange<<" "<<from<<" "<<to<<endl;
        file.flush();
    }
    void WriteIDF( const vector<double>& idf )
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
