#ifndef External_Score_File_
#define External_Score_File_

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#ifdef USE_LEMUR
# include "Param.hpp"
#endif //USE_LEMUR
#include "logging.h"

using namespace std;

//------------parameters--------------

class ExtScoreParam {
public:
    //access
    string trainFile() const { return trainFileName; }
    string testFile() const { return testFileName; }
    string labelsFile() const { return labelsFileName; }
    unsigned fieldNo1() const { return scoreFieldNo1; };
    string reportTopic( const char* topic ) const {
        string report;
        const vector<string>& l = labels( topic );
        for( unsigned i=0;i<l.size(); i++ )
            report += l[i]+" ";
        return report; }
    const vector<string>& labels( const char* topic ) const {
        map<string, vector<string> >::const_iterator
            find = labelsByTopic.find( topic );
        if( find==labelsByTopic.end() )
            return empty;
        else
            return find->second;
    } 
    bool isValid() const { return labelsByTopic.size() > 0; }
    //input
#ifdef USE_LEMUR
    void get()
    {
        trainFileName = ParamGetString("ExtScore.trainFile","");
        testFileName = ParamGetString("ExtScore.testFile","");
        labelsFileName = ParamGetString("ExtScore.labelsFile","");
        scoreFieldNo1 = ParamGetInt("ExtScore.scoreFieldNo",4);
        if( 4!=scoreFieldNo1/*score*/ && 3!=scoreFieldNo1/*labes*/ )
            throw runtime_error("Wrong field number in external score files; should be 3 or 4");

        //initialize
        string buf;
        ifstream lf(labelsFileName.c_str());
        // each row is: <topic> [label]*
        while( lf ) {//by topic
            vector<string> labels;
            getline( lf, buf ); 
            istringstream istr(buf);
            string topic;
            istr>>topic;
            if( istr ) {
                string lbl;
                while( istr>>lbl )
                    labels.push_back(lbl);
                labelsByTopic[topic] = labels;
            }
        }//by topic
    }
#endif //USE_LEMUR
private:
    string trainFileName, testFileName, labelsFileName;
    unsigned scoreFieldNo1;
    map<string, vector<string> > labelsByTopic;
    vector<string> empty;
};

inline std::ostream& operator<<( std::ostream& o, const ExtScoreParam& p ) {
    o << "External scores Train: "<<p.trainFile()
        <<"\t Test: "<<p.testFile()
        <<"\t Labels file: "<<p.labelsFile();
    if( 4!=p.fieldNo1() )
        o << "\tUse judgement as score"<<p.trainFile();
    return o;
}

//------------operational classes-----------------
class FScore {
public:
    struct Record{
        string label;
        string docid;
        bool y;
        double value;   };
private:
    string fname;
    string label;
    unsigned scoreFieldNo1;
    ifstream* pifs;
    //double value;
    struct Record record;
    bool started; //to support read-ahead strategy
    void start() { 
        pifs = new ifstream(fname.c_str());
        while( (*pifs>>record) && record.label!=label );
        started = true;
    }

public:
    FScore(string fname_, string label_, unsigned scoreFieldNo1_) 
        : fname(fname_), label(label_), scoreFieldNo1(scoreFieldNo1_),
        started(false) {}
    ~FScore()    {
        if(started)  delete pifs;
    }
    bool next() {
        if(started) 
            *pifs >> record;
        else
            start();
        if( *pifs && record.label==label ) {
            return true;
        }
        else return false;
    }
    void rewind() { delete pifs; started=false; }
    double val() { return 4==scoreFieldNo1 ? record.value : record.y; }
    //virtual ~FScore() { delete pifs; }
    friend istream& operator>>(istream& i, Record& r);
    friend ostream& operator<<(ostream& o, const Record& r);
};
inline istream& operator>>(istream& i, FScore::Record& r) {
    i>>r.label>>r.docid>>r.y>>r.value;
    return i; }
inline ostream& operator<<(ostream& o, const FScore::Record& r) {
    o<<r.label<<r.docid<<r.y<<r.value;
    return o; }

class ExtScores : public IScores {
    vector<FScore> s;
    void AddScore( string fname_, string label_, unsigned scoreFieldNo1 )  { 
        //Log(10)<<"Adding score "<<fname_<<" "<<label_;
        s.push_back( FScore( fname_, label_, scoreFieldNo1 ) ); }
public:
    ExtScores() {}
    ExtScores( const ExtScoreParam& extScoreParam, string topic, bool train ) {
        const vector<string>& labels = extScoreParam.labels( topic.c_str() );
        //Log(10)<<"\nExtScores, topic "<<topic<<",  labels:";
        for( unsigned i=0; i<labels.size(); i++ )
            AddScore( train ? extScoreParam.trainFile() : extScoreParam.testFile(), 
                labels[i], extScoreParam.fieldNo1() );
    }
    virtual unsigned n() const {return s.size();}
    virtual bool next() {
        for( unsigned i=0; i<s.size(); i++ )
            if( !s[i].next() )
                return false;
        return true;
    }
    virtual void rewind() { 
        for( unsigned i=0; i<s.size(); i++ ) s[i].rewind();
    }
    virtual double val(unsigned i) {
        if(i<s.size()) return s[i].val();
        else throw logic_error("FScore: only one score supported");
    }
    //friend ostream& operator<<( ostream& o, const ExtScores a );
};

#endif //External_Score_File_


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
