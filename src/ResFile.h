#ifndef RESULTS_FILE_
#define RESULTS_FILE_

#include <fstream>
#include <string>
using namespace std;

class ResultsFile {
    bool multiTopic;
    bool rowId;
    bool prob;
    std::string resFName;  
    ofstream *pf;
public:
    void writeline( string topic, string rowName, bool isTest, bool y, 
        double score, double p_hat, bool prediction ) 
    {
        if( multiTopic )  *pf << topic <<" ";
        if( rowId )  *pf << rowName<<" "; //doc id
        if( multiTopic )  *pf << isTest<<" "<< y <<" ";
        if( prob ) *pf << setprecision(12) << p_hat<<" ";
        else   *pf << score<<" ";
        if( multiTopic )  *pf<< prediction; //y_hat boolean
        else   *pf<<( prediction ? 1 : -1 ); //y_hat
	    *pf<<endl;
    }
    bool MultiTopic() const { return multiTopic; }
    bool RowId() const { return rowId; }
    bool Prob() const { return prob; }
    const string& FName() const { return resFName; }
    ResultsFile() //disfunctional ctor
        : multiTopic(false), rowId(false), prob(false), 
        pf(0)   {};
    ResultsFile( bool multiTopic_, bool rowId_, bool prob_, std::string resFName_)
        : multiTopic(multiTopic_), rowId(rowId_), prob(prob_), resFName(resFName_), 
        pf(0)   {};
    ~ResultsFile() {
        delete pf;   }
    void start()
    {
        pf = new ofstream(resFName.c_str());
        if(multiTopic)
            if( prob )
                *pf <<"topic docId isTest label p_hat y_hat"<<endl;
            else
                *pf <<"topic docId isTest label score y_hat"<<endl;
    }
};

#endif //RESULTS_FILE_

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
