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
        if( prob ) *pf << p_hat<<" ";
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
