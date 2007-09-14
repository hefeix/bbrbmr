#define  _USE_MATH_DEFINES
#include <limits>
#include <iomanip>
#include <process.h>

#include "logging.h"
#include "dataBin.h"
#include "bayes.h"
#include "ModelTypeParam.h"
#include "StrataSplit.h"
using namespace std;

void runLIBSVM_cv( const char* topic,
                       RowSetMem & trainData,
                       const class ModelType& modelType,
                       HyperParamPlan& hyperParamPlan,
                       IRowSet & testData,
                       std::ostream& modelFile,
                       std::ostream& result)
{
#ifdef LAUNCH_SVM
    const char* learnApp = "svmtrain-cv.exe";
    const char* classifyApp = "svmpredictscore.exe";
    const char* trainFName = "SVM_Train.dat";
    const char* testFName = "SVM_Test.dat";
    const char* modelFName = "SVM.Model";
    const char* predictFName = "SVM_Predict.dat";
    int ret;

    // write training file for SVMlight
    BoolVector y( false, trainData.n() );
    ofstream ftrain(trainFName);
    ftrain<<setprecision(10);
    unsigned r=0;
    while( trainData.next() ) {
        ftrain<<( trainData.y() ? 1 : -1 );
        const SparseVector x = trainData.xsparse();
        for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
            ftrain<<" "<<ix->first<<":"<<ix->second;
        ftrain<<endl;
        y[r++] = trainData.y();
    }
    ftrain.close();

    //additional parameter(s)
    std::ostrstream sparam;
    double odds = double(ntrue(y)) / (trainData.n()-ntrue(y));
    if( 0==strcmp("balance",modelType.StringParam().c_str()) ) {
        // http://www.cs.cornell.edu/People/tj/publications/morik_etal_99a.pdf - ref from SVMlight:
        //  C+ / C- = number of negative training examples / number of positive training examples
        //sparam<<" -w1 "<< 1/odds <<std::ends;
        //should we treat it as inverse?
        sparam<<" -w1 "<< odds <<std::ends;
        Log(3)<<"\nBalanced training: w1=(odds of positive)= "<<odds;
    }
    else
        sparam<<modelType.StringParam()<<std::ends;

    // Hyperparameter loop
    unsigned bestParamIndex = unsigned(-1);
    if( hyperParamPlan.plan().size() > 1 ) {
        vector<double> cvres;
        for( unsigned iparam=0; iparam<hyperParamPlan.plan().size(); iparam++ )//hyper-parameter loop
        {
            double hpvalue = hyperParamPlan.plan()[iparam];
            Log(5)<<"\nHyperparameter plan #"<<iparam+1<<" value="<<hpvalue;
            std::ostrstream cvparam;
            cvparam<<sparam.str()<<" -v "<<hyperParamPlan.nfolds()<<" -c "<<hpvalue<<std::ends;
            Log(5)<<"\n\nLaunch LIBSVM cv learning - Time "<<Log.time()
                <<"\n  command line: "<<learnApp<<" "<<cvparam.str()<<" "<<trainFName<<" "<<modelFName<<"\n";
            Log(5).flush();
            try{
            ret = _spawnlp( _P_WAIT, learnApp, learnApp, cvparam.str(), trainFName, modelFName, NULL );
            }catch(...){
                Log(1)<<"\nLIBSVM learning exception";
                continue;
            }
            if( 0!=ret ){
                Log(1)<<"\nLIBSVM learning run-time error, return value="<<ret;
                continue;
            }
            Log(3)<<"\nEnd LIBSVM cv learning - Time "<<Log.time();
            
            ifstream accuracyfile(modelFName);
            double accuracy;
            accuracyfile>>accuracy;
            cvres.push_back(accuracy);
            accuracyfile.close();
        }
        // best by cv
        double bestEval = - numeric_limits<double>::max();
        for( unsigned i=0; i<cvres.size(); i++ )
            if( cvres[i]>bestEval ) {
                bestEval = cvres[i];
                bestParamIndex = i;
            }
        if( bestParamIndex==unsigned(-1) )
            throw runtime_error("No good hyperparameter value found");
        Log(5)<<"\nBest parameter value "<<hyperParamPlan.plan()[bestParamIndex]<<" cv average accuracy "<<bestEval;
    }
    else //only one choice
        bestParamIndex = 0;

    // launch SVM final
    std::ostrstream finparam;
    finparam<<sparam.str()<<" -c "<<hyperParamPlan.plan()[bestParamIndex]<<std::ends;
    Log(5)<<"\n\nLaunch LIBSVM final learning - Time "<<Log.time()
        <<"\n  command line: "<<learnApp<<" "<<finparam.str()<<" "<<trainFName<<" "<<modelFName<<"\n";
    Log(5).flush();
    try{
    ret = _spawnlp( _P_WAIT, learnApp, learnApp, finparam.str(), trainFName, modelFName, NULL );
    }catch(...){
        throw runtime_error("LIBSVM learning exception"); }
    if( 0!=ret )
        throw runtime_error("SVM learning run-time error");
    Log(5)<<"\nEnd SVM final learning - Time "<<Log.time();

    // append model file
    modelFile<<"Topic: "<<topic<<endl;
    ifstream tmpMdl(modelFName);
    string buf;
    while( getline( tmpMdl, buf ) )
        modelFile<<buf<<endl;
    tmpMdl.close();
    modelFile.flush();

    // resubstitution - evaluate
    Log(5)<<"\n\nLaunch LIBSVM classify on training sample - Time "<<Log.time()
            <<"\n  command line: "<<classifyApp<<" "<<trainFName<<" "<<modelFName<<" "<<predictFName<<endl;
    Log(5).flush();
    try{
    ret = _spawnlp( _P_WAIT, classifyApp, classifyApp, trainFName, modelFName, predictFName, NULL );
    }catch(...){
        throw runtime_error("LIBSVM classify exception"); }
    if( 0!=ret )
        throw runtime_error("LIBSVM-classify run-time error");
    Log(5)<<"\nEnd SVM classify on training sample - Time "<<Log.time();

    vector<double> resubstScore( trainData.n(), 0.0 );
    ifstream fresubst(predictFName);
    for( unsigned i=0; i<trainData.n(); i++ ) {
        fresubst>>resubstScore[i];
    }
    fresubst.close();

    Log(3)<<std::endl<<"Built SVM model - - Time "<<Log.time();
    
    // tune threshold
    double threshold = tuneThreshold( resubstScore, y, modelType );
    Log(3)<<"\nTuned threshold="<<threshold;

    // resubstitution - evaluate
    BoolVector resubst( trainData.n() );
    for( unsigned i=0; i<trainData.n(); i++ )
        resubst[i] =( resubstScore[i]>=threshold );
    Log(3)<<"\n\n---Resubstitution results---";
    displayEvaluation( Log(3), y, resubst );

    Log(12)<<"Resubstitution Scores";
    for( unsigned i=0; i<trainData.n(); i++ )
        Log(12)<<endl<<resubstScore[i]<<":"<<resubst[i];

    // write results file
    trainData.rewind();
    for( unsigned r=0; trainData.next(); r++ ) {
        result
            << topic
            <<" " << trainData.currRowName() //doc id
            <<" " << false //isTest
	        <<" "<< y[r] //label
            <<" "<< resubstScore[r]
	        <<" "<< resubst[r] //y_hat
	        << endl;
    }
    if( Log.level()<=5 ) remove(trainFName);

    // write test file for SVMlight
    ofstream ftest(testFName);
    ftest<<setprecision(10);
    while( testData.next() ) {
        ftest<<( testData.y() ? 1 : -1 );
        const SparseVector x = testData.xsparse();
        for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
            ftest<<" "<<ix->first<<":"<<ix->second;
        ftest<<endl;
    }
    ftest.close();

    //% TEST
    Log(3)<<"\n\nLaunch LIBSVM classify on test sample - Time "<<Log.time()
            <<"\n  command line: "<<classifyApp<<" "<<testFName<<" "<<modelFName<<" "<<predictFName<<endl;
    Log(3).flush();
    try{
    ret = _spawnlp( _P_WAIT, classifyApp, classifyApp, testFName, modelFName, predictFName, NULL );
    }catch(...){
        throw runtime_error("LIBSVM classify exception"); }
    if( 0!=ret )
        throw runtime_error("LIBSVM classify run-time error");
    if( Log.level()<=5 ) remove(testFName);
    if( Log.level()<=5 ) remove(modelFName);
    Log(3)<<"\nEnd SVM classify on test sample - Time "<<Log.time();
    /*testModel( topic,
                modelType,
                beta,
                threshold,
                testData,
                result);*/
    int TP=0, FP=0, FN=0, TN=0;
        Log(8)<<"\nValidation scores ";
    double logLhood = 0;

    // evaluate, write res file
    testData.rewind();
    ifstream feval(predictFName);
    double predictScore;
    while( testData.next() ) //testData
    {
        feval >> predictScore;

        //% get the classes on the test data
        bool prediction=( predictScore>=threshold );
            Log(12)<<endl<<predictScore<<":"<<prediction;
        if( prediction )
            if( testData.y() )  TP++; //testData
            else           FP++;
        else
            if( testData.y() )  FN++; //testData
            else           TN++;

        // write results file
        result
            << topic
            <<" " << testData.currRowName() //doc id
            <<" " << true //isTest
	        <<" "<< testData.y() //label
            <<" "<< predictScore
	        <<" "<< ( predictScore>=threshold ) //y_hat
	        << endl;
    }//test data loop
    feval.close();
    remove(predictFName);

    Log(1)<<"\n\n---Validation results---";
    displayEvaluationCT( Log(1), TP, FP, FN, TN );

    Log(3)<<endl<<"Time "<<Log.time();
#else
    throw runtime_error("SVM supported for Windows only");
#endif
}

static void write_data_file( IRowSet & rs, const char* fname )
{
    ofstream f(fname);
    f<<setprecision(10);
    rs.rewind();
    while( rs.next() ) {
        f<<( rs.y() ? 1 : -1 );
        const SparseVector x = rs.xsparse();
        for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
            f<<" "<<ix->first<<":"<<ix->second;
        f<<endl;
    }
    f.close();
}


class HyperParLoop {
    vector<double> paramPlan;
    vector<double> accs;
public:
    const vector<double>& Accuracies() const { return accs; }
    //it's all in the ctor
    HyperParLoop( 
        const char* learnApp,
        const char* classifyApp,
        const char* trainFName,
        const char* testFName,
        const char* modelFName,
        const char* predictFName,
        string paramstr,
        const vector<double>& paramPlan_,
        double balance =1
        )   : //drs(drs_), modelType(modelType_), 
        paramPlan(paramPlan_)
    {        
        unsigned planSize = paramPlan.size();
        for( unsigned iparam=0; iparam<planSize; iparam++ )//hyper-parameter loop
        {
            Log(5)<<"\nHyperparameter plan #"<<iparam+1<<" value="<<(paramPlan[iparam]);
            std::ostrstream cvparam;
            cvparam<<paramstr<<" -v 1 -c "<<(paramPlan[iparam])<<std::ends;

            // learn and classify
            Log(3)<<"\n\nLaunch SVMlight learning - Time "<<Log.time()
                <<"\n  command line: "<<learnApp<<" "<<cvparam.str()<<" "<<trainFName<<" "<<modelFName<<"\n";
            Log(3).flush();
            int ret = _spawnlp( _P_WAIT, learnApp, learnApp, cvparam.str(), trainFName, modelFName, NULL );
            if( 0!=ret )
                throw runtime_error("SVMlight-learn run-time error");
            Log(3)<<"\n\nLaunch SVMlight-classify on cv sample - Time "<<Log.time()
                <<"\n  command line: "<<classifyApp<<" -v 1 "<<testFName<<" "<<modelFName<<" "<<predictFName<<endl;
            Log(3).flush();
            ret = _spawnlp( _P_WAIT, classifyApp, classifyApp, " -v 1 ", testFName, modelFName, predictFName, NULL );
            if( 0!=ret )
                throw runtime_error("SVMlight-classify run-time error");

            //accuracy
            unsigned n=0, ncorr=0;
            double neghingeloss = 0;
            ifstream predict(predictFName);
            ifstream test(testFName);
            int TP=0, FP=0, FN=0, TN=0;
            while( test.good() )
            {
                string buf;
                getline( test, buf );
                if( buf[0]=='#' )
                    continue; //look for not-a-comment line
                istringstream rowbuf( buf);//.str() );
                int y;
                rowbuf>>y;
                if( rowbuf.fail() ) //empty line
                    continue;

                double score;
                predict>>score;
                if( predict.fail() )
                    throw runtime_error("Corrupt cv prediction file");
                int yhat = score>=0 ? 1 : -1;

                n++;
                if( yhat==y ) ncorr++;
                if( yhat==1 )
                    if( y==1 )  TP++; //testData
                    else     FP++;
                else
                    if( y==1 )  FN++; //testData
                    else     TN++;
                if( y*score < 1 )
                    if( y==1 )  neghingeloss -= (1-y*score)*balance;
                    else        neghingeloss -= (1-y*score);
            }
            accs.push_back( neghingeloss/n );               //double(ncorr)/n );
            if( ncorr!=TP+TN ) Log(1)<<"\n***ERROR confusion tables calc"; /*paranoid*/
            Log(6)<<"\nCV split TP/FP/FN/TN "<<TP<<" "<<FP<<" "<<FN<<" "<<TN<<" Ave hinge loss "<<-neghingeloss;;

        }//hyper-parameter loop
    }
};//class HyperParLoop


void runSVMlight( const char* topic,
                       RowSetMem & trainData,
                       const class ModelType& modelType,
                       HyperParamPlan& hyperParamPlan,
                       IRowSet & testData,
                       std::ostream& modelFile,
                       std::ostream& result)
{
#ifdef LAUNCH_SVM
    const char* learnApp = "svm_learn.exe"; // /SVMlight/
    const char* classifyApp = "svm_classify.exe"; // /SVMlight/
    const char* trainFName = "SVMlight_Train.dat";
    const char* testFName = "SVMlight_Test.dat";
    const char* modelFName = "SVMlight_Model.dat";
    const char* predictFName = "SVMlight_Predict.dat";

    //additional parameter(s)
    std::ostrstream sparam;
    double balance = 1;
    if( 0==strcmp("balance",modelType.StringParam().c_str()) ) {
        // http://www.cs.cornell.edu/People/tj/publications/morik_etal_99a.pdf - ref from SVMlight:
        //  C+ / C- = number of negative training examples / number of positive training examples
        balance = double(trainData.n()-trainData.npos()) / trainData.npos();
        sparam<<" -j "<< balance <<std::ends;
        Log(3)<<"\nBalanced training: j=(odds of negative)= "<<balance;
    }
    else
        sparam<<modelType.StringParam()<<std::ends;

        //prepare CV
        unsigned planSize = hyperParamPlan.plan().size();
        vector<double> paramEval( planSize, 0.0 );
        const double randmax = RAND_MAX;
        unsigned nfolds = hyperParamPlan.nfolds(); //10
        if( nfolds>trainData.n() ) {
            nfolds = trainData.n();
            Log(1)<<"\nWARNING: more folds requested than there are data. Reduced to "<<nfolds;
        }
        unsigned nruns = hyperParamPlan.nruns(); //2
        if( nruns>nfolds )  nruns = nfolds;
        unsigned runsize = 0;

        vector<int> rndind = PlanStratifiedSplit( trainData, nfolds );

        vector<bool> foldsAlreadyRun( nfolds, false );

        //cv loop
        for( unsigned irun=0; irun<nruns; irun++ )
        {
            //next fold to run
            unsigned x = unsigned( rand()*(nfolds-irun)/(randmax+1) );
            unsigned ifold, y=0;
            for( ifold=0; ifold<nfolds; ifold++ )
                if( !foldsAlreadyRun[ifold] ) {
                    if( y==x ) break;
                    else y++;
                }
            foldsAlreadyRun[ifold] = true;
            Log(5)<<"\nCross-validation "<<nfolds<<"-fold; Run "<<irun+1<<" out of "<<nruns<<", fold "<<ifold;

            //training-test
            SamplingRowSet cvtrain( trainData, rndind, ifold, ifold+1, false );
            write_data_file( cvtrain, trainFName );
            Log(5)<<"\ncv training sample "<<cvtrain.n()<<" rows";
            SamplingRowSet cvtest( trainData, rndind, ifold, ifold+1, true );
            write_data_file( cvtest, testFName );
            Log(5)<<"\ncv test sample "<<cvtest.n()<<" rows";
            HyperParLoop hyperParLoop( 
                            learnApp,
                            classifyApp,
                            trainFName,
                            testFName,
                            modelFName,
                            predictFName,
                            sparam.str(),
                            hyperParamPlan.plan(),
                            balance );
        
            Log(5)<<"\nCross-validation test accuracy: ";
            unsigned prevrunsize = runsize;
            runsize += cvtest.n();
            for( unsigned iparam=0; iparam<planSize; iparam++ ) {
                double eval = hyperParLoop.Accuracies()[iparam];
                Log(5)<<eval<<" ";
                paramEval[iparam] = paramEval[iparam]*prevrunsize/runsize + eval*cvtest.n()/runsize; //weighted ave
            }

        }//cv loop

        // best by cv
        double bestEval = - numeric_limits<double>::max();
        unsigned bestParam = unsigned(-1);
        Log(4)<<"\nCross-validation results - hyperparameter values, cv average accuracy:";
        for( unsigned i=0; i<planSize; i++ ) {
            if( paramEval[i]>bestEval ) {
                bestEval = paramEval[i];
                bestParam = i;
            }
            Log(4)<<"\n\t"<<hyperParamPlan.plan()[i]<<"\t"<<paramEval[i];
        }
        if( bestParam==unsigned(-1) )
            throw runtime_error("No good hyperparameter value found");
        Log(3)<<"\nBest hyperparameter value "<<hyperParamPlan.plan()[bestParam]<<" cv-average accuracy "<<bestEval;

        //build final model
        std::ostrstream sparam2;
        sparam2<<sparam.str()<<" -c "<<hyperParamPlan.plan()[bestParam]<<std::ends;

    // write training file for SVMlight
    BoolVector y( false, trainData.n() );
    ofstream ftrain(trainFName);
    ftrain<<setprecision(10);
    unsigned r=0;
    while( trainData.next() ) {
        ftrain<<( trainData.y() ? 1 : -1 );
        const SparseVector x = trainData.xsparse();
        for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
            ftrain<<" "<<ix->first<<":"<<ix->second;
        ftrain<<endl;
        y[r++] = trainData.y();
    }
    ftrain.close();

    // launch SVMlight
    Log(3)<<"\n\nLaunch final SVMlight learning - Time "<<Log.time();
    Log(3)<<"\n  command line: "<<learnApp<<" "<<sparam2.str()<<" "<<trainFName<<" "<<modelFName<<"\n";
    Log(3).flush();
    int ret = _spawnlp( _P_WAIT, learnApp, learnApp, sparam2.str(), trainFName, modelFName, NULL );
    if( 0!=ret )
        throw runtime_error("SVMlight learning run-time error");
    Log(3)<<"\nEnd SVMlight learning - Time "<<Log.time();

    // append model file
    modelFile<<"Topic: "<<topic<<endl;
    ifstream tmpMdl(modelFName);
    string buf;
    while( getline( tmpMdl, buf ) )
        modelFile<<buf<<endl;
    tmpMdl.close();
    modelFile.flush();

    // resubstitution - evaluate
    Log(3)<<"\n\nLaunch SVMlight-classify on training sample - Time "<<Log.time()
        <<"\n  command line: "<<classifyApp<<" "<<trainFName<<" "<<modelFName<<" "<<predictFName<<endl;
    Log(3).flush();
    ret = _spawnlp( _P_WAIT, classifyApp, classifyApp, trainFName, modelFName, predictFName, NULL );
    if( 0!=ret )
        throw runtime_error("SVMlight-classify run-time error");
    Log(3)<<"\nEnd SVMlight-classify on training sample - Time "<<Log.time();

    vector<double> resubstScore( trainData.n(), 0.0 );
    ifstream fresubst(predictFName);
    for( unsigned i=0; i<trainData.n(); i++ ) {
        fresubst>>resubstScore[i];
    }
    fresubst.close();

    Log(3)<<std::endl<<"Built SVM model - - Time "<<Log.time();
    
    // tune threshold
    double threshold = tuneThreshold( resubstScore, y, modelType );
    Log(3)<<"\nTuned threshold="<<threshold;

    // resubstitution - evaluate
    BoolVector resubst( trainData.n() );
    for( unsigned i=0; i<trainData.n(); i++ )
        resubst[i] =( resubstScore[i]>=threshold );
    Log(3)<<"\n\n---Resubstitution results---";
    displayEvaluation( Log(3), y, resubst );

    Log(12)<<"Resubstitution Scores";
    for( unsigned i=0; i<trainData.n(); i++ )
        Log(12)<<endl<<resubstScore[i]<<":"<<resubst[i];

    // write results file
    trainData.rewind();
    for( unsigned r=0; trainData.next(); r++ ) {
        result
            << topic
            <<" " << trainData.currRowName() //doc id
            <<" " << false //isTest
	        <<" "<< y[r] //label
            <<" "<< resubstScore[r]
	        <<" "<< resubst[r] //y_hat
	        << endl;
    }
    if( Log.level()<=5 ) remove(trainFName);

    // write test file for SVMlight
    vector<bool> testY;
    ofstream ftest(testFName);
    ftest<<setprecision(10);
    while( testData.next() ) {
        testY.push_back( testData.y() );
        ftest<<( testData.y() ? 1 : -1 );
        const SparseVector x = testData.xsparse();
        for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ )
            ftest<<" "<<ix->first<<":"<<ix->second;
        ftest<<endl;
    }
    ftest.close();

    //% TEST
    Log(3)<<"\n\nLaunch SVMlight-classify on test sample - Time "<<Log.time()
        <<"\n  command line: "<<classifyApp<<" "<<testFName<<" "<<modelFName<<" "<<predictFName<<endl;
    Log(3).flush();
    ret = _spawnlp( _P_WAIT, classifyApp, classifyApp, testFName, modelFName, predictFName, NULL );
    if( 0!=ret )
        throw runtime_error("SVMlight-classify run-time error");
    if( Log.level()<=5 ) remove(testFName);
    if( Log.level()<=5 ) remove(modelFName);
    Log(3)<<"\nEnd SVMlight-classify on test sample - Time "<<Log.time();
    /*testModel( topic,
                modelType,
                beta,
                threshold,
                testData,
                result);*/
    int TP=0, FP=0, FN=0, TN=0;
        Log(8)<<"\nValidation scores ";
    double logLhood = 0;

    // evaluate, write res file
            //obsolete testData.rewind();
    ifstream feval(predictFName); //testData
    double predictScore;
    std::vector< std::pair<double,bool> > forROC;
    for( vector<bool>::const_iterator itrTestY=testY.begin(); itrTestY!=testY.end(); itrTestY++ )
    {
        feval >> predictScore;
        if( feval.fail() )
            throw runtime_error("SVMlight predictions file too short");

        // get the classes on the test data
        bool prediction=( predictScore>=threshold );
            Log(12)<<endl<<predictScore<<":"<<prediction;
        if( prediction )
            if( *itrTestY )  TP++; //testData
            else           FP++;
        else
            if( *itrTestY )  FN++; //testData
            else           TN++;
        forROC.push_back( pair<double,bool>(predictScore,*itrTestY) );

        // write results file
        result
            << topic
            <<" " << testData.currRowName() //doc id
            <<" " << true //isTest
	        <<" "<<  *itrTestY //label
            <<" "<< predictScore
	        <<" "<< ( predictScore>=threshold ) //y_hat
	        << endl;
    }//test data loop
    feval.close();
    //remove(predictFName);

    Log(1)<<"\n\n---Validation results---";
    displayEvaluationCT( Log(1), TP, FP, FN, TN );

    //roc
    double area = calcROC(forROC);
    Log(1)<<"\nROC area under curve "<<area;

    Log(3)<<endl<<"Time "<<Log.time();
#else
    throw runtime_error("SVMlight supported for Windows only");
#endif
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
