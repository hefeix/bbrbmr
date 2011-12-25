#ifdef GROUPS
# define VERSION "3.02"
#else
# define VERSION "2.10"
#endif

#ifdef ROWID //  Mar 29      PATCH: row id in the data file, option -R
# define VERSION "-R_" VERSION
#endif
// 0.3      Apr 04      speed up
// 0.31     Apr 04      default hyperparameter: no const term in avg x*x and no penalty for the intercept
// 0.4      Jun 04      individual priors (Lemur only)
// 0.41     Jul 13, 04  1) exponential/negative exponential priors (Lemur only)
//                      2) probability threshold for testing (Lemur only)
// 0.42     Jul 25, 04  bug fixed, seg fault (Lemur only)
// 0.43     Jul 26, 04  separate list of docs for IDF calculation (Lemur only)
// 0.50     Aug 24, 04  1) ind prior in command-line ver 
//                      2) ind prior for intercept 
//                      3) infinite prior variance
// 0.51     Aug 25, 04  standardization fixed 
// 0.52     Aug 30, 04  added threshold tuning based on estimated prob (V.M.)
// 0.53     Sep 02, 04  allow stdin for data in bbrtrain/bbrclassify (AG)
// 0.54     Sep 15, 04  linux bugs fixed: 1) stdin dtor  2)RAND_MAX (AG)
// 0.55     Oct 04, 04  precision/recall/F1 are 1 when denoms are zero (AG)
// 0.56     Oct 18, 04  fixed bug with empty vectors
// 0.57     Nov 04, 04  new random split for cv: fixed sizes; class-stratified; #folds <= #cases
// 0.58     Dec 01, 04  prior skew for "ordinary" variables in BBRtrain
//                      "=", not "-" argument indicates stdin for data 
// 0.59     Dec 08, 04  --in process-- prep for 2.0 release
// 2.01     Feb 08, 05  bugs in StrataSplit
// 2.02     Feb 14-23   reporting #feats left in model, ZO.cpp
//                      reporting ROC area for training, ZO.cpp
//                      one std error rule for hyperparameter cv (option --errbar)
//                      probability threshold option (--probthres)
//                      no iterations limit
//                      fixed bug with convergence parameter
//                      "-" argument indicates stdin for data, "=" also allowed 
// 2.03     Mar 01      reporting dropped intercept, ZO.cpp
// 2.04     Mar 03      accurate summation, OneCoordStep(), ZO.cpp
//                      new options: high accuracy mode; iter. limit 
// 2.05     Mar 29      comment line in data file, starts with '#'
// 2.06     Apr 21, 05  no double-try for Laplace at mode, //ZO.cpp
// 2.06a    May 05, 05  BayesParameter/HyperparamPlan classes refactored
// 2.07     May 16, 05  Groups - Hierarchical modeling
// 2.10     May 30, 05  StandardizedRowSet: check if stddev is zero //released as 2.04a
// 2.10     Jun 01, 05  DisplayModel: fixed bug with duplicate beta values; don't use std::map!
// 2.10     Jun 20, 05  one std error rule: fixed bug with stderr denom calc //ZO.cpp
// 2.10     Jun 20, 05  bug with ind.priors file read - fixed //DataFactory.cpp
//
// 2.50     Jun 20, 05  this ver is the umbrella for "Groups - Hierarchical modeling"
// 2.51     Jun 27, 05  bug: cv with default on 1st level - fixed
// 2.55     Jul 05, 05  hierarchical fitting init by ordinary fit
// 2.56     Jul 10, 05  hierarchical Laplace redesigned
// 2.57     Aug 15, 05  bug: zero ind.prior var with normal prior - fixed
// 2.58     Sep 06, 05  ModelFile: token-wise file read for very large data; end-of-line ignored!
// 2.60     Oct 05, 05  hyperparameter search without a grid
// 2.61     Nov 05, 05  final prior variance reporting for bootstrap
// 2.62     Nov 10, 05  hyperparameter autosearch: observations weighted by inverse stddev
// 3.00                 fixed log.prior with infinite prior var
// 3.01     Jan 31, 06  fixed bug: Gaussian penalty should be 1/(2*var), not 1/var //ZO.cpp
// 3.02     Sep 01, 11  watch for denormal or infinite limits during autosearch //HParSearch.cpp

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>
#define  _USE_MATH_DEFINES
#include <math.h>

using namespace std;

#ifdef USE_LEMUR
# include "IndexManager.hpp"  //Lemur
# include "TextQueryRep.hpp"  //Lemur
# include "BasicDocStream.hpp" //Lemur
# include "LemurData.h"
# include "Oracle.hpp"// was AdaptFilter
#endif //USE_LEMUR

#include "logging.h"
#include "FeatSelectParamManager.h"
#include "TFIDFParamManager.h"
#include "BayesParamManager.h"
#include "ModelTypeParam.h"
#include "dataBin.h"
#include "bayes.h"
#include "PriorTerms.h"
#include "ExtScore.h"
#include "DataFactory.h"
#include "ModelFile.h"
#include "Squeezer.h"
#include "ResFile.h"

logging Log(5);
static bool multiLabel;
static FeatSelectByTopic featSelectMetaParam;
static TFIDFParameter tfidfParameter;
static PriorTermsByTopic priorTermsByTopic; 
static HyperParamPlan hyperParamPlan;
static HyperParamPlan hyperParamPlan2; //between-group level of priors
static Squeezer squeezer;
static ModelType modelType;
static DesignParameter designParameter;
static ExtScoreParam extScoreParam;
static unsigned ngroups =0;

enum ETrainTestSplit { randomSplit=1, fileSplit=2, undef } 
    eTrainTestSplit;


#ifdef USE_LEMUR
static bool oracleNRelDefault; //no info in oracle means doc is non-relevant
static std::string databaseIndex;
static std::string trainIndex;
#endif //USE_LEMUR
static std::string trainQrelFile;
static std::string testQrelFile;
static std::string textQuerySet;
static std::string trainDocsFile;
static std::string testDocsFile;
static std::string idfDocsFile;

/*static std::string resultFile;  
static ResultFormat resultFormat;*/
static ResultsFile resFile;

static std::string bagOfWordsFile;
static std::string priorTermsFile;
static double priorMean;

static std::string  modelWriteFileName;
static std::string  modelReadFileName;

static std::string  trainPlainFile;
static std::string  testPlainFile;

static double probTestThreshold = -1;

static bool rowIdMode=false;

#ifdef USE_LEMUR

#define MULTI_TOPIC

void exhaustDocTerms( Document *d ) {
    d->startTermIteration();
    while (d->hasMore()) {
        TokenTerm *tt = d->nextTerm();
        //Log(12)<<endl<<tt->spelling();
    }
}

#endif //USE_LEMUR

int runBatch()
{
    try{

	class Index  *ind=0, *trainInd=0;

#ifdef USE_LEMUR
        if( !oracleNRelDefault )
            throw runtime_error("Should be <Oracle non-relevant by default> - that's the only option currently supported");
        if( databaseIndex.size()>0 )  {      //load Lemur stuff
            ind = IndexManager::openIndex(databaseIndex.c_str());
            Log(3)<<endl<<"Index; Docs: "<<ind->docCount()<<"  Terms: "<<ind->termCountUnique()
                <<"  Time "<<Log.time();
            trainInd = 0==trainIndex.size() ? 0     // is there train index?
                : IndexManager::openIndex(trainIndex.c_str());
            if( trainInd != 0)
                Log(3)<<endl<<"Train Index; Docs: "<<trainInd->docCount()<<"  Terms: "<<trainInd->termCountUnique()
                    <<"  Time "<<Log.time();
        }
#endif //USE_LEMUR

        //output
        resFile.start();
        /*ofstream result(resultFile.c_str());
        if( resProb==resultFormat )
            result <<"topic docId isTest label p_hat y_hat"<<endl;
        else if( resScore==resultFormat )
            result <<"topic docId isTest label score y_hat"<<endl;*/

        //input model file
        ReadModel readModel( modelReadFileName );
        if( readModel.Active() ) {
            tfidfParameter = readModel.tfidfParam();
        }

        DataFactory* dataFactory=0;
        if( eTrainTestSplit==randomSplit )
            //dataFactory = new RandomSplitDataFactory( ind, testQrelFile );
            throw logic_error("RandomSplitDataFactory has been deprecated");
        else
        if( 0==ind ) //plain files
#ifdef USE_LEMUR
            if( multiLabel )
                if( readModel.Active() ) //test only
                    dataFactory = new PlainFileDataFactory( 
                        testPlainFile.c_str(),
                        testQrelFile.c_str(),
                        tfidfParameter.tfMethod(), tfidfParameter.idfMethod(), tfidfParameter.cosineNormalize(),
                        readModel.Feats(),
                        readModel.Idf()
                    );
                else //train and test
                    dataFactory = new PlainFileDataFactory( 
                        trainPlainFile.c_str(),
                        testPlainFile.c_str(),
                        trainQrelFile.c_str(),
                        testQrelFile.c_str(),
                        tfidfParameter.tfMethod(), tfidfParameter.idfMethod(), tfidfParameter.cosineNormalize()
                    );
            else //single label - single "topic"
#endif //USE_LEMUR
                if( readModel.Active() ) //test only
                    dataFactory = new PlainFileYDataFactory( 
                        testPlainFile.c_str(), rowIdMode,
                        tfidfParameter.tfMethod(), tfidfParameter.idfMethod(), tfidfParameter.cosineNormalize(),
                        readModel.Feats(),
                        readModel.Idf(),
                        readModel.NGroups()
                    );
                else //train and test
                    dataFactory = new PlainFileYDataFactory( 
                        trainPlainFile.c_str(), 
                        testPlainFile.c_str(), rowIdMode,
                        tfidfParameter.tfMethod(), tfidfParameter.idfMethod(), tfidfParameter.cosineNormalize(),
                        ngroups
                    );
#ifdef USE_LEMUR
        else //Lemur
        if( readModel.Active() )
            dataFactory = new LemurDataFactory( 
                ind,
                testDocsFile.c_str(),
                testQrelFile.c_str(),
                tfidfParameter.tfMethod(), tfidfParameter.idfMethod(), tfidfParameter.cosineNormalize(),
                readModel.Feats(),
                readModel.Idf()
            );
        else if( 0==trainInd ) //single index, separate qrels, separate doc lists -- like OldReuters
            dataFactory = new LemurDataFactory( 
                ind,
                trainDocsFile.c_str(),
                testDocsFile.c_str(),
                trainQrelFile.c_str(),
                testQrelFile.c_str(),
                bagOfWordsFile.c_str(),
                tfidfParameter.tfMethod(), tfidfParameter.idfMethod(), tfidfParameter.cosineNormalize(),
                idfDocsFile
            );
        else //separate train index, single qrel, doc list for test only -- like RCV1-v2
            dataFactory = new LemurDataFactory(
                ind, 
                trainInd, 
                testDocsFile.c_str(),
                trainQrelFile.c_str(),
                bagOfWordsFile.c_str(),
                tfidfParameter.tfMethod(), tfidfParameter.idfMethod(), tfidfParameter.cosineNormalize(),
                idfDocsFile
             );
        delete trainInd;
#endif //USE_LEMUR
        //Log(8)<<"\nData factory; Time "<<Log.time()<<endl;

        //model file
        WriteModel* writeModel=0;
        ofstream* svmModelFile=0;
        if( modelType.Optimizer()==modelType.ZO ){ //ZO
            writeModel = new WriteModel( modelWriteFileName, *dataFactory );
        }
#ifdef LAUNCH_SVM
        else if( modelType.Optimizer()==modelType.SVMlight ){
            svmModelFile = new ofstream( modelWriteFileName.c_str() );
        }
#endif //LAUNCH_SVM

        //collect topics
        vector<string> allTopics;
        if( ! readModel.Active() ) {
#           ifdef MULTI_TOPIC
            if( multiLabel ) {
               DocStream *qryStream = new BasicDocStream(textQuerySet.c_str());
                Log(4)<<endl<<"Query Stream; Time "<<Log.time()<<endl;
                qryStream->startDocIteration();
                while (qryStream->hasMore()) {
                    Document *topicDoc = qryStream->nextDoc(); 
                    allTopics.push_back( topicDoc->getID() );
                    exhaustDocTerms( topicDoc );
                }
                delete qryStream;
            }
            else
#           endif //MULTI_TOPIC
                allTopics.push_back("<class>"); //dummy name
        }

        //topic loop
        for( unsigned it=0; true; it++ ) { //topic loop

            string topic;
            if( readModel.Active() )
                if( readModel.NextTopic() )
                    topic = readModel.Topic();
                else
                    break;              //----->>--
            else
                if( it < allTopics.size() )
                    topic = allTopics[it];
                else
                    break;              //----->>--

           if( multiLabel ) { //allTopics.size()>1 || readModel.Active() )
                Log(2)<<"\n\nTopic: "<<topic;
                if( allTopics.size()>1 )
                    Log(2)<<" "<<it+1<<" of "<<allTopics.size();
            }
            //Log(2)<<"  Time "<<Log.time()<<endl;
            Log(99)<<"\nExternal score labels: "<<extScoreParam.reportTopic(topic.c_str());

            FeatSelectParameter featSelectParameter = featSelectMetaParam.GetForTopic( topic.c_str() );

            //------------------------
            try{
                if( modelType.Optimizer()==modelType.ZO ){ //ZO

                    ZOLRModel model; //( dataFactory->TopicFeats() );
                    RowSetMem* TrainRowSet =0;
                    IRowSet* TestRowSet =0;

#if !defined(TEST_ONLY)
                    if( ! readModel.Active() ) { //train
                        TrainRowSet = dataFactory->NewTrainRowSet(
                                        topic.c_str(),
                                        featSelectParameter,
                                        priorTermsByTopic.GetForTopic(topic),
                                        extScoreParam );

                        // Listing the greater "training population", if needed 
                        // for thresholding option (B.2). (--VM)
                        IRowSet* TrainPopRowSet = 0;
                        if (modelType.TuneEst() ) {
#ifdef USE_LEMUR
                            TrainPopRowSet = 
	                            ((LemurDataFactory*)dataFactory)->
	                            NewTrainPopRowSet(
			                            topic.c_str(), 
			                            // readModel.TopicFeats(), 
			                            extScoreParam , idfDocsFile );
                            //Stats stat1(*TrainPopRowSet );
                            //Log(4) << "\nB2: runBatch(): |trainPopData| = " << stat1.AvgSquNorm();

#else
                            throw logic_error("Tuning based on estimated prob is not supported w/o LEMUR");
#endif
                        }

                        model.Train( topic.c_str(),
                            *TrainRowSet,
                            TrainPopRowSet,
                            hyperParamPlan,
                            hyperParamPlan2,
                            squeezer,
                            priorTermsByTopic,
                            designParameter,
                            modelType,
                            *writeModel,
                            resFile );
                        delete TrainPopRowSet;
                    }
                    Log(3)<<"\nTime "<<Log.time();
#endif //!TEST_ONLY
#if !defined(TRAIN_ONLY)
                    if( readModel.Active() ) {
                        TestRowSet = dataFactory->NewTestRowSet( topic.c_str(), readModel.TopicFeats(), extScoreParam  );
                        //Log(8)<<"\nTestRowSet, Dim "<<TestRowSet->dim();
                        model.Restore( readModel, *TestRowSet );
                    }
                    else
                        TestRowSet = dataFactory->NewTestRowSet( topic.c_str(), extScoreParam  );

                    if( probTestThreshold>=0 ) //use this param
                        model.Test( *TestRowSet, resFile, probTestThreshold );
                    else
                        model.Test( *TestRowSet, resFile );
#endif //!TRAIN_ONLY
                    delete TrainRowSet; //don't delete earlier: 'IDesign' depends on it
                    delete TestRowSet;
                }
#if !defined(TRAIN_ONLY) && !defined(TEST_ONLY) //not for the public version
#ifdef EM_ENABLED
                else if( modelType.Optimizer()==modelType.EM ) 
                {
                    if( extScoreParam.isValid() )
                        throw runtime_error("External scores not supported for EM!");
                    DenseData* TrainData = dataFactory->NewTrainData(
                                    topic.c_str(),
                                    featSelectParameter,
                                    priorTermsByTopic.GetForTopic(topic) );
                    if( 0==TrainData->nRel() ) {
                        Log(1)<<"No relevant docs in the topic; skipping topic"<<endl;
                        continue;
                    }

                    IRowSet* TestRowSet = dataFactory->NewTestRowSet( topic.c_str(), extScoreParam );
                        Log(4)<<"\nTestRowSet; Time "<<Log.time()<<endl;

                    runSparseEMprobit( topic.c_str(),
                        *TrainData,
                        hyperParamPlan.bp(), //bayesParameter,
                        priorTermsByTopic,
                        designParameter,
                        modelType,
                        *TestRowSet,
                        result);
                    delete TrainData;
                }
#endif //EM_ENABLED
#ifdef LAUNCH_SVM
                else if( modelType.Optimizer()==modelType.SVMlight ){
                    RowSetMem* TrainRowSet = dataFactory->NewTrainRowSet(
                                    topic.c_str(),
                                    featSelectParameter,
                                    priorTermsByTopic.GetForTopic(topic),
                                    extScoreParam );
                    IRowSet* TestRowSet = dataFactory->NewTestRowSet( topic.c_str(), extScoreParam  );
                        Log(4)<<"\nTestRowSet; Time "<<Log.time()<<endl;
                    ofstream result(resFile.FName().c_str());
                    if( resFile.Prob() ) result<<"topic docId isTest label p_hat y_hat"<<endl;
                    else result <<"topic docId isTest label score y_hat"<<endl;
                    runSVMlight(  topic.c_str(),     //runSVM_cv
                        *TrainRowSet,
                        modelType,
                        hyperParamPlan,
                        *TestRowSet,
                        *svmModelFile,
                        result);

                    delete TrainRowSet;
                    delete TestRowSet;
                }
#endif //LAUNCH_SVM

#endif //!defined(TRAIN_ONLY) && !defined(TEST_ONLY) //not for the public version
            }catch( std::exception& e){
                Log(0)<<std::endl<<"***Exception: "<<e.what();
                cerr<<std::endl<<"***Exception: "<<e.what();
            }

        }//topic loop

        delete writeModel;
        //delete readModel;
        delete svmModelFile;

        //delete qrel;
        delete dataFactory;
#ifdef USE_LEMUR
        delete ind;
#endif //USE_LEMUR
        Log(0)<<endl;
        return 0;

    }catch( std::exception& e){
        Log(0)<<std::endl<<"***Exception: "<<e.what();
        cerr<<std::endl<<"***Exception: "<<e.what();
#ifdef USE_LEMUR
    }catch (Exception &ex) { //Lemur
        Log(0)<<std::endl<<"***Lemur exception: ";
        ex.writeMessage( Log(0) );
#endif //USE_LEMUR
    }catch(...){
        Log(0)<<std::endl<<"***Unrecognized exception";
        cerr<<std::endl<<"***Unrecognized exception";
    }
    return 1;
}

#if defined(TRAIN_ONLY) || defined(TEST_ONLY) //public version

#include <tclap/CmdLine.h>
using namespace TCLAP;

#if defined(TRAIN_ONLY)

int main(int argc, char** argv)
{
	try {  

	// Define the command line object.
	CmdLine cmd(argv[0], "Bayesian Binary Regression - training", VERSION );

#ifdef ROWID
    SwitchArg rowIdArg("R","rowid","Row Id mode",false); cmd.add( rowIdArg );
#endif //ROWID
    ValueArg <int>  logArg("l","log","Log verbosity level",false,0, "[0..2]"); cmd.add( logArg );

    ValueArg<string> resfileArg("r","resultfile","Results file",false,"","resultfile"); cmd.add( resfileArg );

    ValueArg <int>  featSelectNumArg("f","features","Number of features to select",false,0,"integer"); cmd.add( featSelectNumArg );
    ValueArg <int>  featSelectUtilArg("u","utility",
        "Feature selection utility, 0-Corr 1-Yule's Q 2-Chi-square 3-BNS",false,0,"[0..3]"); cmd.add( featSelectUtilArg );

    ValueArg<double> probThrArg("","probthres","Probability threshold",false,0.5,"0<=p<=1"); cmd.add(probThrArg);
    ValueArg <int>  thrArg("t","threshold",
        "Threshold tuning, 0-no 1-sum error 2-T11U 3-F1 4-BER 5-T13U",false,0,"[0..5]"); cmd.add( thrArg );

    SwitchArg highAccuracyArg("","accurate","High accuracy mode",false); cmd.add( highAccuracyArg );
    ValueArg <unsigned> iterLimitArg("","iter","Max number of iterations",false,iterDefault,"integer"); cmd.add( iterLimitArg );
    ValueArg <double>  convergeArg("e","eps","Convergence threshold",false,convergeDefault,"float"); cmd.add( convergeArg );

    SwitchArg cosnormArg("c","cosnorm","Cosine normalization",false); cmd.add( cosnormArg ); //,0,"[0,1]"
    SwitchArg stdArg("s","standardize","Standardize variables",false); cmd.add( stdArg ); //,0,"[0,1]"

    ValueArg<string> indPriorsArg("I","indPriorsFile","Individual Priors file",false,"","indPriorsFile"); cmd.add(indPriorsArg);

    SwitchArg negOnlyArg("","neg","Negative only model coefficients", false); cmd.add( negOnlyArg );
    SwitchArg posOnlyArg("","pos","Positive only model coefficients", false); cmd.add( posOnlyArg );

    //-hierarchical modeling: group level prior
#ifdef GROUPS
    /* later:
    SwitchArg gnegOnlyArg("","gneg","Negative only model coefficients for group level", false); cmd.add( gnegOnlyArg );
    SwitchArg gposOnlyArg("","gpos","Positive only model coefficients for group level", false); cmd.add( gposOnlyArg ); */
    //no separate nfolds/nruns for levels
    ValueArg <string>  gpriorVarArg("","gvar",
        "Prior variance values for group level; if more than one, cross-validation will be used",false,"",
        "number[,number]*"); cmd.add( gpriorVarArg );
    /* later
    ValueArg <int>  gpriorArg("","gprior","Type of prior for group level, 1-Laplace 2-Gaussian",false,2,"[1,2]"); cmd.add( gpriorArg );*/
#endif //GROUPS

    //back-compatibility only -->
    ValueArg <string>  hypGridArg("S","search",
        "DEPRECATED Search for hyperparameter value",false,"","list of floats, comma-separated, no spaces"); cmd.add( hypGridArg );
    ValueArg <double>  hypArg("H","hyperparameter",
        "DEPRECATED Hyperparameter, depends on the type of prior", false,0,"float"); cmd.add( hypArg );
    //<--back-compatibility only

    SwitchArg errBarArg("","errbar","Error bar rule for cross-validation", false); cmd.add( errBarArg );
    SwitchArg pvarSearchArg("","autosearch","Auto search for prior variance, no grid required", false); cmd.add( pvarSearchArg );
    ValueArg <string>  cvArg("C","cv","Cross-validation",false,"10,10","#folds[,#runs]"); cmd.add( cvArg );
    ValueArg <string>  priorVarGridArg("V","variance",
        "Prior variance values; if more than one, cross-validation will be used",false,"",
        "number[,number]*"); cmd.add( priorVarGridArg );
    ValueArg <int>  priorArg("p","prior","Type of prior, 1-Laplace 2-Gaussian",false,2,"[1,2]"); cmd.add( priorArg );

#ifdef GROUPS
    ValueArg <int>  groupsArg("g","groups","Number of groups",false,0,"integer"); cmd.add( groupsArg );
#endif //GROUPS

    UnlabeledValueArg<string>  datafileArg("datafile","Data file; '-' for stdin","","datafile"); cmd.add( datafileArg );
    UnlabeledValueArg<string>  modelfileArg("modelfile","Model file","","modelfile"); cmd.add( modelfileArg );

	// Parse the args.
	cmd.parse( argc, argv );

    //----set parameters---
    Log(0)<<endl<<"Bayesian Binary Regression - Training. \tVer. "<<VERSION;
    Log(2)<<"\nCommand line: ";
    for( int i=0; i<argc; i++ )
        Log(2)<<" "<<argv[i];
    Log.setLevel( logArg.getValue() +5 );    
    Log(2)<<"\nLog Level: "<<Log.level()-5;

    multiLabel = 0;
    eTrainTestSplit = (ETrainTestSplit)fileSplit;

#ifdef GROUPS
    if( groupsArg.isSet() )  ngroups = groupsArg.getValue();
    else 
#endif //GROUPS
        ngroups = 0;

    {// Bayes parameters
        enum PriorType prior = PriorType(priorArg.getValue());
        if( prior!=1 && prior!=2 )
            throw runtime_error("Illegal prior type; should be 1-Laplace or 2-Gaussian");
        int skew = posOnlyArg.getValue() ? 1 : negOnlyArg.getValue() ? -1 : 0;
        if( priorVarGridArg.isSet() ) { // grid search
            if( pvarSearchArg.isSet() ) throw runtime_error("Incompatible arguments: auto search and grid");
            hyperParamPlan = HyperParamPlan( prior, skew, priorVarGridArg.getValue(), cvArg.getValue(),
                        HyperParamPlan::AsVar, errBarArg.getValue() );
        }
        else if( hypGridArg.isSet() ) //back-compatibility
            hyperParamPlan = HyperParamPlan( prior, skew, hypGridArg.getValue(), cvArg.getValue(),
                        HyperParamPlan::Native, errBarArg.getValue() );
        else if( hypArg.isSet() ) //fixed hyperpar - back-compatibility
            hyperParamPlan = HyperParamPlan( prior, hypArg.getValue(), skew );
        else if( pvarSearchArg.isSet() ) { // auto search, no grid
            hyperParamPlan = HyperParamPlan( prior, skew, cvArg.getValue() );
        }
        else //norm-based default hyperpar
            hyperParamPlan = HyperParamPlan( prior, skew );
        Log(2)<<endl<<hyperParamPlan;  //<<endl<<bayesParameter
    }
#ifdef GROUPS
    if( ngroups>0 ) {// Bayes parameters - between-group level
        /*later: 
        enum PriorType prior = gpriorArg.isSet() ? PriorType(gpriorArg.getValue())
            : hyperParamPlan.PriorType(); //by default, same as base level
        if( prior!=1 && prior!=2 )
            throw runtime_error("Illegal prior type for the group level; should be 1-Laplace or 2-Gaussian");
        int skew = gposOnlyArg.getValue() ? 1 : gnegOnlyArg.getValue() ? -1 : 0;*/
        if( gpriorVarArg.isSet() )  // new mode
            hyperParamPlan2 = HyperParamPlan( hyperParamPlan.PriorType(), //same as base level
                        0, //skew, 
                        gpriorVarArg.getValue(),
                        cvArg.getValue(), //same as for level 1
                        HyperParamPlan::AsVar, errBarArg.getValue() ); //errbar is same as base level
        else //auto-select hyperpar not allowed for level 2
            ;// means no hier modeling
        if( hyperParamPlan2.valid() )
            Log(2)<<endl<<"Groups level: "<<hyperParamPlan2;
    }
#endif //GROUPS
    if( indPriorsArg.isSet() ) {
        priorTermsByTopic = PriorTermsByTopic( indPriorsArg.getValue(), indPriorsModeRel );
        Log(2)<<endl<<priorTermsByTopic;
    }

    featSelectMetaParam = FeatSelectByTopic( 
        FeatSelectParameter::UtilityFunc(featSelectUtilArg.getValue()), 
        featSelectNumArg.getValue() );
    Log(2)<<endl<<featSelectMetaParam;

    tfidfParameter = TFIDFParameter( RAWTF, NOIDF, cosnormArg.getValue() );
    Log(2)<<"\nCosine normalization: "<<( cosnormArg.getValue() ? "Yes" : "No" );

    //threshold: if probThrArg is set (0.5 or whatever), tuning is off
    if( probThrArg.isSet() )
        modelType = ModelType( ModelType::logistic, ModelType::ZO, 
            probThrArg.getValue(), 
            stdArg.getValue(),
            convergeArg.isSet() ? convergeArg.getValue() : convergeDefault,
            iterLimitArg.isSet() ? iterLimitArg.getValue() : iterDefault, 
            highAccuracyArg.getValue() );
    else
        modelType = ModelType( ModelType::logistic, ModelType::ZO, 
            ModelType::thrtune( thrArg.getValue() ), 
            stdArg.getValue(),
            convergeArg.isSet() ? convergeArg.getValue() : convergeDefault,
            iterLimitArg.isSet() ? iterLimitArg.getValue() : iterDefault, 
            highAccuracyArg.getValue() );
    Log(2)<<endl<<modelType;

    designParameter = DesignParameter(designPlain);    //Log(2)<<endl<<designParameter;
    //no extScoreParam.get();    Log(2)<<endl<<extScoreParam;
    //no bagOfWordsFile = ParamGetString("bagOfWordsFile",""); Log(2)<<"\nbagOfWordsFile: "<<( bagOfWordsFile.size()>0 ? bagOfWordsFile.c_str() : "<none>" );
    //no priorTermsFile = ParamGetString("priorTermsFile","");
    //no priorMean = ParamGetDouble("priorMean",1);   Log(2)<<"\npriorTermsFile: "<<( priorTermsFile.size()>0 ? priorTermsFile.c_str() : "<none>" )     <<"\tpriorMean="<<priorMean;
 
    trainPlainFile = datafileArg.getValue();
#ifdef ROWID
    rowIdMode =  rowIdArg.getValue();
    Log(2)<<"\nRow Id mode: "<<( rowIdMode ? "On" : "Off");
#endif //ROWID
    Log(2)<<"\nData file for Training: "<<( readFromStdin(trainPlainFile.c_str()) ? "stdin" : trainPlainFile);
    modelWriteFileName = modelfileArg.getValue();  Log(2)<<"\nWrite Model file: "<<modelWriteFileName;

    resFile = ResultsFile(false,rowIdMode,true,resfileArg.getValue());
    Log(2)<<endl<<"Results file: "<<( resFile.FName().size()>0 ? resFile.FName() : " no");

	} catch (ArgException e)  {
        cerr << "***Command line error: " << e.error() << " for arg " << e.argId() << endl;
        return 1;
    }catch( std::exception& e){
        Log(0)<<std::endl<<"***Exception: "<<e.what();
        cerr<<std::endl<<"***Exception: "<<e.what();
        return 1;
    }catch(...){
        Log(0)<<std::endl<<"***Unrecognized exception";
        cerr<<std::endl<<"***Unrecognized exception";
        return 1;
    }

    //finally - do what you always wanted
    return runBatch();
}
#elif defined(TEST_ONLY)
int main(int argc, char** argv)
{
	try {  

	// Define the command line object.
	CmdLine cmd(argv[0], "Bayesian Binary Regression - Classification", VERSION );

#ifdef ROWID
    SwitchArg rowIdArg("R","rowid","Row Id mode",false); cmd.add( rowIdArg );
#endif //ROWID
    ValueArg <int>  logArg("l","log","Log verbosity level",false,0, "[0..2]"); cmd.add( logArg );

    ValueArg<double> probThrArg("","probthres","Probability threshold",false,0.5,"0<=p<=1"); cmd.add(probThrArg);
    ValueArg<string> resfileArg("r","resultfile","Results file",false,"","resultfile"); cmd.add( resfileArg );

    UnlabeledValueArg<string>  datafileArg("datafile","Data file; '-' for stdin","","datafile"); cmd.add( datafileArg );
    UnlabeledValueArg<string>  modelfileArg("modelfile","Model file","","modelfile"); cmd.add( modelfileArg );
    //UnlabeledValueArg<string>  resfileArg("resultfile","Result file","","resfile"); cmd.add( resfileArg );

	// Parse the args.
	cmd.parse( argc, argv );

    //----set parameters---
    Log.setLevel( logArg.getValue() +5 );    //Log(2)<<"\nLog Level: "<<Log.level();
    Log(0)<<endl<<"Bayesian Binary Regression - Classification. \tVer. "<<VERSION;

    multiLabel = 0;
    eTrainTestSplit = (ETrainTestSplit)fileSplit;
 
#ifdef ROWID
    rowIdMode =  rowIdArg.getValue();
    Log(2)<<"\nRow Id mode: "<<( rowIdMode ? "On" : "Off");
#endif //ROWID
    testPlainFile = datafileArg.getValue();   
    Log(2)<<"\nData file for Testing: "<<( readFromStdin(testPlainFile.c_str()) ? "stdin" : testPlainFile);
    modelReadFileName = modelfileArg.getValue();  Log(2)<<"\nRead Model file: "<<modelReadFileName;

    if( probThrArg.isSet() )
        probTestThreshold = probThrArg.getValue();

    resFile = ResultsFile(false,rowIdMode,true,resfileArg.getValue());
    Log(2)<<endl<<"Results file: "<<( resFile.FName().size()>0 ? resFile.FName() : " no");

	} catch (ArgException e)  {
        cerr << "***Command line error: " << e.error() << " for arg " << e.argId() << endl;
        return 1;
    }catch( std::exception& e){
        Log(0)<<std::endl<<"***Exception: "<<e.what();
        cerr<<std::endl<<"***Exception: "<<e.what();
        return 1;
    }catch(...){
        Log(0)<<std::endl<<"***Unrecognized exception";
        cerr<<std::endl<<"***Unrecognized exception";
        return 1;
    }

    //finally - do want you always wanted to
    return runBatch();
}
#endif // TEST_ONLY

#else //defined(TRAIN_ONLY) || defined(TEST_ONLY) //<==public version   home version==>
#ifdef USE_LEMUR

void GetAppParam()
{
    try{

    int logLevel = ParamGetInt("logLevel",5);
    Log.setLevel(logLevel);
	Log(1)<<"Bayesian Binary Regression ver."<<VERSION;
    Log(2)<<std::endl<<"Log Level: "<<Log.level();

    multiLabel =( 1==ParamGetInt("multiLabel",1) );

    eTrainTestSplit = (ETrainTestSplit)ParamGetInt("trainTestSplit",fileSplit);
    Log(2)<<std::endl<<"trainTestSplit: "<<( eTrainTestSplit==randomSplit ? "random" 
        : eTrainTestSplit==fileSplit ? "from files" : "undefined");

    oracleNRelDefault = (1==ParamGetInt("oracleNRelDefault",1));
    Log(2)<<std::endl<<"Oracle Non-Relevant by Default: "<<( oracleNRelDefault ? "Yes" : "No" );

    featSelectMetaParam.get();    Log(2)<<featSelectMetaParam;
    tfidfParameter.get();    Log(2)<<std::endl<<tfidfParameter;
    modelType.get();    Log(2)<<std::endl<<modelType;
    //bayesParameter.get();    Log(2)<<std::endl<<bayesParameter;
    priorTermsByTopic.get();    Log(2)<<std::endl<<priorTermsByTopic;
    hyperParamPlan.get();    Log(2)<<std::endl<<hyperParamPlan;
    squeezer.get();    Log(2)<<std::endl<<squeezer;
    designParameter.get();    Log(2)<<std::endl<<designParameter;
    extScoreParam.get();    Log(2)<<std::endl<<extScoreParam;

    bagOfWordsFile = ParamGetString("bagOfWordsFile","");
    Log(2)<<std::endl<<"bagOfWordsFile: "<<( bagOfWordsFile.size()>0 ? bagOfWordsFile.c_str() : "<none>" );

    priorTermsFile = ParamGetString("priorTermsFile","");
    priorMean = ParamGetDouble("priorMean",1);
    //Log(8)<<std::endl<<"priorTermsFile: "<<( priorTermsFile.size()>0 ? priorTermsFile.c_str() : "<none>" )
        //<<"\tpriorMean="<<priorMean;

    trainDocsFile = ParamGetString("trainDocsFile","");
    Log(2)<<std::endl<<"trainDocsFile: "<<( trainDocsFile.size()>0 ? trainDocsFile.c_str() : "<none>" );
    testDocsFile = ParamGetString("testDocsFile","");
    Log(2)<<std::endl<<"testDocsFile: "<<( testDocsFile.size()>0 ? testDocsFile.c_str() : "<none>" )<<endl;
    idfDocsFile = ParamGetString("docsForIDF","");
    if( idfDocsFile.size()>0 )  Log(2)<<std::endl<<"Docs for IDF calculation: "<<idfDocsFile<<endl;

    databaseIndex = ParamGetString("index",""); Log(2)<<endl<<"databaseIndex "<<databaseIndex;
    trainIndex = ParamGetString("trainIndex",""); Log(2)<<endl<<"trainIndex "<<trainIndex;
    trainQrelFile = ParamGetString("trainQrelFile",""); Log(2)<<endl<<"trainQrelFile "<<trainQrelFile;
    testQrelFile = ParamGetString("testQrelFile",""); Log(2)<<endl<<"testQrelFile "<<testQrelFile;
    textQuerySet = ParamGetString("textQuery",""); Log(2)<<endl<<"textQuerySet "<<textQuerySet;

    trainPlainFile = ParamGetString("trainPlainFile","");
    testPlainFile = ParamGetString("testPlainFile","");
    if( testPlainFile.size()>0 )
        Log(2)<<"\nData file for Training: "<<trainPlainFile
                <<"\nData file for  Testing: "<<testPlainFile;

    modelReadFileName = ParamGetString("modelReadFile","");
    Log(2)<<"\nRead Model file: "<<modelReadFileName;
    modelWriteFileName = ParamGetString("modelWriteFile","");
    Log(2)<<"\nWrite Model file: "<<modelWriteFileName;

    probTestThreshold = ParamGetDouble("probTestThreshold",-1);
    if(probTestThreshold!=-1) //attempt to use parameter
        if( 0<=probTestThreshold && probTestThreshold<=1 )
            Log(2)<<"\nProbability threshold for testing: "<<probTestThreshold;
        else
            throw runtime_error("Illegal Probability threshold for testing");

    string resFName = ParamGetString("resultFile",""); Log(2)<<endl<<"resultFile "<<resFName;  
    string sResultFormat = ParamGetString("resultFormat","prob"); Log(2)<<endl<<"resultFormat "<<sResultFormat;
    bool resProb;
    if( 0==stricmp(sResultFormat.c_str(),"prob") ) resProb=true;
    else if( 0==stricmp(sResultFormat.c_str(),"score") ) resProb=false;
    else throw runtime_error("Wrong result format");
    resFile = ResultsFile(true,true,resProb,resFName);
    Log(2)<<endl;

    }catch( std::exception& e){
        Log(0)<<std::endl<<"***Exception: "<<e.what();
    }catch (Exception &ex) { //Lemur
        Log(0)<<std::endl<<"***Lemur exception: ";
        ex.writeMessage( Log(0) );
    }catch(...){
        Log(0)<<std::endl<<"***Unrecognized exception";
    }
}

int AppMain(int argc, char *argv[])  {
    //copy param file contents to output
    Log(1)<<"\n\n***** parameter file begin *****\n";
    ifstream i( argv[1] );
    string buf;
    while( getline( i, buf ) )
        Log(1)<<buf<<endl;
    i.close();
    Log(1)<<"***** parameter file end *****\n\n";

    //finally, do the job
    return runBatch();
}

#endif //USE_LEMUR
#endif //defined(TRAIN_ONLY) || defined(TRAIN_ONLY) //home version

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
