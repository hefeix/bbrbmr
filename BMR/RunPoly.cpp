#define VERSION "2.02"
/*
  0.21    Oct 04, 04  
                        more stable numerics in ZO
                        '-y' option deprecated
                        precision/recall/f1 are 100% when denom==0 according to Dave Lewis
                        fixed bug with unseen classes in Classify module
  0.21a   Oct           in process of adding quasi-Newton optimizer
  0.22    Oct 18, 04    fixed bug with empty vectors
  0.23    Oct 21, 04    numeric improvement, see PolyZO.cpp:ZOLR()
  0.23a   Oct 21, 04    in process of adding quasi-Newton optimizer: normal prior
  0.24    Nov 04, 04
                        new random split for cv: fixed sizes; class-stratified; #folds <= #cases
                        numeric improvement, see PolyZO.cpp:OneCoordStep(...)
  0.25    Dec 02, 04    "=", not "-" argument indicates stdin for data 
  1.0     Dec 16, 04
  1.01    Feb 18-23 05  bugs in StrataSplit: 1) rand()/(randmax+1)   2) class size =0
                        error bar rule for hyperparameter cv (option --errbar)
                        "-" argument indicates stdin for data, also allowed "="
  1.04    Mar 04, 05    new options: high accuracy mode; iter. limit 
    TODO: not really supported
  1.04a   May 05, 05    BayesParameter/HyperparamPlan classes refactored
  1.05    Jun 20, 05    one std error rule: fixed bug with stderr denom calc //PolyZO.cpp
  1.53    Sep 02, 05    fixed bug with data from stdin //DataFactory.cpp
  1.60    Oct 24, 05    sparse beta in model file
                        Individual priors
  2.00                  fixed log.prior with infinite prior var
  2.01    Jan 31, 06    fixed bug: Gaussian penalty should be 1/(2*var), not 1/var //PolyZO.cpp
  2.02    Oct 08, 06    fixed bug: -R not working without -z //PolyZO.cpp, poly.h
  2.03    Mar 16, 07    fixed bug with "infinite penalty", detected by Shenzhi Li //PolyZO.cpp
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <stdexcept>
#define  _USE_MATH_DEFINES
#include <math.h>

using namespace std;

#include "logging.h"
#include "FeatSelectParamManager.h"
#include "TFIDFParamManager.h"
#include "BayesParamManager.h"
#include "ModelTypeParam.h"
#include "dataPoly.h"
#include "poly.h"
#include "PriorTerms.h"
#include "Design.h"
#include "ExtScore.h"
#include "DataFactory.h"
#include "ModelFile.h"

logging Log(5);
static FeatSelectByTopic featSelectMetaParam;
static TFIDFParameter tfidfParameter;
static PriorTermsByTopic priorTermsByTopic; 
static HyperParamPlan hyperParamPlan;
static ModelType modelType;
static DesignParameter designParameter;
static ExtScoreParam extScoreParam;

enum ETrainTestSplit { randomSplit=1, fileSplit=2, undef } 
    eTrainTestSplit;

static bool oracleNRelDefault; //no info in oracle means doc is non-relevant

static std::string databaseIndex;
static std::string trainIndex;
static std::string trainQrelFile;
static std::string testQrelFile;
static std::string textQuerySet;
static std::string trainDocsFile;
static std::string testDocsFile;

static std::string resultFile;  
static ResultFormat resultFormat;

static std::string bagOfWordsFile;
static std::string priorTermsFile;
static double priorMean;

static std::string  modelWriteFileName;
static std::string  modelReadFileName;

static std::string  trainPlainFile;
static std::string  testPlainFile;

int runBatch()
{
    try{
        if( !oracleNRelDefault )
            throw runtime_error("Should be <Oracle non-relevant by default> - that's the only option currently supported");

        //output
        ofstream result(resultFile.c_str());

        //input model file
        ReadModel readModel( modelReadFileName );
        if( readModel.Active() ) {
            tfidfParameter = readModel.tfidfParam();
            //c = readModel.NClasses();
        }

        DataFactory* dataFactory=0;
        if( readModel.Active() ) //test only
            dataFactory = new PlainFileYDataFactory( 
                testPlainFile.c_str(),
                readModel.Classes(),
                tfidfParameter.tfMethod(), tfidfParameter.idfMethod(), tfidfParameter.cosineNormalize(),
                readModel.Feats(),
                readModel.Idf()
            );
        else //train and test
            dataFactory = new PlainFileYDataFactory( 
                trainPlainFile.c_str(),
                testPlainFile.c_str(),
                //c,
                tfidfParameter.tfMethod(), tfidfParameter.idfMethod(), tfidfParameter.cosineNormalize()
            );

        //model file
        WriteModel* writeModel=0;
        ofstream* svmModelFile=0;
        writeModel = new WriteModel( modelWriteFileName, *dataFactory );

            string topic = "<class>";
            FeatSelectParameter featSelectParameter = featSelectMetaParam.GetForTopic( topic.c_str() );

                if( modelType.Optimizer()==modelType.ZO
#ifdef QUASI_NEWTON
                    || modelType.Optimizer()==modelType.QN_smooth
                    || modelType.Optimizer()==modelType.QN_2coord 
#endif
                ) {
                    LRModel model;
                    RowSetMem* TrainRowSet =0;
                    IRowSet* TestRowSet =0;

#if !defined(TEST_ONLY)
                    if( ! readModel.Active() ) { //train
                        TrainRowSet = dataFactory->NewTrainRowSet(
                                        topic.c_str(),
                                        featSelectParameter,
                                        priorTermsByTopic.GetForTopic(topic),
                                        extScoreParam );

                        model.Train( topic.c_str(),
                            *TrainRowSet,
                            hyperParamPlan,
                            priorTermsByTopic,
                            designParameter,
                            modelType,
                            *writeModel,
                            result, resultFormat);
                    }
                    Log(3)<<"\nTime "<<Log.time();
#endif //!TEST_ONLY
#if !defined(TRAIN_ONLY)
                    if( readModel.Active() ) {
                        TestRowSet = dataFactory->NewTestRowSet( topic.c_str(), readModel.TopicFeats(), extScoreParam  );
                        model.Restore( readModel, *TestRowSet );
                    }
                    else
                        TestRowSet = dataFactory->NewTestRowSet( topic.c_str(), extScoreParam  );

                    model.Test( *TestRowSet, result, resultFormat );
#endif //!TRAIN_ONLY
                    delete TrainRowSet; //don't delete earlier: 'IDesign' depends on it
                    delete TestRowSet;
                }


        delete writeModel;
        delete dataFactory;
        Log(0)<<endl;

    }catch( std::exception& e){
        Log(0)<<std::endl<<"***Exception: "<<e.what();
        cerr<<std::endl<<"***Exception: "<<e.what();
        return 1;
    }catch(...){
        Log(0)<<std::endl<<"***Unrecognized exception";
        cerr<<std::endl<<"***Unrecognized exception";
        return 1;
    }
    return 0;
}

#include <tclap/CmdLine.h>
using namespace TCLAP;

#if defined(TRAIN_ONLY)

int main(int argc, char** argv)
{
	try {  

	// Define the command line object.
	CmdLine cmd( "Bayesian Multinomial Logistic Regression - Training", ' ', VERSION );

    ValueArg <int>  nclassesArg("y","yvalues","OBSOLETE: Number of classes",false,2,"[2..]");
	cmd.add( nclassesArg );

    ValueArg <int>  logArg("l","log","Log verbosity level",false,0, "[0..2]");
	cmd.add( logArg );

    //TODO SwitchArg highAccuracyArg("","accurate","High accuracy mode",false); cmd.add( highAccuracyArg );
    ValueArg <unsigned> iterLimitArg("","iter","Max number of iterations",false,iterDefault,"integer"); cmd.add( iterLimitArg );
    ValueArg <double>  convergeArg("e","eps","Convergence threshold",false,convergeDefault,"float");	cmd.add( convergeArg );

    SwitchArg cosnormArg("c","cosnorm","Cosine normalization",false); cmd.add( cosnormArg );
    SwitchArg stdArg("s","standardize","Standardize variables",false); cmd.add( stdArg );
    //ValueArg <int>  cosnormArg("c","cosnorm","Cosine normalization",false,0,"[0,1]"); cmd.add( cosnormArg );
    //ValueArg <int>  stdArg("s","standardize","Standardize variables",false,0,"[0,1]"); cmd.add( stdArg );

    ValueArg<string> indPriorsArg("I","indPriorsFile","Individual Priors file",false,"","indPriorsFile"); cmd.add(indPriorsArg);

    SwitchArg resScoreArg("","rscore","Scores, not probabilities, in the result file", false);
	cmd.add( resScoreArg );
    ValueArg<string> resfileArg("r","resultfile","Results file",false,"","resultfile");
	cmd.add( resfileArg );

#ifdef QUASI_NEWTON
    ValueArg <int> optArg("o","opt",
        "Optimizer: \n\t 1-ZO \n\t 2-quasi-Newton, smoothed penalty \n\t 3-quasi-Newton, double coordinate",
        false,1,"[1,3]");
	cmd.add( optArg );
#endif

    SwitchArg allZeroArg("z","zerovars","Exclude all-zero per class", false);
	cmd.add( allZeroArg );
    SwitchArg refClassArg("R","refClass","Use Reference Class", false);
	cmd.add( refClassArg );

    //back-compatibility only -->
    ValueArg <string>  searchArg("S","search",
        "DEPRECATED Search for hyperparameter value",false,"","list of floats, comma-separated, no spaces"); cmd.add( searchArg );
    ValueArg <double>  hypArg("H","hyperparameter",
        "DEPRECATED Hyperparameter, depends on the type of prior", false,0,"float"); cmd.add( hypArg );
    //<--back-compatibility only

    SwitchArg errBarArg("","errbar","Error bar rule for cross-validation", false); cmd.add( errBarArg );
    ValueArg <string>  cvArg("C","cv","Cross-validation",false,"10,10","#folds[,#runs]"); cmd.add( cvArg );
    ValueArg <string>  priorVarArg("V","variance",
        "Prior variance values; if more than one, cross-validation will be used",false,"",
        "number[,number]*"); cmd.add( priorVarArg );

    ValueArg <int>  priorArg("p","prior","Type of prior, 1-Laplace 2-Gaussian",false,2,"[1,2]"); cmd.add( priorArg );

    UnlabeledValueArg<string>  datafileArg("datafile","Data file; '-' signifies standard input","","data file");
	cmd.add( datafileArg );
    UnlabeledValueArg<string>  modelfileArg("modelfile","Model file","","model file");
	cmd.add( modelfileArg );

	// Parse the args.
	cmd.parse( argc, argv );

    //----set parameters---
    Log.setLevel( logArg.getValue() +5 );
    Log(0)<<endl<<"Bayesian Multinomial Logistic Regression - Training \tVer. "<<VERSION;
    Log(2)<<"\nCommand line: ";
    for( int i=0; i<argc; i++ )
        Log(2)<<" "<<argv[i];
    Log(2)<<"\nLog Level: "<<Log.level()-5;
    eTrainTestSplit = (ETrainTestSplit)fileSplit;
    oracleNRelDefault = 1;

    if( nclassesArg.isSet() )
        Log(2)<<"\n\"-y\" argument is deprecated, has no effect";
    //c = nclassesArg.getValue();   //Log(2)<<std::endl<<"Classes: "<<c;

    // Bayes parameters
    enum PriorType prior = PriorType(priorArg.getValue());
    if( prior!=1 && prior!=2 )
        throw runtime_error("Illegal prior type; should be 1-Laplace or 2-Gaussian");
    int skew = 0; //not supported by BMR
    if( priorVarArg.isSet() )  // new mode
        hyperParamPlan = HyperParamPlan( prior, skew, priorVarArg.getValue(), cvArg.getValue(),
                    HyperParamPlan::AsVar, errBarArg.getValue() );
    else if( searchArg.isSet() ) //back-compatibility
        hyperParamPlan = HyperParamPlan( prior, skew, searchArg.getValue(), cvArg.getValue(),
                    HyperParamPlan::Native, errBarArg.getValue() );
    else if( hypArg.isSet() ) //fixed hyperpar - back-compatibility
        hyperParamPlan = HyperParamPlan( prior, hypArg.getValue(), skew );
    else //auto-select hyperpar
            hyperParamPlan = HyperParamPlan( prior, skew );
    Log(2)<<endl<<hyperParamPlan;  //<<endl<<bayesParameter

    if( indPriorsArg.isSet() ) {
        priorTermsByTopic = PriorTermsByTopic( indPriorsArg.getValue(), indPriorsModeRel );
        Log(2)<<endl<<priorTermsByTopic;
    }

    featSelectMetaParam = FeatSelectByTopic(  //not supported by BMR
        FeatSelectParameter::corr, //whatever -not working now    //UtilityFunc(featSelectUtilArg.getValue()), 
        0 ); //featSelectNumArg.getValue() );
    //Log(2)<<endl<<featSelectMetaParam;

    tfidfParameter = TFIDFParameter( RAWTF, NOIDF, cosnormArg.getValue() );
    Log(2)<<"\nCosine normalization: "<<( cosnormArg.getValue() ? "Yes" : "No" );
    ModelType::optimizer opt;
#ifdef QUASI_NEWTON
    opt = optArg.getValue()==3 ? ModelType::QN_2coord : optArg.getValue()==2 ? ModelType::QN_smooth : ModelType::ZO;
#else
    opt = ModelType::ZO;
#endif
    modelType = ModelType( ModelType::logistic,
        opt,
        ModelType::thrNo, //thrtune( thrArg.getValue() ), 
        stdArg.getValue(),
        convergeArg.isSet() ? convergeArg.getValue() : convergeDefault,
        iterLimitArg.isSet() ? iterLimitArg.getValue() : iterDefault, 
        false, //TODO highAccuracyArg.getValue(),
        refClassArg.getValue(),
        allZeroArg.getValue()
        );
    Log(2)<<endl<<modelType;

    designParameter = DesignParameter(designPlain);    //Log(2)<<endl<<designParameter;
 
    trainPlainFile = datafileArg.getValue();
    Log(2)<<"\nData file for Training: "<<( readFromStdin(trainPlainFile.c_str()) ? "stdin" : trainPlainFile);
    modelWriteFileName = modelfileArg.getValue();  Log(2)<<"\nWrite Model file: "<<modelWriteFileName;

    resultFile = resfileArg.getValue();
    if( resfileArg.isSet() ) {
        resultFormat = resScoreArg.getValue() ? resScore : resProb;
        Log(2)<<"\nResult file: "<<resultFile
            <<"\t Format: "<<( resultFormat==resScore ? "scores" : "probabilities" );
    }

	} catch (ArgException e)  {
        cerr << "***Command line error: " << e.error() << " for arg " << e.argId() << endl;
    }catch( std::exception& e){
        Log(0)<<std::endl<<"***Exception: "<<e.what();
    }catch(...){
        Log(0)<<std::endl<<"***Unrecognized exception";
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
	CmdLine cmd("Bayesian Multinomial Logistic Regression - Classification", ' ', VERSION );

    ValueArg <int>  logArg("l","log","Log verbosity level",false,0, "[0..2]");
	cmd.add( logArg );

    //TODO ValueArg<string> resfileArg("r","resultfile","Results file",false,"","resultfile");
	//cmd.add( resfileArg );
    SwitchArg resScoreArg("","rscore","Scores, not probabilities, in the result file", false);
	cmd.add( resScoreArg );
    ValueArg<string> resfileArg("r","resultfile","Results file",false,"","resultfile"); cmd.add( resfileArg );

    UnlabeledValueArg<string>  datafileArg("datafile","Data file; '-' signifies standard input","","datafile");
	cmd.add( datafileArg );
    UnlabeledValueArg<string>  modelfileArg("modelfile","Model file","","modelfile");
	cmd.add( modelfileArg );
    //UnlabeledValueArg<string>  resfileArg("resultfile","Result file","","resfile");	cmd.add( resfileArg );

	// Parse the args.
	cmd.parse( argc, argv );

    //----set parameters---
    Log.setLevel( logArg.getValue() +5 );
    Log(0)<<endl<<"Bayesian Multinomial Logistic Regression - Classification \tVer. "<<VERSION;
    Log(2)<<"\nCommand line: ";
    for( int i=0; i<argc; i++ )
        Log(2)<<" "<<argv[i];
    Log(2)<<"\nLog Level: "<<Log.level()-5;

    eTrainTestSplit = (ETrainTestSplit)fileSplit;
    oracleNRelDefault = 1;
 
    testPlainFile = datafileArg.getValue();
    Log(2)<<"\nData file for Testing: "<<( readFromStdin(testPlainFile.c_str()) ? "stdin" : testPlainFile);
    modelReadFileName = modelfileArg.getValue();  Log(2)<<"\nRead Model file: "<<modelReadFileName;

    resultFile = resfileArg.getValue();
    if( resfileArg.isSet() ) {
        resultFormat = resScoreArg.getValue() ? resScore : resProb;
        Log(2)<<"\nResult file: "<<resultFile
            <<"\t Format: "<<( resultFormat==resScore ? "scores" : "probabilities" );
    }

	} catch (ArgException e)  {
        cerr << "***Command line error: " << e.error() << " for arg " << e.argId() << endl;
        return 1;
    }catch( std::exception& e){
        Log(0)<<std::endl<<"***Exception: "<<e.what();
    }catch(...){
        Log(0)<<std::endl<<"***Unrecognized exception";
        return 1;
    }

    //finally - do want you always wanted to
    return runBatch();
}
#endif // TEST_ONLY

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
