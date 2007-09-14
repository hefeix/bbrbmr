/*
 * Originally designed by Vladimir Menkov
 * modified by Alex Genkin
 */

#include <stdio.h>
#include <assert.h>

#include <sstream>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "Oracle.hpp"
#include "logging.h"

/*!  Creates an Oracle that will obtain judgments from a specified QREL
  file. Doc IDs are translated from the string format used externally
  into integer doc IDs used in the specified LEMUR index.

  Assumes that each line in the input file has the following format:

  Query-id  Doc-id Judgment
  e.g.
  1 1410 1
*/

Oracle::Oracle( char const * inputFile, IDocIndex& docIdExtractor, bool _isTraining):
  isTraining( _isTraining)
{
    Init( inputFile, docIdExtractor );
}
void Oracle::Init( char const * inputFile, IDocIndex& docIdExtractor )
{
  strcpy(file, inputFile);
  cout << "\nReading oracle file `" << file << "'\n";
  ifs = new ifstream(inputFile, ios::in);
  if (ifs->fail() ) {
    string msg("can't open the QREL data file `");
    msg += file;
    msg +=  "'";
    throw runtime_error(string("Oracle ")+msg);
  }


  int lineCnt = 0, nonEmptyLineCnt=0;
  int invalidDocCnt=0;
  char qs[bufsize], ds[bufsize];
  while(!ifs->eof()) {
    ifs->getline( buf, bufsize );

    lineCnt ++;
    // skip empty lines
    if (strspn(buf, " ") == strlen(buf)) continue;
    nonEmptyLineCnt++;
    
    int judg;
    int n = sscanf( buf, "%s %s %d", qs, ds, &judg);
    //Log(12)<<endl<<nonEmptyLineCnt<<" "<<qs<<" "<<ds<<" "<<judg;

    if (n != 3) {
      char msgbuf[255 + bufsize ];
      sprintf(msgbuf, "Cannot parse line %d in qrel file `%s': [%s]", 
	      lineCnt, file, buf);
      throw runtime_error(string("Oracle ")+msgbuf);
    }

    // convert to LEMUR numerical doc index
    int di = docIdExtractor.lookup(ds);    //dbIndex.document(ds);

    if (di < 1) {
      // The qrel file refers to a document that is not participating in
      // this experiment
      invalidDocCnt++;
      continue;
    }

    // validate judgment 
    switch(judg) {
    case NONREL:
    case REL:
    case UNJUDGED:
      break; // OK
    default:
        throw runtime_error(string("Oracle: Invalid judgment value found in the qrel file"));
    }

    // store the data
    // FIXME a memory leak here, with those strdup() calls!
#ifdef _MSC_VER
    dataMap[string(qs)][di] = judg;
#else
    dataMap[strdup(qs)][di] = judg;
#endif
    //    cout <<     "dataMap["<<qs<<"]["<<ds<<"] = "<<judg <<"; size="<< dataMap.size()<< "\n";
    // cout << "The total of " << totalSize() << " judgments\n";

  }
  cout << "\nOracle(" << file << "): " << nonEmptyLineCnt << " lines read\n";
  cout << invalidDocCnt << " lines has been ignored, as they refer to unavailable docs\n";
  cout << "Map has data for " << dataMap.size() << " queries\n";
  cout << "The total of " << totalSize() << " judgments\n";
}


OracleLabel Oracle::ask(const char * q_id, int doc_id) {
  QDJMap::iterator p = dataMap.find( q_id);
  if (p == dataMap.end()) return UNJUDGED;
  DJMap &m = p->second;  
  DJMap::iterator it = m.find( doc_id);
  if (it == m.end()) return UNJUDGED;
  int ju = it->second;
  return (OracleLabel)ju;
}

// Does this oracle have any judgments for this query at all?
int Oracle::docCnt( const char * q_id ) {
  QDJMap::iterator p = dataMap.find( q_id);
  return (p == dataMap.end()) ? 0 : p->second.size();
}

int Oracle::docCnt( const char * q_id,OracleLabel label ) {
  QDJMap::iterator p = dataMap.find( q_id);
  return (p == dataMap.end()) ? 0 : docCnt( p->second, label);
}

// Does this oracle have any judgments for this query at all?
int Oracle::docCnt(DJMap & m,OracleLabel label ) {
  int cnt=0;
  for(  DJMap::iterator it = m.begin(); it != m.end(); it++) {
    if ((*it).second == label)  cnt++;
  }
  return cnt;
}


/// Returns the list of documents with the specifed label (REL or
// NON) for the specified query.
// This method can only be used with a training set, of course
vector<int> Oracle::docList( const char * q_id, OracleLabel label) {
  if (!isTraining) 
      throw logic_error(string("Oracle::relList: Prohibited to look ahead unless this is TRAIN!"));
  QDJMap::iterator p = dataMap.find( q_id);
  if (p== dataMap.end()) {
    cout <<"Warning in Oracle::docList("<<label<<"): No data found for query "
	 << q_id	 <<endl;
    vector<int> x(0);
    return x;
  }
  DJMap &m = p->second;  
  int cnt= docCnt(m, label);
  int k=0;
  vector<int> x(cnt);
  for(  DJMap::iterator it = m.begin(); it != m.end(); it++) {
    if ((*it).second == label) {
      int di = (*it).first;
      x[k++] = di;
    }
  }
  assert(k==cnt);
  return x;  
}


/// Returns the list of documents with the specifed label (REL or
// NON) for the specified query.
// This method can only be used with a training set, of course
vector<int> Oracle::docList( const char * q_id) {
  if (!isTraining)
      throw logic_error(string("Oracle::relList: Prohibited to look ahead unless this is TRAIN!"));
  QDJMap::iterator p = dataMap.find( q_id);
  if (p== dataMap.end()) {
    cout << "Warning in Oracle::docList: no judgments found for query "<< q_id
	 <<endl;
    vector<int> x(0);
    return x;
  }
  DJMap &m = p->second;  
  int cnt= m.size();
  int k=0;
  vector<int> x(cnt);
  for(  DJMap::iterator it = m.begin(); it != m.end(); it++) {
    int di = (*it).first;
    x[k++] = di;
  }
  assert(k==cnt);
  return x;  
}



/// Sums up the total number of entries for all queries stored in this Oracle
int Oracle::totalSize() {

  int sum = 0;
  QDJMap::iterator it;

  for(  it = dataMap.begin(); it != dataMap.end(); it++) {
    //const char * q = (*it).first;
    DJMap &m = (*it).second;
    //    cout << "[Q="<<q<<"] -> "<< m.size()<<" val\n";
    sum += m.size();
  }
  return sum;
  
}
