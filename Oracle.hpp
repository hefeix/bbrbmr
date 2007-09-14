/*
 * based on the original design by Vladimir Menkov
 */

#ifndef _ORACLE_HPP
#define _ORACLE_HPP

#include <map>
#ifdef _MSC_VER
 #include <hash_map>
#else
 #include <ext/hash_map>
 using namespace __gnu_cxx;
#endif
#include <stdexcept>

#ifdef USE_LEMUR
# include "common_headers.hpp"
# include "Index.hpp"
#endif //USE_LEMUR

using namespace std;

/// An Oracle is an interface to a file of experts' judgments (qrel)

/// Possible judgments to be returned by an Oracle
enum OracleLabel  {
  // Marked as "non-relevant" in the QREL file
  NONREL=0, 
  // Marked as "relevant" in the QREL file
  REL=1, 
  // Not present in Qrel file (or marked as "unjudged"), but we looked
  UNJUDGED=-1,
  // We never asked the Oracle
  UNLABELED=-2
};


/// Doc-ID to judgment map; we'll have one of these for each topic.
// The doc ID is w.r.t. the main index.
#ifdef _MSC_VER
    typedef hash_map<int, int> DJMap;
    typedef map<string, DJMap> QDJMap;
#else
    struct eqstr {
    bool operator()(const char* s1, const char* s2) const  {
        return strcmp(s1, s2) == 0;
    }
    };

    struct eqint {
    bool operator()(int s1, int s2) const  {
        return (s1 == s2);
    }
    };
    typedef hash_map<int, int, hash<int>, eqint> DJMap;
    typedef hash_map<const char*, DJMap, hash<const char*>, eqstr> QDJMap;
#endif

class Oracle {
public:
#ifdef USE_LEMUR
  Oracle ( char const * inputFile, Index &dbIndex, bool isTraining=false);
#endif //USE_LEMUR
  Oracle( char const * inputFile, map<string,int>& docIdByName, bool _isTraining);
  Oracle( char const * inputFile, class IDocIndex& docIdExtractor, bool _isTraining);
  ~Oracle()  {};

  OracleLabel ask(const char * q_id, int doc_id);

  // Same as ask(), but returns the answer as a printable string
  const char * askAsString(const char * q_id, int doc_id) {
    return label2string( ask(q_id, doc_id));
  }

  static const char * label2string(OracleLabel label) {
    return (label == NONREL) ? "Non": (label == REL) ? "Rel": "Unj";
  }

  // How many judgments of this kind does the oracle have for this query?
  int docCnt( const char * q_id,OracleLabel label );
  int docCnt( const char * q_id );

  vector<int> docList( const char * q_id, OracleLabel label);
  vector<int> docList( const char * q_id);


private:
  void Init( char const * inputFile, class IDocIndex& docIdExtractor );

  bool isTraining;
  char file[1024];
  ifstream *ifs;
  static const int bufsize = 2000;
  char buf[bufsize];

  /// Maps a key (query id) to a value (a hash map that maps doc id to judgment)
  QDJMap dataMap;

  /// Sums up the total number of entries for all queries stored in this Oracle
  int totalSize();   
  int docCnt(DJMap & m,OracleLabel label );

};

/*--- wrap Oracle when there is no Lemur Index ---*/
class IDocIndex{ public: 
virtual int lookup( const char* docname ) const =0;
//virtual const char* lookup( int docnum ) const =0;
};
#ifdef USE_LEMUR
class LemurDocIndex : public IDocIndex {
    Index &dbIndex;
public: 
    LemurDocIndex(Index &dbIndex_) : dbIndex(dbIndex_) {};
    int lookup( const char* docname ) const { return dbIndex.document(docname); }
};
#endif //USE_LEMUR
class PlainDocIndex : public IDocIndex {
    const map<string,int>& docIdByName;
public: 
    PlainDocIndex( const map<string,int>& docIdByName_ ) : docIdByName(docIdByName_) {};
    int lookup( const char* docname ) const {
        map<string,int>::const_iterator itr = docIdByName.find(docname);
        return itr==docIdByName.end() ? -1 : itr->second; }
};

class InstantDocIndex : public IDocIndex {
    mutable map<string,int> docIdByName;
    bool locked;
public: 
    InstantDocIndex() {
        locked = false; };
    int lookup( const char* docname_c ) const {
        /*cout<<"\noracle lookup "<<docname_c<<" in "<<docIdByName.size();dbg*/
        string docname(docname_c);
        map<string,int>::const_iterator itr = docIdByName.find(docname);
        if( itr==docIdByName.end() )
            if( locked ) return -1;
            else {
                int id = docIdByName.size()+1;
                docIdByName[docname] = id;
                return id;
            }
        else
            return itr->second;
    }
    void lock() { locked=true; }
};

class PlainOracle { //wraps Oracle, allows to ask by name
    Oracle *pqrel;
    InstantDocIndex docIndex;
public:
    PlainOracle( const char* qrelFile, bool isTraining )
        //: docIndex(), qrel( qrelFile, docIndex, isTraining )
    {
        pqrel = new Oracle( qrelFile, docIndex, isTraining );
        docIndex.lock();    };
    ~PlainOracle() { delete pqrel; } 
    OracleLabel ask(const char * topic, const char * docName) {
        int ind = docIndex.lookup(docName);
        if( ind<0 )  return UNJUDGED;
        else return pqrel->ask( topic, ind );
    }
};

class TwoWayDocIndex : public IDocIndex {
    mutable map<string,int> docIdByName;
    mutable map<int,string> docNameById;
    bool locked;
public: 
    TwoWayDocIndex() {
        locked = false; };
    TwoWayDocIndex( map<string,int>& docIdByName_ ) : docIdByName(docIdByName_) {
        locked = false; };
    void add( string docname, int docid ) {
        if( locked ) throw logic_error("InstantDocIndex is locked");
        else{ 
            docIdByName[docname] = docid;   
            docNameById[docid] = docname;   
        }
    }
    void clear() { docIdByName.clear(); }
    int lookup( const char* docname_c ) const {
        string docname(docname_c);
        map<string,int>::const_iterator itr = docIdByName.find(docname);
        if( itr==docIdByName.end() )
            if( locked ) return -1;
            else {
                int id = docIdByName.size()+1;
                docIdByName[docname] = id;
                docNameById[id] = docname;
                return id;
            }
        else
            return itr->second;
    }
    const char* lookup( int id ) const { 
        map<int,string>::const_iterator itr = docNameById.find(id);
        if( itr==docNameById.end() ) throw runtime_error(string("InstantDocIndex: id not found: "));
        else return itr->second.c_str();
    }
    void lock() { locked=true; }
};

#endif

