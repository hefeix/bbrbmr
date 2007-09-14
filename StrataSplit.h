#ifndef _STRATA_SPLIT_HEADER
#define _STRATA_SPLIT_HEADER

vector<int> PlanStratifiedSplit( class IRowSet & drs, unsigned nfolds );
vector< vector<unsigned> >  //returns arrays of fold ids, one array per stratum
StrataSplit(
             const vector<unsigned>& strataSizes, //sizes per strata; length defines #strata
             unsigned nfolds );
void ReportSplit( unsigned nfolds, const vector< vector<unsigned> >& split, ostream& o );

#endif //_STRATA_SPLIT_HEADER
