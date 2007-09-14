// 2.07a    Jun 01, 05  DisplayModel: fixed bug with duplicate beta values; don't use map! //released as 2.04a

#ifndef _DATA_DESIGN_HEADER_
#define _DATA_DESIGN_HEADER_

#include <iomanip>
#include "data.h"
#ifdef USE_LEMUR
# include "Param.hpp"
#endif //USE_LEMUR

enum DesignType { designPlain=0, designInteractions=1, designUndef };

class DesignParameter {
public:
    enum DesignType DesignType() const { return m_design; }
#ifdef USE_LEMUR
    void get()
    {
        m_design = (enum DesignType)ParamGetInt("DesignType", designPlain);
        if( designUndef==m_design )
            throw runtime_error("Undefined Design type");
    }
#endif //USE_LEMUR
    //ctor
    DesignParameter( enum DesignType d =designPlain ) : m_design(d) {}
private:
    enum DesignType m_design;
};

inline std::ostream& operator<<( std::ostream& o, const DesignParameter& dp ) {
    o << "DesignType: "<<( 
        dp.DesignType()==designPlain ? "Plain" 
        : dp.DesignType()==designInteractions ? "Interactions" 
        : "<undefined>" );
    return o;
}

class IDesign { //: public INameResolver {
public:
    virtual unsigned dim() const =0;
    virtual string colName( unsigned c ) const=0;
    virtual int colId( unsigned c ) const=0;
    virtual unsigned colNumById( unsigned c ) const=0;
    virtual unsigned interceptColNum() const=0;

    //EM only: virtual Matrix MakeDense( const IDenseData & data ) const =0;
    virtual class DesignRowSet MakeRowSet( IRowSet& data ) const =0;
    virtual SparseVector designSparse( SparseVector& x ) const =0;
    //virtual double score( 
    //    const valarray<double>& beta, const SparseVector& x ) const =0;
#ifndef POLYTOMOUS
    virtual void displayModel( ostream& o, const Vector& beta, double threshold=0 ) const=0;
#endif //no POLYTOMOUS
    virtual ~IDesign() {}
};

class DesignRowSet : public IRowSet {
    const IDesign* pParent;
    IRowSet& data;
    SparseVector x;
public:
    DesignRowSet( const IDesign* pParent_, IRowSet& data_ ) : pParent(pParent_), data(data_) {};
    bool next() {
        if( !data.next() ) return false; //---->>--
        const SparseVector& xIn = data.xsparse();
        x.clear();
        for( SparseVector::const_iterator iIn=xIn.begin(); iIn!=xIn.end(); iIn++ )
            x.insert( data.colNumById(iIn->first), iIn->second ); //x[ data.colNumById(iIn->first) ] = iIn->second
        x.sort();
        x = pParent->designSparse( x );
        return true;
    }
    const SparseVector& xsparse() const {
        return x; 
    }
    unsigned colNumById( unsigned id ) const { return id; } //that's the rule of DesignRowSet
    //delegate to design
    unsigned dim() const { return pParent->dim(); }
    string colName( unsigned c ) const { return pParent->colName(c); }
    int colId( unsigned c ) const { return pParent->colId(c); }
    //delegate to data
#ifdef POLYTOMOUS
    bool ygood() const { return data.ygood(); }
    unsigned c() const { return data.c(); }
    int classId( unsigned c ) const { return data.classId(c); }
    unsigned classNumById( int c ) const { return data.classNumById(c); }
#endif //POLYTOMOUS
    YType y() const { return data.y(); }
    string rowName( unsigned r ) const { return data.rowName( r ); }
    void rewind() { data.rewind(); }
    string currRowName() const { return data.currRowName(); }
    unsigned n() const { return data.n(); }
#ifdef GROUPS
    const vector<bool>& groups() const  { return data.groups(); }
    bool group( unsigned g ) const  { return data.group(g); }
    unsigned ngroups() const { return data.ngroups(); }
#endif //GROUPS
};

class PlainDesign : public IDesign {
    const INameResolver& md_ref;

public:
    PlainDesign( const INameResolver & md )
        : md_ref(md)
    {
    }
    //EM only: Matrix MakeDense( const IDenseData & data ) const { return horizJoin( data.x(), Matrix( data.n(), 1, valarray<double>(1.0,data.n()) ) ); }
    DesignRowSet MakeRowSet( IRowSet& data ) const {
        return DesignRowSet(this,data); }
    SparseVector designSparse( SparseVector& x ) const {
        x.insert( md_ref.dim(), 1.0 ); //intercept //x[md_ref.dim()] = 1.0
        x.sort();
        return x;
    }
    /*double score( const valarray<double>& beta, const SparseVector& x ) const
    {
        double predictScore = beta[ dim()-1 ]; //intercept

        //merge beta and x
        // - go from x to beta
        for( SparseVector::const_iterator ix=x.begin(); ix!=x.end(); ix++ ) {
            try{
                unsigned b = md_ref.colNumById( ix->first );
                if( 0!=beta[b] )
                    predictScore += ix->second * beta[b];
            }catch(logic_error e){} //alien term - colNumById() failed
        }
        /**go from beta to x - obsolete
        for( int b=0;b<dim()-1;b++)
            if( 0!=beta[b] )
            {
                SparseVector::const_iterator ix=x.find( md_ref.colId(b) );
                if( ix!=x.end() )
                    predictScore += ix->second * beta[b];
            }*
            
        return predictScore;
    }*/
#ifndef POLYTOMOUS
    void displayModel( ostream& o, const Vector& beta, double threshold=0 ) const
    {
        vector<pair<double,int> > betaByValue;    //map<double,int>
        for( unsigned b=0;b<dim()-1;b++)
            betaByValue.push_back( pair<double,int>(beta[b],b) );  //[ beta[b] ] = b;
        sort( betaByValue.begin(), betaByValue.end() );
        for( vector<pair<double,int> >::const_iterator itr=betaByValue.begin();itr!=betaByValue.end();itr++)
            if( itr->first != 0 )
                o<<std::endl<<std::setw(20)
                <<itr->first<<"\t"<<colName(itr->second)<<"\t"<<colId(itr->second);
        o<<std::endl<<std::setw(20)<<beta[dim()-1]<<"\t"<<colName(dim()-1);
        if( 0!=threshold )
            o<<std::endl<<"\tTuned threshold: "<<threshold;
    }
#endif //no POLYTOMOUS

    // implement INameResolver
    unsigned dim() const { return md_ref.dim()+1; }
    string colName( unsigned c ) const { 
        return c<md_ref.dim() ? md_ref.colName(c) : "<Intercept>"; }
    int colId( unsigned c ) const { 
        return c<md_ref.dim() ? md_ref.colId(c) : 0; }
    unsigned colNumById( unsigned id ) const { 
        //throw logic_error("wrong sparsity with design");
        return 0==id ? md_ref.dim() : md_ref.colNumById(id); }
    unsigned interceptColNum() const { return md_ref.dim(); } //the last column
    virtual string rowName( unsigned r ) const { return md_ref.rowName(r); }
};

class InteractionsDesign : public IDesign {
    const INameResolver& md_ref;
    vector< pair<unsigned,unsigned> > ia; //interactions
public:
    InteractionsDesign( const INameResolver & md )
        : md_ref(md)
    {
        for( unsigned i=0; i<md_ref.dim(); i++ )
            for( unsigned j=i+1; j<md_ref.dim(); j++ )
                ia.push_back( pair<unsigned,unsigned>(i,j) );
    }
    InteractionsDesign( const IDenseData & data )
        : md_ref(data)
    {
        valarray<double> tmpcol( data.n() );
        for( unsigned i=0; i<md_ref.dim(); i++ )
            for( unsigned j=i+1; j<md_ref.dim(); j++ ) {
                tmpcol = COL( data.x(), i ) * COL( data.x(), j );
                if( tmpcol.sum() > 0 )
                    ia.push_back( pair<unsigned,unsigned>(i,j) );
            }
    }
    /*/EM only: 
    Matrix MakeDense( const IDenseData & data ) const {
        Matrix M( data.n(), dim() );
        valarray<double> tmpcol( data.n() );
        for( unsigned i=0; i<md_ref.dim(); i++ ) {
            tmpcol = COL( data.x(), i );
            COL( M, i ) = tmpcol;
        }
        for( unsigned i=0; i<ia.size(); i++ ) {
            tmpcol = COL( data.x(), ia[i].first ) * COL( data.x(), ia[i].second );
            COL( M, i+md_ref.dim() ) = tmpcol;
        }
        COL( M, dim()-1 ) = valarray<double>(1.0,data.n());

        return M;
    }*/
    DesignRowSet MakeRowSet( IRowSet& data ) const {
        return DesignRowSet(this,data); }
    SparseVector designSparse( SparseVector& x ) const {
        //throw logic_error("Interactions Rowset: not implemented yet");
        //HACK: 
        // input 'x' column id's have numbers from 0 to md_ref.dim()-1
        for( unsigned i=0; i<ia.size(); i++ ) {
            SparseVector::const_iterator a=x.find( ia[i].first );
            if( a!=x.end() )  {
                SparseVector::const_iterator b=x.find( ia[i].second );
                if( a!=x.end() )
                    x.insert( i+md_ref.dim(), a->second * b->second ); //x[ i+md_ref.dim() ]  = a->second * b->second
            }
        }
        x.insert( md_ref.dim()+ia.size(), 1.0 ); //[ md_ref.dim()+ia.size() ] = 1.0//intercept
        x.sort();
        return x;
    }
    /*double score( const valarray<double>& beta, const SparseVector& x ) const
    {
        double predictScore = beta[ dim()-1 ]; //intercept
        //overlap beta and x
        map<unsigned,double> overlap;
        for( unsigned i=0; i<md_ref.dim(); i++)
            if( 0!=beta[i] )
            {
                SparseVector::const_iterator ix=x.find( md_ref.colId(i) );
                if( ix!=x.end() ) { //found
                    predictScore += ix->second * beta[i];
                    overlap[i] = ix->second; 
                }
            }

        map<unsigned,double>::const_iterator it1, it2;
        for( unsigned i=0; i<ia.size(); i++)
            if( (it1=overlap.find(ia[i].first))!=overlap.end()
                &&(it2=overlap.find(ia[i].second))!=overlap.end() 
                )
                    predictScore += it1->second * it2->second * beta[i+md_ref.dim()];
           
        return predictScore;
    }*/
#ifndef POLYTOMOUS
    void displayModel( ostream& o, const Vector& beta, double threshold=0 ) const
    {
        throw logic_error("Not supported: InteractionsDesign::displayModel");
    }
#endif //no POLYTOMOUS
    // implement INameResolver
    unsigned dim() const { return md_ref.dim()+ia.size()+1; }
    string colName( unsigned c ) const { 
        if( c<md_ref.dim() ) return md_ref.colName(c);
        else if( c<md_ref.dim()+ia.size() ) {
            const pair<unsigned,unsigned>& ia_ = ia[c-md_ref.dim()];
            string s = string(md_ref.colName(ia_.first))+"*"+md_ref.colName(ia_.second);
            return s;
        }
        else return "<Intercept>"; }
    int colId( unsigned c ) const { 
        return c<md_ref.dim() ? md_ref.colId(c) : 0; }
    unsigned colNumById( unsigned id ) const { 
        throw logic_error("wrong sparsity with design");
        return 0; }
    unsigned interceptColNum() const { return md_ref.dim(); } //the last column
};

#endif //_DATA_DESIGN_HEADER_
