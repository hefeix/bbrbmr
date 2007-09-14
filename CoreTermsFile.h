#ifndef CORE_TERMS_RESTRICTION_FILES_
#define CORE_TERMS_RESTRICTION_FILES_

#include <set>
#include <string>
#include <fstream>
#include <sstream>

#include "Param.hpp"
#include "logging.h"

class CoreTermsRestriction {
    std::string fileMask;
    std::set<int> indices;
public:
    bool isOn() const { return fileMask.size()>0; }
    bool has( int i ) const {
        if( isOn() )
            return( indices.find(i) != indices.end() );
        else
            return true; //no restriction
    }
    size_t nterms() const {
        return( isOn() ? indices.size() : 0 );   }
    void get() {
        fileMask = ParamGetString("coreFile","");
        Log(2)<<std::endl<<"Core Terms File Mask: "
            <<( isOn() ? fileMask.c_str() : "<none>");
    }
    bool setTopic( const char* topic ){
        if( !isOn() )
            return true; //--->>--

        indices.clear();

        std::string fileName=fileMask;
        size_t pos=fileName.find_first_of('?');
        if( pos<fileName.size() )
            fileName.replace(pos,1,topic);

        std::ifstream f(fileName.c_str());
        if( !f.good() )
            return false; //--->>--
        const int bufsize=1000;
        char buf[bufsize+1];
        while( f.getline( buf, bufsize ).good() ) {
            int i;
            istringstream rowbuf( buf );
            rowbuf >> i/*ignore first*/ >> i;
            if( rowbuf.fail() )
                return false; //--->>--
            indices.insert( i );
        }
        return true;
    }
};

#endif //CORE_TERMS_RESTRICTION_FILES_
