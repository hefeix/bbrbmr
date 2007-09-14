#ifndef logging_level_controlled_
#define logging_level_controlled_

#include <iostream>
#include <fstream>
#include <strstream>
#include <string>
#include <time.h>

class logging {
    unsigned int m_level;
    std::ostrstream dull;
    time_t start;
    std::ofstream logstream;
    bool isstdout;
    void init() {
        dull.freeze();
        start = ::time(NULL);
    }
public:
    logging(unsigned int level_=0) : m_level(level_) {
        isstdout = true;
        init(); }
    logging( std::string name, unsigned int level_=0) : m_level(level_), logstream((name+".lis").c_str()) {
        isstdout = false;
        init(); }
    void setLevel(unsigned int level_ ) {
        m_level = level_;     }
    std::ostream& operator()(unsigned int l=0) {
        if( l<=m_level) //return( isstdout ? std::cout : logstream );
            if( isstdout ) return std::cout;
            else return logstream;
        else return dull;
    }
    time_t time() const { return ::time(NULL) - start; }
    unsigned int level() const {  return m_level; }
};

extern logging Log;

#endif //logging_
