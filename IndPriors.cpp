#include <fstream>
#include <sstream>
#include <string>
#include <limits>
using namespace std;
#ifndef _MSC_VER
# define  stricmp(a, b)   strcasecmp(a,b) 
# include <strings.h>
#endif

#include "logging.h"
#include "PriorTerms.h"

#define KW_topic "topic"
#define KW_endoftopic "endoftopic"
#define KW_class "class"

static TModeVarSkew parsePriorsLine( istringstream& s )
{
    TModeVarSkew p(0,1,0); //skew is optional, so needs to be initialized here
    std::string varstr;

    s>>p.mode>>varstr;
    if( 0==stricmp( "inf", varstr.c_str() ) )
        p.var = std::numeric_limits<double>::infinity();
    else{
        std::istringstream varbuf(varstr);
        varbuf>>p.var;
    }

    if( s.fail() ) //feat, mode,var are mandatory
        throw runtime_error(string("Corrupt individual priors file line: ") + s.str());

    s>>p.skew; //optional
    if( p.skew!=0 && p.skew!=1 && p.skew!=-1 )
        throw runtime_error(string("Illegal skew value: should be 1, 0, or -1"));

    return p;
}

void PriorTermsByTopic:: initIndPriors() {
        ifstream ifs(m_indPriorsFile.c_str());
        if( !ifs.good() )
            throw runtime_error(string("Cannot open individual priors file ")+m_indPriorsFile);

        int nrows = 0;
        string topic;
        IndPriors indPriors(m_indPriorsMode);
        Log(10)<<"\nReading  individual priors file "<<m_indPriorsFile;
        while(  ifs.good() )
        {
            string buf;
            getline( ifs, buf );
            //check for keyword
            std::istringstream rowbuf( buf);
            string keyword;
            rowbuf>>keyword;
            if( rowbuf.fail() ) //empty line // 2.10 Jun 20, 05  bug with ind.priors file read - fixed
                break;      //--->>--
            if( 0==stricmp( KW_topic, keyword.c_str() ) ) {
                string newtopic;
                rowbuf>>newtopic;
                topic = newtopic;
            }
            else if( 0==stricmp( KW_endoftopic, keyword.c_str() ) ) {
                m_indPriorsByTopic[topic] = indPriors;
                indPriors = IndPriors(m_indPriorsMode); //renew
            }
            else //next row within the topic
            if( 0==stricmp( KW_class, keyword.c_str() ) ) { //class-specific prior
                int iclass;
                unsigned feat;
                rowbuf>>iclass>>feat;  //>>mode>>varstr;
                TModeVarSkew prior = parsePriorsLine(rowbuf);
                indPriors.m_byClass[iclass][feat] = prior;
                Log(10)<<"\n- "<<iclass<<" "<<feat<<" "<<prior.mode<<" "<<prior.var<<" "<<prior.skew;
            }
            else {//common prior
                std::istringstream rowbuf2( buf); //read whole line from the begining
                unsigned feat;
                rowbuf2>>feat;  //>>mode>>varstr;
                TModeVarSkew prior = parsePriorsLine(rowbuf2);
                indPriors.m[feat] = prior;
                Log(10)<<"\n- "<<topic<<" "<<feat<<" "<<prior.mode<<" "<<prior.var<<" "<<prior.skew;
            }
            nrows ++;
        }
        if( !m_multiTopic )
            m_indPriorsByTopic[topic] = indPriors;
        if( ! ifs.eof() ) { //the only legal way to end the loop
            std::ostringstream buf; 
            buf<<"Corrupt individual priors file after line # " << nrows;
            throw runtime_error(buf.str());
        }
};

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
