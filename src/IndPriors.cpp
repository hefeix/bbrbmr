#include <fstream>
#include <sstream>
#include <string>
#include <limits>
using namespace std;

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

/*
format for the ind prior file for bbrtrain:
  <feature_id> <mode> <variance> [<skew>]

format for the ind prior file for bmrtrain:
  class <class_id> <feature_id> <mode> <variance> [<skew>]
*/

void PriorTermsByTopic:: initIndPriors() {
    // open the individual prior file
    ifstream ifs(m_indPriorsFile.c_str());
    // if cannot open the file, exit with error
    if( !ifs.good() )
	throw runtime_error(string("Cannot open individual priors file ")+m_indPriorsFile);

    int nrows = 0;
    string topic;
    IndPriors indPriors(m_indPriorsMode);
    Log(10)<<"\nReading  individual priors file "<<m_indPriorsFile;

    while(  ifs.good() )
    {
	// read in one line from the prior file
	string buf;
	getline( ifs, buf );

	std::istringstream rowbuf( buf);

	// read in the keyword, i.e., the first word in the line
	string keyword;
	rowbuf>>keyword;
	// if the reading fails, i.e., the line is empty; stop reading the file ???
	if( rowbuf.fail() ) //empty line // 2.10 Jun 20, 05  bug with ind.priors file read - fixed
	    // break;      //--->>--
	    continue;  // [modified by shenzhi, for task 4, 6] 

	// [added by shenzhi for task 6]
	// if the line starts with #, skip it;
	if(keyword[0]=='#') 
	    continue;

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
    } // end of while;
    
    if( !m_multiTopic )
	m_indPriorsByTopic[topic] = indPriors;
    if( ! ifs.eof() ) { //the only legal way to end the loop
	std::ostringstream buf; 
	buf<<"Corrupt individual priors file after line # " << nrows;
	throw runtime_error(buf.str());
    }
};


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
