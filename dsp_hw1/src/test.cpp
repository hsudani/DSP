#include "hmm.h"
#include <math.h>

class testmodels
{
private:
    /* data */
public:
    testmodels(/* args */);
    ~testmodels();
};

testmodels::testmodels(/* args */)
{
}

testmodels::~testmodels()
{
}


int main()
{
/*
	HMM hmms[5];
	load_models( "modellist.txt", hmms, 5);
	dump_models( hmms, 5);
*/
	HMM hmm_initial;
	loadHMM( &hmm_initial, "../model_init.txt" );
	dumpHMM( stderr, &hmm_initial );

	printf("log(0.5) = %f\n", log(1.5) );
	return 0;
}
