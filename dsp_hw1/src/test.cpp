#include "hmm.h"
#include <math.h>

#define TRAIN_SIZE 10240
#define MAX_MODEL 10

class testmodels
{
private:
    /* data */
    HMM hmm[MAX_MODEL];
    char testing_data[TRAIN_SIZE][MAX_SEQ]={'\0'};
    double sigma[MAX_SEQ][MAX_STATE]={0};
    int test_lines = 0;
    int test_len = 0;
    int model_num = 0;


public:
    testmodels(/* args */);
    ~testmodels();
    void read_file(const char*);
    void read_models(const char *);
    double recursion(HMM &, const char*);
    void run_all_models();
};

testmodels::testmodels(/* args */)
{
};

testmodels::~testmodels()
{
};

void testmodels::read_file(const char *filename)
{
    FILE *fp = open_or_die(filename, "r");
    char token[MAX_LINE] = "";
    
    while(fscanf(fp, "%s", token) > 0 )
    {
        strcpy(testing_data[test_lines], token);
        test_lines++;
    }
    test_len = strlen(token);

    fclose(fp);
};

void testmodels::read_models(const char *filename)
{
    FILE *fp = open_or_die(filename, "r");
    char token[MAX_LINE] = "";
    int model_num=0;
    while(fscanf(fp, "%s", token) > 0 && model_num<MAX_MODEL)
    {
        HMM temp;
        char* modle_path = {'\0'};
        loadHMM(&temp,token);
        hmm[model_num] = temp;
        model_num++;
    }
    
    fclose(fp);
};

double testmodels::recursion(HMM &h, const char *Ot){
    // initiate
    for(int i=0;i<h.state_num;++i)
        sigma[0][i]=h.initial[i]*h.observation[Ot[0]-'A'][i];
    // recursion
    for(int t=1;t<test_len;++t){
        for(int j=0;j<h.state_num;++j){
            sigma[t][j]=0;
            for(int i=0;i<h.state_num;++i){
                if(sigma[t][j]<(sigma[t-1][j]*h.transition[i][j])){
                    sigma[t][j]=sigma[t-1][j]*h.transition[i][j];
                }
            }
            sigma[t][j] = sigma[t][j]*h.observation[Ot[t]-'A'][j];
        }
    }
    // termination
    double prob=0;
    for(int i=0;i<h.state_num;++i){
        if(prob<sigma[test_len-1][i]){
            prob=sigma[test_len-1][i];
        }
    }

    return prob;
};

void testmodels::run_all_models(){
    for(int i=0;i<model_num;++i){

    }
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
