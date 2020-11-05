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
    int result_path[TRAIN_SIZE]={0};
    double result_prob[TRAIN_SIZE]={0};

public:
    testmodels(/* args */);
    ~testmodels();
    void read_file(const char*);
    void read_models(const char *);
    double recursion(HMM &, const char*);
    void run_all_models();
    void output_result(const char*);
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
    // filename : 
    FILE *fp = open_or_die(filename, "r");
    char token[MAX_LINE] = "";
    model_num=0;
    while(fscanf(fp, "%s", token) > 0 && model_num<MAX_MODEL)
    {
        HMM temp;
        loadHMM(&temp,token);
        hmm[model_num] = temp;
        model_num++;
    }
    // printf("model num : %d",model_num);
    fclose(fp);
};

double testmodels::recursion(HMM &h, const char *Ot){
    // initiate
    for(int i=0;i<h.state_num;++i){
        sigma[0][i]=(h.initial[i])*(h.observation[Ot[0]-'A'][i]);
    }
    // recursion
    for(int t=1;t<test_len;++t){
        for(int j=0;j<h.state_num;++j){
            sigma[t][j]=0;
            // get max
            for(int i=0;i<h.state_num;++i){
                if(sigma[t][j] < ((sigma[t-1][i])*(h.transition[i][j])) ){
                    sigma[t][j] = (sigma[t-1][i])*(h.transition[i][j]);
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
    for(int t=0;t<test_lines;++t){
        double max_prob=0;
        int max_path=0;
        for(int i=0;i<model_num;++i){
            double prob=0;
            prob=recursion(hmm[i],testing_data[t]);
            if(max_prob<prob){
                max_prob=prob;
                max_path=i;
            }
        }
        result_prob[t]=max_prob;
        result_path[t]=max_path+1;
    }
}
void testmodels::output_result(const char * path){
    FILE *fl = open_or_die(path,"w");
    for(int i=0;i<test_lines;++i){
        fprintf(fl,"model_%02d.txt %.7e\n",result_path[i],result_prob[i]);
    }
    
    
}


int main(int argc, char *argv[])
{
/*
	HMM hmms[5];
	load_models( "modellist.txt", hmms, 5);
	dump_models( hmms, 5);
*/
    if(argc!=4){
        printf("incorrect inputs : %d\n",argc);
        exit(-1);
    }
	testmodels testing;
    // model list path : argv[1]
    testing.read_models(argv[1]);
    // seq path : argv[2]
    testing.read_file(argv[2]);
    // testing
    testing.run_all_models();
    // output path : argv[3]
    testing.output_result(argv[3]);

	return 0;
}
