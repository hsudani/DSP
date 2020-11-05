#include "hmm.h"
#include <math.h>

#define TRAIN_SIZE 10240

class trainmodels
{
private:
    HMM *hmm;
    double alpha[MAX_SEQ][MAX_STATE]= {0.0};
    double beta[MAX_SEQ][MAX_STATE] = {0.0};
    double epsilon[MAX_SEQ][MAX_STATE][MAX_STATE]={0.0};
    double gamma[MAX_SEQ][MAX_STATE]={0.0};
    double gamma_observe[MAX_OBSERV][MAX_STATE]={0.0};
    char training_data[TRAIN_SIZE][MAX_SEQ]={'\0'};
    int train_lines = 0;
    int train_len = 0;
    double P_term[MAX_SEQ] = {0.0};
    double accumulate_gamma[MAX_STATE]={0.0};
    double accumulate_epsilon_t1[MAX_STATE][MAX_STATE] = {0.0};
    double accumulate_gamma_t1[MAX_STATE]={0.0};
    double accumulate_observ_gamma[MAX_OBSERV][MAX_STATE]={0.0};
    double accumulate_gamma_t[MAX_STATE]={0.0};

public:
    trainmodels(/* args */);
    trainmodels(HMM &);
    ~trainmodels();
    void read_file(const char*);
    void forward(const char*);
    void backward(const char*);
    void calculate_gamma(const char*);
    void calculate_epsilon(const char*);
    void accumulate();
    void reset_var();
    void update_model();
    void train(int);
};

trainmodels::trainmodels(/* args */)
{
    // alpha[0][0]=0;
};

trainmodels::~trainmodels()
{
};

void trainmodels::reset_var(){
    memset(alpha, 0, sizeof alpha);
    memset(beta, 0, sizeof beta);
    memset(epsilon, 0, sizeof epsilon);
    memset(gamma, 0, sizeof gamma);
    memset(gamma_observe, 0, sizeof gamma_observe);
    memset(P_term, 0, sizeof P_term);
    memset(accumulate_gamma, 0, sizeof accumulate_gamma);
    memset(accumulate_epsilon_t1, 0, sizeof accumulate_epsilon_t1);
    memset(accumulate_gamma_t1, 0, sizeof accumulate_gamma_t1);
    memset(accumulate_observ_gamma, 0, sizeof accumulate_observ_gamma);
    memset(accumulate_gamma_t,0, sizeof accumulate_gamma_t);
    
};


trainmodels::trainmodels(HMM &h):hmm(&h)
{
    // hmm = &h;   
    for (int i = 0; i < TRAIN_SIZE; ++i)
        memset(training_data[i], '\0', sizeof(char) * MAX_LINE);
    reset_var();
    
     
};


void trainmodels::read_file(const char *filename)
{
    FILE *fp = open_or_die(filename, "r");
    char token[MAX_LINE] = "";
    
    while(fscanf(fp, "%s", token) > 0 )
    {
        strcpy(training_data[train_lines], token);
        train_lines++;
    }
    train_len = strlen(token);

    fclose(fp);
};


void trainmodels::forward(const char* Ot){
    // init
    for(int i=0;i<hmm->state_num;++i){
        alpha[0][i]= (hmm->initial[i]) * (hmm->observation[Ot[0]-'A'][i]);
    }
    // induction
    for(int t=0;t<train_len-1;++t){
        for(int j=0; j<hmm->state_num;++j){
            double sum = 0;
            for(int i = 0;i<hmm->state_num;++i)
                sum += (alpha[t][i])*(hmm->transition[i][j]);
            alpha[t+1][j] = sum*hmm->observation[Ot[t+1]-'A'][j];
        }
    }
    for(int t = 0; t<train_len;++t){
        double sum=0;
        for(int j=0; j<hmm->state_num;++j){
            sum += alpha[t][j];
        }
        P_term[t] = sum;
    }
    // printf("forward done \nmax:%f, %d %d",max,max_int[0],max_int[1]);
};

void trainmodels::backward(const char* Ot){
    //initialize
    for(int i=0;i<hmm->state_num;++i)
        beta[train_len-1][i] = 1;
    // induction
    for(int t=train_len-2;t>-1;--t){
        for(int i=0;i<hmm->state_num;++i){
            for(int j=0;j<hmm->state_num;++j){
                beta[t][i] += hmm->transition[i][j]*hmm->observation[Ot[t+1]-'A'][j]*beta[t+1][j];
            }
        }
    }

};

void trainmodels::calculate_gamma(const char* Ot){
    for(int t=0;t<train_len;++t){
        double sum = 0;
        for(int j=0;j<hmm->state_num;++j){
            sum += alpha[t][j]*beta[t][j];
        }
        for(int i=0;i<hmm->state_num;++i){
            gamma[t][i] = alpha[t][i]*beta[t][i]/sum;
        }
    }
    for(int t=0;t<train_len;++t){
        for(int i=0;i<hmm->state_num;++i)
            gamma_observe[Ot[t]-'A'][i] += gamma[t][i];
    }
};

void trainmodels::calculate_epsilon(const char* Ot){
    for(int t=0;t<train_len-1;++t){
        double sum=0;
        for(int i=0;i<hmm->state_num;++i){
            for(int j=0;j<hmm->state_num;++j){
                sum += alpha[t][i]*(hmm->transition[i][j])*hmm->observation[Ot[t+1]-'A'][j]*beta[t+1][j];
            }
        }
        for(int i=0;i<hmm->state_num;++i){
            for(int j=0;j<hmm->state_num;++j){
                epsilon[t][i][j] = (alpha[t][i]*(hmm->transition[i][j])*hmm->observation[Ot[t+1]-'A'][j]*beta[t+1][j]) / sum;
            }
        }
    }
};

void trainmodels::accumulate(){
    for(int i=0;i<hmm->state_num;++i)
        accumulate_gamma[i] += gamma[0][i];
    
    for(int i=0;i<hmm->state_num;++i){
        for(int j=0;j<hmm->state_num;++j){
            for(int t=0;t<train_len-1;++t){
                accumulate_epsilon_t1[i][j] += epsilon[t][i][j];
            }
        }
        for(int aa=0;aa<train_len-1;++aa){
            accumulate_gamma_t1[i] += gamma[aa][i];
        }
    }
    
    for(int i=0;i<hmm->state_num;++i){
        for(int t=0;t<train_len;++t){
            accumulate_gamma_t[i] += gamma[t][i];
        }
    }
}



void trainmodels::update_model(){
    for(int i=0;i<hmm->state_num;++i){
        hmm->initial[i]=accumulate_gamma[i]/train_lines;
    }

    for(int i=0;i<hmm->state_num;++i){
        for(int j=0; j<hmm->state_num;++j){
            hmm->transition[i][j]=accumulate_epsilon_t1[i][j]/accumulate_gamma_t1[i];
        }
    }
    for(int i=0;i<hmm->state_num;++i){
        for(int ob=0;ob<hmm->observ_num;++ob){
            hmm->observation[ob][i] = gamma_observe[ob][i]/accumulate_gamma_t[i];
            // accumulate_observ_gamma[ob][i]/accumulate_gamma_t[i];
        }
    }
    
};

void trainmodels::train(int epoch=1){
    for(int ep=0;ep<epoch;++ep){
        reset_var();
        for(int l=0;l<train_lines;++l){
            forward(training_data[l]);
            backward(training_data[l]);
            calculate_gamma(training_data[l]);
            calculate_epsilon(training_data[l]);
            accumulate();
        }
        update_model();
    }
};

// ==================================================================================================================
int main(int argc, char *argv[])
{
/*
	HMM hmms[5];
	load_models( "modellist.txt", hmms, 5);
	dump_models( hmms, 5);
*/
    if(argc!=5){
        printf("incorrect inputs : %d\n",argc);
        exit(-1);
    }
    int iter =atoi(argv[1]);
    // printf("iter %d \n",iter);

    // load initial path : argv[2] = model_init_path
	HMM hmm_initial;
	loadHMM( &hmm_initial, argv[2] );
    trainmodels train_test(hmm_initial);
    
    // sequence path : argv[3] = seq_path
    train_test.read_file(argv[3]);
    train_test.train(iter);
    // output model path : argv[4]
    FILE *output_model = open_or_die(argv[4],"w");
    dumpHMM( output_model, &hmm_initial );
    dump_models( &hmm_initial,1 );
    
    return 0;
}
