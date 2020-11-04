#ifndef MODELS_HEADER_
#define MODELS_HEADER_

#include<iostream>
using namespace std;
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "hmm.h"

#define TRAIN_SIZE 10240


class train_models
{
private:
    HMM *MYHMM;
    double alpha[MAX_SEQ][MAX_STATE]= {0.0};
    double beta[MAX_SEQ][MAX_STATE] = {0.0};
    double epsilon[MAX_SEQ][MAX_STATE][MAX_STATE]={0.0};
    double gamma[MAX_SEQ][MAX_STATE]={0.0};
    double gamma_observe[MAX_OBSERV][MAX_STATE]={0.0};
    char training_data[TRAIN_SIZE][MAX_SEQ]={'\0'};
    int train_lines = 0;
    int train_len = 0;
    double P_term[MAX_SEQ] = {0.0};
    double accumulate_gamma[TRAIN_SIZE][MAX_STATE]={0.0};
    double accumulate_epsilon_t1[TRAIN_SIZE][MAX_STATE][MAX_STATE] = {0.0};
    double accumulate_gamma_t1[TRAIN_SIZE][MAX_STATE]={0.0};
    double accumulate_observ_gamma[TRAIN_SIZE][MAX_OBSERV][MAX_STATE]={0.0};
    double accumulate_gamma_t[TRAIN_SIZE][MAX_STATE]={0.0};

public:
    train_models(/* args */);
    train_models(HMM& );
    ~train_models();
    void read_file(const char*);
    void forward(const char*);
    void backward(const char*);
    void calculate_gamma(const char*);
    void calculate_epsilon(const char*);
    void accumulate(int);
    void reset_var();
    void update_model();
    void train(int);
};

void train_models::read_file(const char *filename)
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


void train_models::forward(const char* Ot){
    // init
    for(int i=0;i<MYHMM->state_num;++i)
        alpha[0][i]=MYHMM->initial[i]*MYHMM->observation[Ot[i]-'A'][i];
    // induction
    for(int t=0;t<train_len-1;++t){
        for(int j=0; j<MYHMM->state_num;++j){
            double sum = 0;
            for(int a = 0;a<MYHMM->state_num;++a)
                sum += alpha[t][a]*MYHMM->transition[a][j];
            alpha[t+1][j] = sum*MYHMM->observation[Ot[t+1]-'A'][j];
        }
    }
    for(int t = 0; t<train_len;++t){
        double sum=0;
        for(int j=0; j<MYHMM->state_num;++j){
            sum += alpha[t][j];
        }
        P_term[t] = sum;
    }
    // printf("forward done \nmax:%f, %d %d",max,max_int[0],max_int[1]);
};

void train_models::backward(const char* Ot){
    //initialize
    for(int i=0;i<MYHMM->state_num;++i)
        beta[train_len-1][i] = 1;
    // induction
    for(int t=train_len-2;t>-1;--t){
        for(int i=0;i<MYHMM->state_num;++i){
            double sum = 0;
            for(int b=0;b<MYHMM->state_num;++b){
                sum += MYHMM->transition[i][b]*MYHMM->observation[Ot[t+1]-'A'][b]*beta[t+1][b];
            }
            beta[t][i] = sum;
        }
    }

};

void train_models::calculate_gamma(const char* Ot){
    for(int t=0;t<train_len;++t){
        double sum = 0;
        for(int i=0;i<MYHMM->state_num;++i)
            sum += alpha[t][i]*beta[t][i];
        for(int i=0;i<MYHMM->state_num;++i){
            gamma[t][i] = alpha[t][i]*beta[t][i]/sum;
        }
    }
    for(int t=0;t<train_len;++t){
        for(int i=0;i<MYHMM->state_num;++i)
            gamma_observe[Ot[t]-'A'][i]+=gamma[t][i];
    }
};

void train_models::calculate_epsilon(const char* Ot){
    for(int t=0;t<train_len;++t){
        double sum=0;
        for(int i=0;i<MYHMM->state_num;++i){
            for(int j=0;j<MYHMM->state_num;++j){
                sum += alpha[t][i]*MYHMM->transition[i][j]*MYHMM->observation[Ot[t+1]-'A'][j]*beta[t+1][j];
            }
        }
        for(int i=0;i<MYHMM->state_num;++i){
            for(int j=0;j<MYHMM->state_num;++j){
                epsilon[t][i][j] = alpha[t][i]*MYHMM->transition[i][j]*MYHMM->observation[Ot[t+1]-'A'][j]*beta[t+1][j]/sum;
            }
        }
    }
    
};
void train_models::accumulate(int line){
    for(int i=0;i<MYHMM->state_num;++i)
        accumulate_gamma[line][i]=gamma[0][i];
    
    for(int i=0;i<MYHMM->state_num;++i){
        for(int j=0;j<MYHMM->state_num;++j){
            for(int t=0;t<train_len-1;++t)
                accumulate_epsilon_t1[line][i][j]+=epsilon[t][i][j];
        }
        for(int aa=0;aa<train_len;++aa)
            accumulate_gamma_t1[line][i]+=gamma[aa][i];
    }
    for(int i=0;i<MYHMM->state_num;++i){
        for(int ob=0;ob<MYHMM->observ_num;++ob)
        accumulate_observ_gamma[line][ob][i]=gamma_observe[ob][i];
    }
    for(int i=0;i<MYHMM->state_num;++i){
        for(int t=0;t<train_len;++t){
            accumulate_gamma_t[line][i]+=gamma[t][i];
        }
    }
}

void train_models::reset_var(){
    for(int i=0;i<train_len;++i){
        for(int j=0;j<MYHMM->state_num;++j){
            alpha[i][j] = 0;
            beta[i][j] = 0;
        }
    }
    // for(int t=0;t<train_len;++t){
    //     for(int i=0;i<train_len)
    // }

    for(int t=0;t<train_len;++t){
        for(int i=0;i<MYHMM->state_num;++i){
            for(int j=0; j<MYHMM->state_num;++j)
                epsilon[t][i][j] = 0;
        }
    }
    // memset(alpha, 0, sizeof alpha);
    // memset(beta, 0, sizeof beta);
    // memset(epsilon, 0, sizeof epsilon);
    memset(gamma, 0, sizeof gamma);
    memset(P_term, 0, sizeof P_term);
    
};

void train_models::update_model(){
    for(int i=0;i<MYHMM->state_num;++i){
        double sum=0;
        for(int tr=0;tr<train_lines;++tr){
            sum+=accumulate_gamma[tr][i];
        }
        MYHMM->initial[i]=sum;
    }

    for(int i=0;i<MYHMM->state_num;++i){
        for(int j=0; j<MYHMM->state_num;++j){
            double sum_gamma=0;
            double sum_epsilon=0;
            for(int tr=0;tr<train_lines;++tr){
                sum_epsilon += accumulate_epsilon_t1[tr][i][j];
                sum_gamma += accumulate_gamma_t1[tr][i];
            }
            MYHMM->transition[i][j]=sum_epsilon/sum_gamma;
        }
    }
    for(int i=0;i<MYHMM->state_num;++i){
        for(int ob=0;ob<MYHMM->observ_num;++ob){
            double sum_observe=0;
            double sum_gamma=0;
            for(int tr=0;tr<train_lines;++tr){
                sum_observe += accumulate_observ_gamma[tr][ob][i];
                sum_gamma += accumulate_gamma_t[tr][i];
            }

        }
    }
    // reset_var();
};

void train_models::train(int epoch=1){
    for(int ep=0;ep<epoch;++ep){
        reset_var();
        for(int l=0;l<train_lines;++l){
            forward(training_data[l]);
            backward(training_data[l]);
            calculate_gamma(training_data[l]);
            calculate_epsilon(training_data[l]);
            accumulate(l);
        }
        update_model();
    }
};

train_models::train_models(/* args */)
{
};

train_models::train_models(HMM &HMM_in){
    reset_var();
    MYHMM = &HMM_in;    
};

train_models::~train_models()
{
    // reset_var();
};

#endif
