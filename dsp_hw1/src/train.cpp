#include "hmm.h"
#include <math.h>
#include"models.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
int main()
{
/*
	HMM hmms[5];
	load_models( "modellist.txt", hmms, 5);
	dump_models( hmms, 5);
*/
	// printf("train log(0.5) = %f\n", log(1.5) );
	HMM hmm_initial;
	const char* file_path="test1.txt";
	FILE *ouput_model = open_or_die(file_path,"w");
	loadHMM( &hmm_initial, "model_init.txt" );
	const char* path = "data/train_seq_01.txt";
	const int epoc = 1;
	printf("%s\n",hmm_initial.model_name);
	// dumpHMM( stderr, &hmm_initial );
	train_models train(hmm_initial);
	train.read_file(path);
	train.train(epoc);
	dumpHMM(ouput_model,&hmm_initial);
	

	printf("train log(0.5) = %f\n", log(1.5) );
	return 0;
}
