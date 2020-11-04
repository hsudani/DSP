#include <math.h>
// #include "../inc/hmm.h"
// #include "../inc/models.h"
#include <iostream>
using namespace std;


int main()
{
/*
	HMM hmms[5];
	load_models( "modellist.txt", hmms, 5);
	dump_models( hmms, 5);
*/

	int a[62][3] = {{1, 2, 3},{4, 5, 6}};
    int b [3]= {8,8,8};
    int N = 10000;
    double alpha[N];
    // alpha[0] = hmm_initial.initial;
    
    printf("hihihih\n" );
    printf("%d\n",a[0][0] );
    printf("%d\n",a[1][2] );
    printf("%d%d%d\n",b[0],b[1],b[2]);
    b[0] = a[0][0];
    printf("%d%d%d\n",b[0],b[1],b[2]);
    printf("%d\n",a[1][2] );
    // memset(a, 0, sizeof a);
    printf("%d\n",a[0][0] );
    printf("%d\n",a[1][2] );
    printf("log(0.5) = %f\n", log(1.5) );
	return 0;
}