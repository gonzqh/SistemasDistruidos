#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <setjmp.h>

#define N 100 // tamano de los vectores
#define ND 105 // tamano de diccionario
#define NW 100 // tamano de palabra

#define TRY do{ jmp_buf ex_buf__; if( !setjmp(ex_buf__) ){
#define CATCH } else {
#define ETRY } }while(0)
#define THROW longjmp(ex_buf__, 1)

int main(int argc, char** argv){
	
	int *dicc_index, iterator_index, size_diccionario, actual_index;
	char *dicc_data;

	char *temp_word_dicc;
	char *temp_word_input;
	
	dicc_index = (int *)malloc(ND*sizeof(int));
	dicc_data = (char *)malloc(ND*sizeof(char));
	//cadinput = (char *)malloc(N*sizeof(char));
	temp_word_dicc = (char *)malloc(NW*sizeof(char));
	temp_word_input = (char *)malloc(NW*sizeof(char));
	
	char cadinput[N] = "";
	
	
	#pragma acc data copy(cadinput)
	

	size_diccionario = 0;
	actual_index = 0;

	dicc_data[size_diccionario] = cadinput[0];
	dicc_index[size_diccionario] = 0;
	
	int index_word = 0;
	int word_temp_size = 1;	
	  clock_t t_ini, t_fin;
	  double secs;
	t_ini = clock();
	
	
	for(int pindex = 1; pindex < N; pindex++){
		
		temp_word_input[index_word] = cadinput[pindex];
		index_word++;
	
				
	int flag_existe = 1;
	
	#pragma acc kernels
	for(int i = 0 ; i <= actual_index ; i++){
		flag_existe = 1;
		int index_dicc_temp=0;
	
		for(int j = 0; j <= size_diccionario; j++){
			if(dicc_index[j] == i){
				temp_word_dicc[index_dicc_temp] = dicc_data[j];
				
				index_dicc_temp++;			
			}
		}
		for(int u = index_dicc_temp ; u < NW; u++){
			temp_word_dicc[u] = 0;
		}
		
		for(int r = 0; r<word_temp_size ; r++){
		if(temp_word_dicc[r] == temp_word_input[r]){}
			else{
				flag_existe = 0;
				break;
			}
		}
		if(flag_existe == 1){
		break;
		}
	}


	if(flag_existe == 1){
	word_temp_size++;
		
	}

	else{
		actual_index++;
		for(int u = 0; u <index_word ; u++){
			size_diccionario++;
			dicc_data[size_diccionario] = temp_word_input[u];
			dicc_index[size_diccionario] = actual_index;
		}
		index_word = 0;
		word_temp_size = 1;


	}
	
}

	
t_fin = clock();
	secs = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;
printf("%.16g milisegundos\n", secs * 1000.0);
 
return 0;
}
