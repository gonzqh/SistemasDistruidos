#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_timer.h"
#include "helper_cuda.h"

#include "hemi/grid_stride_range.h"
#include "hemi/launch.h"

#include "chag/pp/prefix.cuh"
#include "chag/pp/reduce.cuh"


#include <conio.h> 

namespace pp = chag::pp;

// Arreglos de memoria global
int* g_symbolsOut;
int* g_countsOut;
int* g_in;
int* g_decompressed;

// Memoria de dispositivo usada en PARLE
int* d_symbolsOut;
int* d_countsOut;
int* d_in;
int* d_totalRuns;
int* d_backwardMask;
int* d_scannedBackwardMask;
int* d_compactedBackwardMask;

const int NUM_TESTS = 11;
const int Tests[NUM_TESTS] = {
	10000, // 10K
	50000, // 50K
	100000, // 100K
	200000, // 200K
	500000, // 500K
	1000000, // 1M
	2000000, // 2M
	5000000, // 5M
	10000000, // 10M
	20000000, // 20M
	30000000, // 30M
};

const int PROFILING_TESTS = 100;
const int MAX_N = 1 << 26; // tamaño máximo de cualquier array que usamos.

void parleDevice(int *d_in, int n,
	int* d_symbolsOut,
	int* d_countsOut,
	int* d_totalRuns
	);

int parleHost(int *h_in, int n,
	int* h_symbolsOut,
	int* h_countsOut);

int rleCpu(int *in, int n,
	int* symbolsOut,
	int* countsOut);

__global__ void compactKernel(int* g_in, int* g_scannedBackwardMask, int* g_compactedBackwardMask, int* g_totalRuns, int n) {
	for (int i : hemi::grid_stride_range(0, n)) {

		if (i == (n - 1)) {
			g_compactedBackwardMask[g_scannedBackwardMask[i] + 0] = i + 1;
			*g_totalRuns = g_scannedBackwardMask[i];
		}

		if (i == 0) {
			g_compactedBackwardMask[0] = 0;
		}
		else if (g_scannedBackwardMask[i] != g_scannedBackwardMask[i - 1]) {
			g_compactedBackwardMask[g_scannedBackwardMask[i] - 1] = i;
		}
	}
}

__global__ void scatterKernel(int* g_compactedBackwardMask, int* g_totalRuns, int* g_in, int* g_symbolsOut, int* g_countsOut) {
	int n = *g_totalRuns;

	for (int i : hemi::grid_stride_range(0, n)) {
		int a = g_compactedBackwardMask[i];
		int b = g_compactedBackwardMask[i + 1];

		g_symbolsOut[i] = g_in[a];
		g_countsOut[i] = b - a;
	}
}

__global__ void maskKernel(int *g_in, int* g_backwardMask, int n) {
	for (int i : hemi::grid_stride_range(0, n)) {
		if (i == 0)
			g_backwardMask[i] = 1;
		else {
			g_backwardMask[i] = (g_in[i] != g_in[i - 1]);
		}
	}
}

void PrintArray(int* arr, int n){
	for (int i = 0; i < n; ++i){
		printf("%d, ", arr[i]);
	}
	printf("\n");
}

char errorString[256];

bool verifyCompression(
	int* original, int n,
	int* compressedSymbols, int* compressedCounts, int totalRuns){

	// descomprimir.
	int j = 0;

	int sum = 0;
	for (int i = 0; i < totalRuns; ++i) {
		sum += compressedCounts[i];
	}

	if (sum != n) {
		sprintf(errorString, "El tamaño descomprimido y original no son iguales %d != %d\n", n, sum);

		for (int i = 0; i < totalRuns; ++i){
			int symbol = compressedSymbols[i];
			int count = compressedCounts[i];

			printf("%d, %d\n", count, symbol);
		}
		return false;
	}

	for (int i = 0; i < totalRuns; ++i){
		int symbol = compressedSymbols[i];
		int count = compressedCounts[i];

		for (int k = 0; k < count; ++k){
			g_decompressed[j++] = symbol;
		}
	}

	// Verifica la compresion.
	for (int i = 0; i < n; ++i) {
		if (original[i] != g_decompressed[i]){

			sprintf(errorString, "El arreglo descomprimido y original no son iguales at %d, %d != %d\n", i, original[i], g_decompressed[i]);
			return false;
		}
	}

	return true;
}

// Obteniendo datos de prueba aleatorio para compresion.
// El tipo de datos generados son:
// 1,1,1,1,4,4,4,4,7,7,7,7,....
// de manera que hay muchas secuencias repetidas. 
int* generateCompressibleRandomData(int n){
	int val = rand() % 10;

	for (int i = 0; i < n; ++i) {
		g_in[i] = val;

		if (rand() % 6 == 0){
			val = rand() % 10;
		}
	}
	return g_in;
}


// Obtener datos de prueba aleatorio para compresion.
// El tipo de datos generados son:
// 1,5,8,4,2,6,....
// de manera completamente aleatoria.
int* generateRandomData(int n){
	for (int i = 0; i < n; ++i) {
		g_in[i] = rand() % 10;;

	}
	return g_in;
}

// usamos f para datos comprimidos con RLE y verificar la compression. 
template<typename F>
void unitTest(int* in, int n, F f, bool verbose)
{
	int totalRuns = f(in, n, g_symbolsOut, g_countsOut);

	if (verbose) {
		printf("n = %d\n", n);
		printf("Tamaño original  : %d\n", n);
		printf("Tamaño comprimido: %d\n", totalRuns * 2);
	}

	if (!verifyCompression(
		in, n,
		g_symbolsOut, g_countsOut, totalRuns)) {
		printf("Prueba fallida %s\n", errorString);
		PrintArray(in, n);

		exit(1);
	}
	else {
		if (verbose)
			printf("Prueba correcta!\n\n");
	}
}

// Perfil de implementacion de RLE en CPU
template<typename F, typename G>
void profileCpu(F rle, G dataGen) {
	for (int i = 0; i < NUM_TESTS; ++i) {
		int n = Tests[i];
		int* in = dataGen(n);

		StartCounter();

		for (int i = 0; i < PROFILING_TESTS; ++i) {
			rle(in, n, g_symbolsOut, g_countsOut);
		}
		printf("Para n = %d, en tiempo %.5f microsegundos\n", n, (GetCounter() / ((float)PROFILING_TESTS)) * 1000.0f);
		//printf("%.5f\n", (GetCounter() / ((float)PROFILING_TESTS)) * 1000.0f);

		// Prueba de unidad para asegurarse de que la compresion es correcta.
		unitTest(in, n, rle, false);
	}
}

// Perfil de implementacion de RLE en GPU
template<typename F, typename G>
void profileGpu(F rle, G dataGen) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int i = 0; i < NUM_TESTS; ++i) {

		int n = Tests[i];
		int* in = dataGen(n);

		// Tranfiere los datos al dispositivo.
		CUDA_CHECK(cudaMemcpy(d_in, in, n*sizeof(int), cudaMemcpyHostToDevice));

		// graba.
		cudaEventRecord(start);
		for (int i = 0; i < PROFILING_TESTS; ++i) {
			parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns);
		}
		cudaEventRecord(stop);
		cudaDeviceSynchronize();

		// Prueba de unidad para asegurarse de que la compresión es correcta
		unitTest(in, n, rle, false);

		float ms;
		cudaEventElapsedTime(&ms, start, stop);



		printf("Para n = %d, en tiempo %.5f microsegundos\n", n, (ms / ((float)PROFILING_TESTS)) *1000.0f);
		//printf("%.5f\n", (ms / ((float)PROFILING_TESTS)) *1000.0f);
	}
}

// Ejecuta varios test en la implemetacion(f) de RLE.
template<typename F>
void runTests(int a, F f) {
	printf("EMPIEZAN PRUEBAS UNITARIAS\n");

	for (int i = 4; i < a; ++i) {
		for (int k = 0; k < 30; ++k) {
			int n = 2 << i;

			if (k != 0) {
				// en la primera prueba se hacen con valores buenos para 'n'
				// en las otras dos pruebas se hacen con valores ligeramente ramdomizados
				n = (int)(n * (0.6f + 1.3f * (rand() / (float)RAND_MAX)));
			}

			int* in = generateCompressibleRandomData(n);

			unitTest(in, n, f, true);
		}
		printf("-------------------------------\n\n");
	}
}

int main(){

	srand(1000);
	CUDA_CHECK(cudaSetDevice(0));

	// Asignar recursos al dispositivo, estas matrices se utilizan globalmente en el programa
	CUDA_CHECK(cudaMalloc((void**)&d_backwardMask, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_scannedBackwardMask, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_compactedBackwardMask, (MAX_N + 1) * sizeof(int)));

	CUDA_CHECK(cudaMalloc((void**)&d_in, MAX_N* sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_countsOut, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_symbolsOut, MAX_N * sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&d_totalRuns, sizeof(int)));

	// asignar recursos al servidor. 
	g_in = new int[MAX_N];
	g_decompressed = new int[MAX_N];
	g_symbolsOut = new int[MAX_N];
	g_countsOut = new int[MAX_N];

	// Ejecutamos este codigo para correr muchas pruebas unitarias.
	/*
	runTests(21, rleCpu);
	runTests(21, parleHost);
	*/

	// Ejecutamos este codigo para perfilar el rendimiento. 
	printf("Perfil: datos aleatorios en CPU\n");
	profileCpu(rleCpu, generateRandomData);

	printf("Perfil: datos comprimibles en CPU\n");
	profileCpu(rleCpu, generateCompressibleRandomData);


	printf("Perfil: datos aleatorios en GPU\n");
	profileGpu(parleHost, generateRandomData);

	printf("Perfil: datos comprimibles GPU\n");
	profileGpu(parleHost, generateCompressibleRandomData);



	// Ejecutamos este código cuando queremos ejecutar NVPP en el algoritmo.
	/*
	int n = 1 << 23;
	unitTest(generateCompressibleRandomData(1<<23), n, rleGPU, true);
	*/

	// liberamos los arreglos del dispositivo.
	CUDA_CHECK(cudaFree(d_backwardMask));
	CUDA_CHECK(cudaFree(d_scannedBackwardMask));
	CUDA_CHECK(cudaFree(d_compactedBackwardMask));
	CUDA_CHECK(cudaFree(d_in));
	CUDA_CHECK(cudaFree(d_countsOut));
	CUDA_CHECK(cudaFree(d_symbolsOut));
	CUDA_CHECK(cudaFree(d_totalRuns));

	CUDA_CHECK(cudaDeviceReset());

	// liberamos los arreglos del servidor.
	delete[] g_in;
	delete[] g_decompressed;

	delete[] g_symbolsOut;
	delete[] g_countsOut;

	while (getch() != '\n');


	return 0;
}



// implementation of RLE on CPU.
int rleCpu(int *in, int n, int* symbolsOut, int* countsOut){

	if (n == 0)
		return 0; // nada para comprimir!

	int outIndex = 0;
	int symbol = in[0];
	int count = 1;

	for (int i = 1; i < n; ++i) {
		if (in[i] != symbol) {
			// ha terminado la ejecucion.
			// ejecuta la salida.
			symbolsOut[outIndex] = symbol;
			countsOut[outIndex] = count;
			outIndex++;

			// y empieza una nueva ejecucion:
			symbol = in[i];
			count = 1;
		}
		else {
			++count; // aun no termina la ejecucion.
		}
	}

	// ultima ejecucion de salida. 
	symbolsOut[outIndex] = symbol;
	countsOut[outIndex] = count;
	outIndex++;

	return outIndex;
}
// En el CPU se hace la preparacion para ejecutar parle, lanza PARLE en GPU y tranfiere los resultados de la CPU
int parleHost(int *h_in, int n,
	int* h_symbolsOut,
	int* h_countsOut){

	int h_totalRuns;

	// tranfiere datos de entrada al dispositivo.
	CUDA_CHECK(cudaMemcpy(d_in, h_in, n*sizeof(int), cudaMemcpyHostToDevice));

	// ejecuta.    
	parleDevice(d_in, n, d_symbolsOut, d_countsOut, d_totalRuns);

	// transfiere los resultados del dispositivo al host.
	CUDA_CHECK(cudaMemcpy(h_symbolsOut, d_symbolsOut, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_countsOut, d_countsOut, n*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(&h_totalRuns, d_totalRuns, sizeof(int), cudaMemcpyDeviceToHost));

	return h_totalRuns;
}

void scan(int* d_in, int* d_out, int N) {
	pp::prefix_inclusive(d_in, d_in + N, d_out);
}

// ejecuta parle en la GPU
void parleDevice(int *d_in, int n,
	int* d_symbolsOut,
	int* d_countsOut,
	int* d_totalRuns
	){
	hemi::cudaLaunch(maskKernel, d_in, d_backwardMask, n);
	scan(d_backwardMask, d_scannedBackwardMask, n);
	hemi::cudaLaunch(compactKernel, d_in, d_scannedBackwardMask, d_compactedBackwardMask, d_totalRuns, n);
	hemi::cudaLaunch(scatterKernel, d_compactedBackwardMask, d_totalRuns, d_in, d_symbolsOut, d_countsOut);
}


