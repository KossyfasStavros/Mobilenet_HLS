#define HALF_ENABLE_CPP11_CMATH 0
#include <ap_int.h>
#include <xil_cache.h>
#include <stdio.h>
#include "xconv.h"
//#include <fstream>
//#include "xparameters.h"
//#include <iostream>
#include <unistd.h>
#include "xtime_l.h" // XTime_GetTime()
// Macros
#define TIMEDIFF(t1,t2) (t2 - t1)
#define MILLISECONDS(t) (1000.0 * t / COUNTS_PER_SECOND)

// Globals
XTime start, end;

XConv myConv;
XConv_Config *myConv_cfg;

#include "newIMG.h"
//#include "w8s_dcrc.h"
#include "w8s_2first.h"
//#include "w8s_dcrc.h"
#include "biases.h"

volatile ap_uint<8> myIMG[150528] = {0}; // Maybe these need to be ints, but I don'ts remember the data type
volatile ap_uint<8> weights[19560];
volatile ap_int<32> bias[19560];
ap_uint<8> ret_val[200704];
//ap_fixed<32,13> ret_val[200704] __attribute__ ((aligned(4)));

// Start a test
void startTest() {
 XTime_GetTime(&start);
}
// End a test
void endTest() {
 XTime_GetTime(&end);
 double time_curr = TIMEDIFF(start, end);
 double msec = MILLISECONDS(time_curr);
 printf("Run-time = %.2f msec...\n", msec);
 // Achieved Bandwidth = (total bytes transferred) / (msec)
 /// Average Latency = (msec) / (total memory accesses)
}

void storeIMG(){
	for(int i = 0; i<150528;i++) myIMG[i] = IMGin[i].V.to_int();
}

void storeWeights(){ // only the weights for the first 2 layers are loaded
	for(int i = 0; i<576;i++) weights[i] = filters[i].V.to_int();
}

void readWeights(){
	for(int i = 0; i < 16*3*3*3; i++){
		//printf("%d\r\n",l1_filter[i].to_int());
		//std::cout << l1_filter[i] << std::endl;
	}
}

void storeBias(){ // all the biases are loaded
	for(int i = 0; i<12200;i++) bias[i] = biases[i].V.to_int();
}

void printRes(){
	float temp;
	//xil_printf("Got in printRes");
	//for(int i = 0; i < 9; i++) XConv_Set_returned_values_V(&myConv, i);
	for(int i = 0; i < 16; i++){
//		xil_printf("%lu\r\n", XConv_Get_returned_values_V(&myConv));
//		ret_val[i]=i;
//		temp = float(ret_val[i*12544]) * (2^(-10));
		temp = ret_val[i*12544] / 1024.;
//		printf("%f\r\n", temp);
//		printf("%b\r\n", (int)(ret_val[i*12544]));
		std::cout << ret_val[i*12544] << std::endl;

//		std::cout << XConv_Get_returned_values_V(res_reg + i) << std::endl;
	}
}

void initStuff(){
	Xil_DCacheDisable();
	int status = 0;
	myConv_cfg = XConv_LookupConfig(XPAR_CONV_0_DEVICE_ID);
	if(myConv_cfg){
		status = XConv_CfgInitialize(&myConv, myConv_cfg);
		if(status != XST_SUCCESS){
			xil_printf("Failed to initialize\r\n");
		}
		else {

			//ap_ctrl_ptr = (int*)(XPAR_XCONV_0_S_AXI_AXILITES_BASEADDR + XCONV_AXILITES_ADDR_AP_CTRL );

//			XConv_Set_length_r(&Conv, 100);

			XConv_Set_myIMG_V(&myConv, (u64)myIMG);
			XConv_Set_model_filters_V(&myConv, (u64)weights);
			XConv_Set_model_biases_V(&myConv, (u64)bias);
			XConv_Set_returned_values_V(&myConv, (u64)ret_val);


//			xil_printf("b4 IMG\r\n");
			storeIMG();
//			xil_printf("after IMG\r\n");
			storeWeights();
//			readWeights();
			storeBias();

		}
	}
}

int main(){

	initStuff();
	//xil_printf("After Init\r\n newwwwwwww\r\n\r\n");
	startTest();
	XConv_Start(&myConv);

	while(!XConv_IsDone(&myConv));
	endTest();
	printRes();
//	readWeights();

	XConv_DisableAutoRestart(&myConv);

    return 0;
}

