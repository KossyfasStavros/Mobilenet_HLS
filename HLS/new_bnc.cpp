#include <fstream>
#include <iostream>
#include <string>
#include <assert.h>

//#define downscaled_img 224
#define pixels 50176
#define filter_elements 1856
#include "convol.h"


void serializeIMG(volatile IMGtype myIMG[3*224*224]);
void loadFilters(volatile kernT  *filters);
void loadBiases(biasInType *biases);
void print_some(volatile finalResultType res[outD*outW*outW], int how_many, bool float_notInt);
int compareResults(volatile finalResultType res[outD*outW*outW]);

int main(){

	finalResultType res[outD][outW][outW];
	finalResultType lin_res[200704]; // This needs to be large enough to accomodate all the results from the most needed layer
	volatile IMGtype myIMG[3*224*224];
//	volatile kernT filters[5578216]; //The full network
	volatile kernT filters[filter_elements]; // The first two layers. This is trouble...
	biasInType biases[12200];

	loadFilters(filters);
	loadBiases(biases);
	serializeIMG(myIMG);

	conv(myIMG, filters, biases, lin_res);
//	compareResults(lin_res);

	print_some(lin_res, 12, false); //*12544
	//*/
	return 0;
}

void print_some(volatile finalResultType res[outD*outW*outW], int how_many, bool float_notInt){
	if(how_many > 0 && how_many < 12544){
		for(int i = 0; i < how_many; i++){
			for(int j=0;j<16;j++){
				finalResultType temp = res[i + j*12544]; // i + j*12544
				if(float_notInt) std::cout << ((temp-3)*0.14285412430763245) << std::endl;
				else std::cout << temp << std::endl;
			}
		}
	}
	else{
		std::cout << "\n\nGiven number exceeds results amount or is negative\n\n";
	}
}

void serializeIMG(volatile IMGtype myIMG[3*224*224]){
	std::string txt;
	std::ifstream input_file("int8IMG_crc.dat");
	int color, row, column;
	int i = 0;
	while(getline(input_file, txt)){
		//myIMG[i / 50176][(i%50176)/224][(i%50176)%224] = (IMGtype)stoi(txt); // Incorrect for the Classes/pythonIMG_sub128.dat
//		color = (i%672)%3;
//		row = i/672;
//		column = (i%672)/3;
//		myIMG[i] = (IMGtype)(stoi(txt)/128.);
		myIMG[i] = (IMGtype)(stoi(txt));
		//std::cout << "color: " << i / 50176 << ", row: " << (i%50176)/224 << ", column: " << (i%50176)%224 << ", value: " << myIMG[i / 50176][(i%50176)/224][(i%50176)%224] << std::endl;
		i++;
	}
}

void loadFilters(volatile kernT *filters){
	std::string txt;
//	std::ifstream input_file("int8_w8s_dcrc.dat");
	std::ifstream input_file("int8_w8s_drcc.dat");

	for(int i = 0; i<filter_elements;i++){
		getline(input_file, txt);
		filters[i] = (ap_fixed<16, 8>)stof(txt);
	}
	input_file.close();
}

void loadBiases(biasInType *biases){
	std::string txt;
	std::ifstream input_file("int8_biases_dcrc.dat");
	for(int i = 0; i<19561; i++){ // There is something I need to figure on line 1752
		getline(input_file, txt);
//		biases[i] = (ap_fixed<16, 8>)stof(txt);
//		std::cout << i <<std::endl;
		biases[i] = (biasInType)stoi(txt);
	}
}


int compareResults(volatile finalResultType res[outD*outW*outW]){

	int i = 0;
	finalResultType temp;
	std::string outside_result;
	std::ifstream res_file;
	res_file.open("py_L2res.dat");
	i = 0;
	int total = 0;
	for(int i = 0; i < 12544; i++){
		for(int j=0;j<16;j++){
			temp = res[i + j*12544];
			getline(res_file, outside_result);
#define findDifferences
#ifdef findDifferences
			if(std::abs((float)temp / std::stof(outside_result) - 1) > 0.5){
				std::cout << "Location: " << i << ", Variance: " << (float)temp / std::stof(outside_result) - 1 << std::endl;
				std::cout << "HLS: " << temp << ", Python: " << outside_result<< std::endl << std::endl;
				total++;
			}
#endif
#ifndef WriteOutB
//			assert(std::abs(temp.to_float() / std::stof(outside_result) - 1) < 0.5);
#endif
		//std::cout << "color: " << i / 50176 << ", row: " << (i%50176)/224 << ", column: " << (i%50176)%224 << ", value: " << myIMG[i / 50176][(i%50176)/224][(i%50176)%224] << std::endl;
		}
	}
	std::cout << total <<std::endl;
	return 0;
}
