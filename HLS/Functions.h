//#include <hls_stream.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include "dTypes.h"

// kernType could probably be  within the class if it does not change throughout the model
template <	typename dataInType, typename interWindowType, typename dataWidthType, int data_width,
			typename kernType, typename interKernelType, typename kernDepthType, int kernel_depth, typename kernWidthType, int kernel_width,
			typename bufferWidthType, typename perColorResType, typename numColorsType, int numColors,
			typename strideType, int stride, typename inScaleT, typename w8sScaleT, typename combScaleT, typename lastScaleT, typename inputReadCountType, int pad, typename biasType,
			typename combinationType, typename testResType, typename resultsType, bool HS, bool RL,int output_width, int output_depth, bool depth_separable,
			int inBias, int w8Bias, int lastBias> class layer{
public:

	//const static unsigned int HALF_SIZE = ((kernel_width - 1) / 2); // Check if I can optimize this
//	const static bufferWidthType buffer_width = (kernel_width - 1) * stride;
	const static unsigned short int buffer_width = (kernel_width - 1) * stride; // might need plain int
	//const static unsigned int buffer_width = kernel_width + (stride - 1);
//	const static inputReadCountType totalInputReadCount = data_width * data_width;// * numColors;
	const static int totalInputReadCount = data_width * data_width;
	//const static unsigned int inputReadCountType = log2(totalInputReadCount) //It requires the cmath header, so avoid it until I know for sure that it is done at preprocessing


	void conv_stride2 (volatile dataInType *layerInput, volatile kernT *layer_filter, volatile resultsType L1results[output_depth][output_width][output_width], biasInType *fold_bias, inScaleT inScale, w8sScaleT w8sScale, lastScaleT lastScale){

		const combScaleT scale = inScale * w8sScale; // this doesn't seem to be the issue
//		const ap_fixed<9,-9> scale = 0.000263835;
		kernType filter[kernel_width][kernel_width];
		dataInType line_buf[numColors][buffer_width][data_width + pad];
//#pragma HLS ARRAY_PARTITION variable=line_buf block factor=4 dim=2//I need to fix factor = 4
#pragma HLS ARRAY_PARTITION variable=line_buf cyclic factor=2 dim=3
		dataInType window[numColors][kernel_width][kernel_width];
//#pragma HLS ARRAY_PARTITION variable=window complete dim=0
		perColorResType per_color_result[numColors];
#pragma HLS INTERFACE ap_fifo port=per_color_result
		inputReadCountType inputReadCount[numColors] = {0};
#pragma HLS ARRAY_PARTITION variable=inputReadCount complete dim=0 //I'm not sure I need this
		combinationType comb = 0; // 23
		testResType testRes = 0;

//#pragma HLS LOOP_MERGE // Some dependencies prevent the merging! Also, if I can make it work, it might produce side effects! Validate output!

		for(numColorsType i = 0; i < numColors; i++){
			initial_bufNwin(&layerInput[i * data_width * data_width], line_buf[i], inputReadCount[i], window[i]); // What about multi depth layers?
		}

		for_y : for (dataWidthType y = 0; y < data_width / stride; y ++){
			for_x : for (dataWidthType x = 0; x < data_width / stride; x ++){
//#pragma HLS LOOP_FLATTEN // Cannot work because of multiple sub loops

//#pragma HLS DATAFLOW // this creates problems!
				filter_depth: for (kernDepthType depth_layer = 0; depth_layer < kernel_depth; depth_layer++){


//#pragma HLS LOOP_UNROLL //could it work with a dataflow or is the i breaking the ability for concurrency?
//#pragma HLS PIPELINE // or DATAFLOW

					num_colors: for (numColorsType i = 0; i < numColors; i++){
//#pragma HLS LOOP_UNROLL factor=3 // Make it sth that can divide all possivle colors exactly

						create_kernel(filter, depth_layer, i, layer_filter);
						per_color_result[i] = separable(window[i], filter);

//#define show_window
#ifdef show_window
//			if((y < 3 || y >= 109) && (x < 3 || x >= 109) && (depth_layer == 0)&&i==0){
			if((y ==0) && (x < 19) && (depth_layer == 0)&&i==0){
				std::cout  << y << "	" << x << '\n';
				win_i : for (ap_uint<2> i = 0; i < kernel_width; i++){
						win_j : for (ap_uint<2> j = 0; j < kernel_width; j++){
							std::cout << "      " << std::setw(10) << std::setprecision(10) << window[0][i][j];
						}
						std::cout  << '\n';
				}//*
				win_i1 : for (ap_uint<2> i = 0; i < kernel_width; i++){
						win_j1 : for (ap_uint<2> j = 0; j < kernel_width; j++){
							std::cout << std::setprecision(10) << filter[i][j] << " || ";
						}
						std::cout  << '\n';
				}//*/
				std::cout  << '\n';
			}
#endif
					}


					if(!depth_separable){
//						testRes = 0;
						comb = 0;
						for(numColorsType i = 0; i < numColors; i++){
//#pragma HLS LOOP_UNROLL // This might be too much in future layers
							comb += per_color_result[i]+ fold_bias[i];
//							testRes += per_color_result[i];
						}//*/
						//ap_int<33> intermediate = comb ;
						testRes = (comb) * scale; // If I use the fixed weights I can rid of the multiplication
//						testRes += fold_bias[depth_layer];
#ifndef SYNTHESIS
//						if(y==0 && x< 50 && depth_layer==0){
//							std::cout << "comb: " << comb << ", intermediate: " << intermediate << ", testRes: " << testRes << std::endl;
//							std::cout << "fold_bias: " << fold_bias[depth_layer] << ", scale: " << scale <<  std::endl << std::endl;
//						}
#endif
						if(HS) L1results[depth_layer][y][x] = (HardSwish(testRes) / lastScale) + lastBias;
						else if(RL) L1results[depth_layer][y][x] = (ReLu(testRes) / lastScale) + lastBias;
						else L1results[depth_layer][y][x] = (testRes / lastScale) + lastBias;
					}
					else{
						for(numColorsType color = 0; color < numColors; color++){
//#pragma HLS LOOP_UNROLL

							//testRes = per_color_result[color] * scale + fold_bias[color];
							testRes = (per_color_result[depth_layer]) * scale;
							if(HS) L1results[color][y][x]= (HardSwish(testRes) / lastScale) + lastBias;
							else if(RL) L1results[color][y][x] = (ReLu(testRes) / lastScale) + lastBias;
							else L1results[depth_layer][y][x] = (testRes / lastScale) + lastBias;
						}
					}
				}

				for(numColorsType i = 0; i < numColors; i++) update_bufNwin(&(layerInput[i * data_width * data_width]), line_buf[i], inputReadCount[i], window[i], x, y);
			}
		}
		return;
	}

	void initial_bufNwin(volatile dataInType *layerInput, dataInType line_buf[buffer_width][data_width + pad], inputReadCountType &read_count, dataInType window[kernel_width][kernel_width]){
//#pragma HLS FUNCTION_INSTANTIATE variable=pad
//#pragma HLS FUNCTION_INSTANTIATE variable=data_width
//#pragma HLS FUNCTION_INSTANTIATE variable=kernel_width
//#pragma HLS PIPELINE
		first_win_row:
		if (pad == 2){
//#pragma HLS DATAFLOW
			for (kernWidthType rep = 0; rep < kernel_width; rep++){
#pragma HLS UNROLL// This might need to be pipeline after all
				window[0][rep] = 0;
			}
		}
		else{
			for (kernWidthType rep = 0; rep < kernel_width; rep++){
#pragma HLS PIPELINE rewind
				window[0][rep] = layerInput[rep];
				read_count++;
			}
		}

		fill_line_buffer:
		for (bufferWidthType i = 0; i < buffer_width; i++){ ///---------------------------------------------------------------
			for (dataWidthType j = 0; j < data_width + pad ; j++){
//#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE rewind

				if(((pad == 1) && (j == data_width - kernel_width)) || (pad == 2 && (((j ==  data_width + pad - 1 - kernel_width) || (j == data_width + pad - kernel_width)) || ((i == 0) && (j < data_width + pad - kernel_width))))) {
					line_buf[i][j] = 0;
				}
				else{
					line_buf[i][j] = (layerInput[read_count]);
					read_count++;
				}
			}
		}

		fill_window:
		for(kernWidthType rep_x = 0; rep_x < kernel_width - 1; rep_x++){
			for (kernWidthType rep_y = 0; rep_y < kernel_width; rep_y++){
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE rewind// it probably can't go here
//#pragma HLS UNROLL factor=kernel_width
				window[rep_x + 1][rep_y] = line_buf[rep_x][data_width + pad - kernel_width + rep_y];
			}
		}

		return;
	}

	void create_kernel(kernType needed_kernel[kernel_width][kernel_width], kernDepthType depth, numColorsType color, volatile kernT *layer_filter){

#pragma HLS PIPELINE
		for (kernWidthType i = 0; i < kernel_width; i++){
			for (kernWidthType j = 0; j < kernel_width; j++){
#pragma HLS LOOP_FLATTEN
//#pragma HLS UNROLL factor=kernel_width
				needed_kernel[i][j] = layer_filter[depth*kernel_width*kernel_width*numColors + i * kernel_width*numColors + j*numColors + color]; // I need it in drcc order
//				needed_kernel[i][j] = layer_filter[depth*kernel_width*kernel_width*numColors + color*kernel_width*kernel_width  + i * kernel_width + j]; // I need it in dcrc order
//				needed_kernel[i][j] -= weightBias;
			}
		}
		return;
	}

	void update_bufNwin(volatile dataInType *layerInput, dataInType line_buf[buffer_width][data_width + pad],
					inputReadCountType &read_count, dataInType window[kernel_width][kernel_width], dataWidthType x, dataWidthType y){

		dataInType right[kernel_width];
		vertical_stride:
		if (stride * x == data_width + pad - kernel_width){ // I've reached the end of the line
			if(stride == 2){
				for (bufferWidthType i = 0; i < buffer_width; i++){
					for (dataWidthType j = 0; j < data_width + pad - 1; j++){
//#pragma HLS LOOP_FLATTEN
//#pragma HLS PIPELINE //might not work
						if (i < buffer_width - 1) {
							line_buf[i][j] = line_buf[i + 1][j];
						}
						else{
//#pragma HLS OCCURENCE buffer_width //the way it is written I can't OCCURENCE it
							if ((y + 1) * stride <= data_width + pad - buffer_width - 1){ // This should be converted to sth more modular
								line_buf[i][j] = (layerInput[read_count]);
								read_count++;
							}
							else line_buf[i][j] = 0;
						}
					}
				}
			}

			for (ap_uint<2> rep = 0; rep < pad; rep ++){

//#pragma HLS PIPELINE
				update_right_0:
				for (kernWidthType i = 0; i < kernel_width - 1; i++) right[i] = line_buf[i][0];
				if (stride == 2) right[kernel_width - 1] = line_buf[kernel_width - 1][0];
				else{
					if (pad == 2 && (rep == 0 || y == data_width + pad - kernel_width - 1)) right[kernel_width - 1] = 0;
					else {
						right[kernel_width - 1] = layerInput[read_count];
						read_count++;
					}
				}


				update_window_0:
				for (kernWidthType i = 0; i < kernel_width; i++){
#pragma HLS UNROLL
					for (kernWidthType j = 0; j < kernel_width - 1; j++) window[i][j] = window[i][j + 1];
					window[i][kernel_width - 1] = right[i];
					//window[i][kernel_width - 1] = line_buf[i][0];
				}

				update_line_buf_0:
				for (bufferWidthType i = 0; i < buffer_width; i++){
//#pragma HLS UNROLL
//#pragma HLS PIPELINE rewind
					for (dataWidthType j = 0; j < data_width + pad; j++){
//#pragma HLS LOOP_FLATTEN
						if (j < data_width + pad - 1){
							line_buf[i][j] = line_buf[i][j + 1];
						}
						else{
//#pragma HLS OCCURRENCE //cycle=(data_width+pad)  // It could be valid, but it doesn't recognize the constants
							if (i < buffer_width - 1){
								line_buf[i][data_width + pad - 1] = line_buf[i + 1][0];
							}
							else {
								if (stride == 1) line_buf[i][j] = right[kernel_width - 1];
								else{
									if (stride * (y + 1) >= data_width + pad - buffer_width - 1) line_buf[i][j] = 0;
									else{
										line_buf[i][j] = (layerInput[read_count]);
										read_count++;
									}
								}
							}
						}
					}
				}
			}
		}

		for (strideType rep = 0; rep < stride; rep++){ // this can stay. stride is either 0 or 1, so not too much trouble here probably
//#pragma HLS PIPELINE
			update_right:
			for (kernWidthType i = 0; i < kernel_width - 1; i++){
#pragma HLS UNROLL
				right[i] = line_buf[i][0];
			}
			/*if (((pad == 2) && ((stride * x == data_width + pad - kernel_width - 1) || (stride * y == data_width + pad - kernel_width)))) right[kernel_width - 1] = 0;
			else //*/
			if (stride == 1) {
				if(pad == 2 && ((x == data_width + pad - kernel_width - 1) || y == data_width + pad - kernel_width || (x == data_width + pad - kernel_width && y == data_width + pad - kernel_width - 1))) right[kernel_width - 1] = 0;
				else{
					right[kernel_width - 1] = (layerInput[read_count]);
					//if ((read_count == 100) || (read_count == 300)) std::cout << read_count / data_width <<  std::endl << read_count % data_width <<  std::endl;
					read_count++;
				}
			}
			else {
				right[kernel_width - 1] = line_buf[kernel_width - 1][0];
			}


			update_window:
			for (kernWidthType i = 0; i < kernel_width; i++){
				for (kernWidthType j = 0; j < kernel_width - 1; j++) window[i][j] = window[i][j + 1];
				window[i][kernel_width - 1] = right[i];
				//window[i][kernel_width - 1] = line_buf[i][0];
			}

			update_line_buf:
			for (bufferWidthType i = 0; i < buffer_width; i++){
				for (dataWidthType j = 0; j < data_width + pad; j++){
					if(j < data_width + pad - 1){
						line_buf[i][j] = line_buf[i][j + 1];
					}
					else{
						if (i < buffer_width - 1){
							line_buf[i][j] = line_buf[i + 1][0];
						}
						else {
							if (stride == 1) line_buf[i][j] = right[kernel_width - 1];
							else {
								if((stride * y >= data_width + pad - buffer_width - 1) || (rep == 1 && (stride * (x + rep) == data_width + pad - kernel_width)) || ( stride * x == data_width + pad - kernel_width && stride * (y + 1) == data_width + pad - buffer_width - 1)) line_buf[i][j] = 0;
								else{
									line_buf[i][j] = (layerInput[read_count]);
									read_count++;
								}
							}
						}
					}
				}
			}
		}
		return;
	}


	// Defines the actual calculation for one output value. For the first kernel ap_fixed<19, 7> is enough (but why not 18,7 ?),
	// but I need to test with the other 15 to make sure that all min and max results can be represented
	//inline ap_fixed<32,7> separable(ap_uint<8> window[kernel_width], ap_uint<7> y, ap_uint<7> x, ap_int<8> kernel[kernel_width][kernel_width]){//why inline?
	inline perColorResType separable(dataInType window[kernel_width][kernel_width], kernType kernel[kernel_width][kernel_width]){//why inline?
//#pragma HLS stable variable = window
//#pragma HLS stable variable = kernel
//#pragma HLS stable variable = y
//#pragma HLS stable variable = x
		//#pragma HLS function_instantiate variable=x
//#pragma HLS function_instantiate variable=kernel_width // does this work?
//#pragma HLS PIPELINE rewind
	//#pragma HLS function_instantiate variable=y
		//ap_fixed<32,7> result = 0; // it was 16,9 Just make sure that all numbers fit inside the dedicated digits
		perColorResType result = 0;
		interWindowType interWin;
		interKernelType interKern;
		win_i : for (kernWidthType i = 0; i < kernel_width; i++){
			win_j : for (kernWidthType j = 0; j < kernel_width; j++){
#pragma HLS loop_flatten
#pragma HLS PIPELINE rewind
				interWin = (window[i][j] - inBias);
				interKern = (kernel[i][j] - w8Bias);
				result += interWin * interKern;
			}
		}
		return result;
	}
	inline testResType HardSwish(testResType testRes){ // this was resultsType, but it didn't work when I wanted int8 as resultsType
//#pragma HLS function_instantiate variable=testRes
		if (testRes < -3){
				return 0;
		}
		else if (testRes < 3){
				return testRes * (testRes + 3) / 6;
		}
		return testRes;
	}

	inline testResType ReLu(testResType testRes){ // same here
#pragma HLS function_instantiate variable=testRes
		if (testRes > 0) return testRes;
		else return 0;
	}

};

	template <typename sizeT, int size>
	void return_res(volatile finalResultType *results, volatile finalResultType *addr){
		for(sizeT i = 0; i < size; i++){
			addr[i] = results[i];
		}
	}

	template<typename sizeT, int size, typename dataT, typename scl1T, typename scl2T, typename scl3T, typename biasT>
	void combine(volatile dataT *in1, volatile dataT *in2, volatile dataT *return_addr, scl1T scale1, scl2T scale2, scl3T scale3, biasT b1, biasT b2, biasT b3){
		dataT temp1, temp2;
//		ap_ufixed<32,8> temp3, temp4;
		for(sizeT i=0;i<size;i++){
			temp1 = in1[i];
			temp2 = in2[i];
//			temp3 = (temp1 - b1) * scale1;
//			temp4 = (temp2 - b2) * scale2;
//			return_addr[i] = (temp3 + temp4) / scale3 + b3;

			return_addr[i] = ((temp1 - b1) * scale1 + (temp2 - b2) * scale2) / scale3 + b3;
		}
	}

	template<typename sizeT, int size, typename dataT, typename scl1T> // By mistake I made BN, not SE
	void BN(volatile dataT *in1, scl1T scale1, ap_uint<8> val_bias){
		dataT temp1;
		for(sizeT i = 0; i < size; i++){
			temp1 = in1[i];
			in1[i] = ((temp1 - val_bias) * scale1 + 3) * (ap_ufixed<9,-1>)0.16666999459266663 / (ap_ufixed<2,-6>)0.003921507392078638;
		}
	}

	template<typename sizeT, int size, typename channelsT, int channels, typename dataT, typename scl1T, typename scl2T, typename scl3T>
	void SE(volatile dataT *in1, volatile dataT *in2, scl1T scale1, scl2T scale2, scl3T scale3){
		dataT temp1, temp2;
		for(channelsT j=0; j<channels; j++){
			temp2 = in2[j]  ;
			for(sizeT i = 0; i < size; i++){
				temp1 = in1[i];
				in1[i] = ((temp1 * scale1) * (temp2 * scale2)) / scale3; // if scale5==scale4*scale3 remove them all from the operation
			}
		}
	}


	template <typename sizeT, int size, typename channelsT, int channels, typename sumT, typename dataT>
	void avg_pool_fn(volatile finalResultType *addr, volatile finalResultType *results){
		dataT temp;
		sumT partial_sum; //maybe it could be 2d, but leave it for now
		for(channelsT j=0; j<channels; j++){
			partial_sum = 0;
			for(sizeT i = 0; i < size; i++){
				temp = addr[i]; // I did this to avoid any volatile operation incompatibilities
				partial_sum += temp;
			}
			results[j] = partial_sum / channels;
		}
	}

	layer<dInT, interWinT, l1::dwidthT, 224, kernT, interKernT, l1::kernDepthT, 16, l1::kernWidthT, 3, l1::bufWidthT, l1::perColResT, l1::numColT, 3, l1::strideT, 2,
	l1::inScaleT , l1::w8sScaleT, l1::combScaleT, l1::lastScaleT, l1::readCountT, 1, biasT, l1::combinationType, l1::testResT, resT, true, false, 112, 16, false, 128, 115, 3> layer1;

	layer<dInT, interWinT, l2::dwidthT, 112, kernT, interKernT, l2::kernDepthT, 1, l2::kernWidthT, 3, l2::bufWidthT, l2::perColResT, l2::numColT, 16,  l2::strideT, 1, l2::inScaleT, l2::w8sScaleT, l2::combScaleT, l2::lastScaleT ,
	l2::readCountT, 2, biasT, l2::combinationType, l2::testResT, resT, false, true, 112, 16, true, 3, 180, 0> layer2;

	layer<dInT, interWinT, l3::dwidthT, 112, kernT, interKernT, l3::kernDepthT, 16, l3::kernWidthT, 1, l3::bufWidthT, l3::perColResT, l3::numColT, 16,  l3::strideT, 1, l3::inScaleT, l3::w8sScaleT, l3::combScaleT, l3::lastScaleT,
	l3::readCountT, 0, biasT, l3::combinationType, l3::testResT, resT, false, false, 112, 16, false, 0, 117, 118> layer3;

	layer<dInT, interWinT, l4::dwidthT, 112, kernT, interKernT, l4::kernDepthT, 64, l4::kernWidthT, 1, l4::bufWidthT, l4::perColResT, l4::numColT, 16,  l4::strideT, 1, l4::inScaleT, l4::w8sScaleT, l4::combScaleT, l4::lastScaleT,
	l4::readCountT, 0, biasT, l4::combinationType, l4::testResT, resT, false, true, 112, 64, false, 110, 132, 0> layer4;

	layer<dInT, interWinT, l5::dwidthT, 112, kernT, interKernT, l5::kernDepthT, 1, l5::kernWidthT, 3, l5::bufWidthT, l5::perColResT, l5::numColT, 64, l5::strideT, 2,
	l5::inScaleT , l5::w8sScaleT, l5::combScaleT, l5::lastScaleT, l5::readCountT, 2, biasT, l5::combinationType, l5::testResT, resT, false, true, 56, 64, true, 0, 121, 0> layer5;

	layer<dInT, interWinT, l6::dwidthT, 56, kernT, interKernT, l6::kernDepthT, 1, l6::kernWidthT, 3, l6::bufWidthT, l6::perColResT, l6::numColT, 64,  l6::strideT, 1, l6::inScaleT, l6::w8sScaleT, l6::combScaleT, l6::lastScaleT ,
	l6::readCountT, 1, biasT, l6::combinationType, l6::testResT, resT, false, false, 56, 24, false, 3, 180, 0> layer6;

	layer<dInT, interWinT, l7::dwidthT, 56, kernT, interKernT, l7::kernDepthT, 72, l7::kernWidthT, 1, l7::bufWidthT, l7::perColResT, l7::numColT, 24,  l7::strideT, 1, l7::inScaleT, l7::w8sScaleT, l7::combScaleT, l7::lastScaleT,
	l7::readCountT, 0, biasT, l7::combinationType, l7::testResT, resT, false, true, 56, 72, false, 0, 117, 118> layer7;

	layer<dInT, interWinT, l8::dwidthT, 56, kernT, interKernT, l8::kernDepthT, 1, l8::kernWidthT, 3, l8::bufWidthT, l8::perColResT, l8::numColT, 72,  l8::strideT, 1, l8::inScaleT, l8::w8sScaleT, l8::combScaleT, l8::lastScaleT,
	l8::readCountT, 2, biasT, l8::combinationType, l8::testResT, resT, false, true, 56, 72, true, 0, 84, 0> layer8;

	layer<dInT, interWinT, l9::dwidthT, 56, kernT, interKernT, l9::kernDepthT, 24, l9::kernWidthT, 1, l9::bufWidthT, l9::perColResT, l9::numColT, 72, l9::strideT, 1,
	l9::inScaleT , l9::w8sScaleT, l9::combScaleT, l9::lastScaleT, l9::readCountT, 0, biasT, l9::combinationType, l9::testResT, resT, false, false, 56, 24, false, 0, 134, 128> layer9;

	layer<dInT, interWinT, l10::dwidthT, 56, kernT, interKernT, l10::kernDepthT, 72, l10::kernWidthT, 1, l10::bufWidthT, l10::perColResT, l10::numColT, 24,  l10::strideT, 1, l10::inScaleT, l10::w8sScaleT, l10::combScaleT, l10::lastScaleT ,
	l10::readCountT, 0, biasT, l10::combinationType, l10::testResT, resT, false, true, 56, 72, false, 133, 110, 0> layer10;

	layer<dInT, interWinT, l11::dwidthT, 56, kernT, interKernT, l11::kernDepthT, 1, l11::kernWidthT, 5, l11::bufWidthT, l11::perColResT, l11::numColT, 72,  l11::strideT, 2, l11::inScaleT, l11::w8sScaleT, l11::combScaleT, l11::lastScaleT,
	l11::readCountT, 2, biasT, l11::combinationType, l11::testResT, resT, false, true, 28, 72, true, 0, 172, 0> layer11;

	layer<dInT, interWinT, l12::dwidthT, 1, kernT, interKernT, l12::kernDepthT, 24, l12::kernWidthT, 1, l12::bufWidthT, l12::perColResT, l12::numColT, 72,  l12::strideT, 1, l12::inScaleT, l12::w8sScaleT, l12::combScaleT, l12::lastScaleT,
	l12::readCountT, 0, biasT, l12::combinationType, l4::testResT, resT, false, true, 1, 24, false, 0, 1, 0> layer12;

	layer<dInT, interWinT, l13::dwidthT, 1, kernT, interKernT, l13::kernDepthT, 72, l13::kernWidthT, 1, l13::bufWidthT, l13::perColResT, l13::numColT, 24, l13::strideT, 1,
	l13::inScaleT , l13::w8sScaleT, l13::combScaleT, l13::lastScaleT, l13::readCountT, 0, biasT, l13::combinationType, l13::testResT, resT, false, true, 1, 72, false, 0, 1, 0> layer13;

	layer<dInT, interWinT, l14::dwidthT, 28, kernT, interKernT, l14::kernDepthT, 40, l14::kernWidthT, 1, l14::bufWidthT, l14::perColResT, l14::numColT, 72,  l14::strideT, 1, l14::inScaleT, l14::w8sScaleT, l14::combScaleT, l14::lastScaleT,
	l14::readCountT, 0, biasT, l14::combinationType, l14::testResT, resT, false, false, 28, 40, false, 0, 137, 138> layer14;

	layer<dInT, interWinT, l15::dwidthT, 28, kernT, interKernT, l15::kernDepthT, 120, l15::kernWidthT, 1, l15::bufWidthT, l15::perColResT, l15::numColT, 40,  l15::strideT, 1, l15::inScaleT, l15::w8sScaleT, l15::combScaleT, l15::lastScaleT,
	l15::readCountT, 0, biasT, l15::combinationType, l15::testResT, resT, false, true, 28, 120, false, 138, 115, 0> layer15;

	layer<dInT, interWinT, l16::dwidthT, 28, kernT, interKernT, l16::kernDepthT, 1, l16::kernWidthT, 5, l16::bufWidthT, l16::perColResT, l16::numColT, 120,  l16::strideT, 1,
	l16::inScaleT , l16::w8sScaleT, l16::combScaleT, l16::lastScaleT, l16::readCountT, 2, biasT, l16::combinationType, l16::testResT, resT, false, true, 28, 120, true, 0, 112, 0> layer16;

	layer<dInT, interWinT, l17::dwidthT, 1, kernT, interKernT, l17::kernDepthT, 32, l17::kernWidthT, 1, l17::bufWidthT, l17::perColResT, l17::numColT, 120,  l17::strideT, 1, l17::inScaleT, l17::w8sScaleT, l17::combScaleT, l17::lastScaleT ,
	l17::readCountT, 0, biasT, l17::combinationType, l17::testResT, resT, false, true, 1, 32, false, 0, 154, 146> layer17;

	layer<dInT, interWinT, l18::dwidthT, 1, kernT, interKernT, l18::kernDepthT, 120, l18::kernWidthT, 1, l18::bufWidthT, l18::perColResT, l18::numColT, 32,  l18::strideT, 1, l18::inScaleT, l18::w8sScaleT, l18::combScaleT, l18::lastScaleT,
	l18::readCountT, 0, biasT, l18::combinationType, l18::testResT, resT, false, true, 1, 120, false, 0, 110, 146> layer18;

	layer<dInT, interWinT, l19::dwidthT, 28, kernT, interKernT, l19::kernDepthT, 40, l19::kernWidthT, 1, l19::bufWidthT, l19::perColResT, l19::numColT, 120,  l19::strideT, 1, l19::inScaleT, l19::w8sScaleT, l19::combScaleT, l19::lastScaleT,
	l19::readCountT, 0, biasT, l19::combinationType, l19::testResT, resT, false, false, 28, 40, false, 0, 129, 123> layer19;

	layer<dInT, interWinT, l20::dwidthT, 28, kernT, interKernT, l20::kernDepthT, 120, l20::kernWidthT, 1, l20::bufWidthT, l20::perColResT, l20::numColT, 40, l20::strideT, 1,
	l20::inScaleT , l20::w8sScaleT, l20::combScaleT, l20::lastScaleT, l20::readCountT, 0, biasT, l20::combinationType, l20::testResT, resT, false, true, 28, 40, false, 0, 129, 123> layer20;

	layer<dInT, interWinT, l21::dwidthT, 28, kernT, interKernT, l21::kernDepthT, 1, l21::kernWidthT, 5, l21::bufWidthT, l21::perColResT, l21::numColT, 120,  l21::strideT, 1, l21::inScaleT, l21::w8sScaleT, l21::combScaleT, l21::lastScaleT ,
	l21::readCountT, 2, biasT, l21::combinationType, l21::testResT, resT, false, true, 28, 120, true, 0, 100, 0> layer21;

	layer<dInT, interWinT, l22::dwidthT, 28, kernT, interKernT, l22::kernDepthT, 120, l22::kernWidthT, 1, l22::bufWidthT, l22::perColResT, l22::numColT, 40,  l22::strideT, 1, l22::inScaleT, l22::w8sScaleT, l22::combScaleT, l22::lastScaleT,
	l22::readCountT, 0, biasT, l22::combinationType, l22::testResT, resT, false, true, 1, 32, false, 120, 112, 0> layer22;

	layer<dInT, interWinT, l23::dwidthT, 1, kernT, interKernT, l23::kernDepthT, 120, l23::kernWidthT, 1, l23::bufWidthT, l23::perColResT, l23::numColT, 32,  l23::strideT, 1, l23::inScaleT, l23::w8sScaleT, l23::combScaleT, l23::lastScaleT,
	l23::readCountT, 0, biasT, l23::combinationType, l23::testResT, resT, false, true, 1, 120, false, 0, 123, 161> layer23;

	layer<dInT, interWinT, l24::dwidthT, 28, kernT, interKernT, l24::kernDepthT, 40, l24::kernWidthT, 1, l24::bufWidthT, l24::perColResT, l24::numColT, 120, l24::strideT, 1,
	l24::inScaleT , l24::w8sScaleT, l24::combScaleT, l24::lastScaleT, l24::readCountT, 0, biasT, l24::combinationType, l24::testResT, resT, false, false, 28, 40, false, 0, 137, 113> layer24;

	layer<dInT, interWinT, l25::dwidthT, 28, kernT, interKernT, l25::kernDepthT, 240, l25::kernWidthT, 1, l25::bufWidthT, l25::perColResT, l25::numColT, 40,  l25::strideT, 1, l25::inScaleT, l25::w8sScaleT, l25::combScaleT, l25::lastScaleT ,
	l25::readCountT, 0, biasT, l25::combinationType, l25::testResT, resT, true, false, 28, 240, false, 118, 145, 105> layer25;

	layer<dInT, interWinT, l26::dwidthT, 28, kernT, interKernT, l26::kernDepthT, 1, l26::kernWidthT, 3, l26::bufWidthT, l26::perColResT, l26::numColT, 240,  l26::strideT, 2, l26::inScaleT, l26::w8sScaleT, l26::combScaleT, l26::lastScaleT,
	l26::readCountT, 1, biasT, l26::combinationType, l26::testResT, resT, true, false, 14, 240, true, 5, 131, 144> layer26;

	layer<dInT, interWinT, l27::dwidthT, 14, kernT, interKernT, l27::kernDepthT, 80, l27::kernWidthT, 1, l27::bufWidthT, l27::perColResT, l27::numColT, 240,  l27::strideT, 1, l27::inScaleT, l27::w8sScaleT, l27::combScaleT, l27::lastScaleT,
	l27::readCountT, 0, biasT, l27::combinationType, l27::testResT, resT, false, false, 14, 80, false, 5, 131, 130> layer27;

	layer<dInT, interWinT, l28::dwidthT, 14, kernT, interKernT, l28::kernDepthT, 200, l28::kernWidthT, 1, l28::bufWidthT, l28::perColResT, l28::numColT, 80, l28::strideT, 1,
	l28::inScaleT , l28::w8sScaleT, l28::combScaleT, l28::lastScaleT, l28::readCountT, 0, biasT, l28::combinationType, l28::testResT, resT, true, false, 14, 200, false, 130, 132, 93> layer28;

	layer<dInT, interWinT, l29::dwidthT, 14, kernT, interKernT, l29::kernDepthT, 1, l29::kernWidthT, 3, l29::bufWidthT, l29::perColResT, l29::numColT, 200,  l29::strideT, 1, l29::inScaleT, l29::w8sScaleT, l29::combScaleT, l29::lastScaleT ,
	l29::readCountT, 2, biasT, l29::combinationType, l29::testResT, resT, true, false, 14, 200, true, 7, 146, 137> layer29;

	layer<dInT, interWinT, l30::dwidthT, 14, kernT, interKernT, l30::kernDepthT, 80, l30::kernWidthT, 1, l30::bufWidthT, l30::perColResT, l30::numColT, 200,  l30::strideT, 1, l30::inScaleT, l30::w8sScaleT, l30::combScaleT, l30::lastScaleT,
	l30::readCountT, 0, biasT, l30::combinationType, l30::testResT, resT, false, false, 14, 80, false, 15, 110, 147> layer30;

	layer<dInT, interWinT, l31::dwidthT, 14, kernT, interKernT, l31::kernDepthT, 184, l31::kernWidthT, 1, l31::bufWidthT, l31::perColResT, l31::numColT, 80,  l31::strideT, 1, l31::inScaleT, l31::w8sScaleT, l31::combScaleT, l31::lastScaleT,
	l31::readCountT, 0, biasT, l31::combinationType, l31::testResT, resT, true, false, 14, 184, false, 143, 114, 112> layer31;

	layer<dInT, interWinT, l32::dwidthT, 14, kernT, interKernT, l32::kernDepthT, 1, l32::kernWidthT, 3, l32::bufWidthT, l32::perColResT, l32::numColT, 184, l32::strideT, 1,
	l32::inScaleT , l32::w8sScaleT, l32::combScaleT, l32::lastScaleT, l32::readCountT, 2, biasT, l32::combinationType, l32::testResT, resT, true, false, 14, 184, true, 7, 150, 143> layer32;

	layer<dInT, interWinT, l33::dwidthT, 14, kernT, interKernT, l33::kernDepthT, 80, l33::kernWidthT, 1, l33::bufWidthT, l33::perColResT, l33::numColT, 184,  l33::strideT, 1, l33::inScaleT, l33::w8sScaleT, l33::combScaleT, l33::lastScaleT ,
	l33::readCountT, 0, biasT, l33::combinationType, l33::testResT, resT, false, false, 14, 80, false, 11, 120, 118> layer33;

	layer<dInT, interWinT, l34::dwidthT, 14, kernT, interKernT, l34::kernDepthT, 184, l34::kernWidthT, 1, l34::bufWidthT, l34::perColResT, l34::numColT, 80,  l34::strideT, 1, l34::inScaleT, l34::w8sScaleT, l34::combScaleT, l34::lastScaleT,
	l34::readCountT, 0, biasT, l34::combinationType, l34::testResT, resT, true, false, 14, 184, false, 129, 124, 127> layer34;

	layer<dInT, interWinT, l35::dwidthT, 14, kernT, interKernT, l35::kernDepthT, 1, l38::kernWidthT, 3, l35::bufWidthT, l35::perColResT, l35::numColT, 184,  l35::strideT, 1, l35::inScaleT, l35::w8sScaleT, l35::combScaleT, l35::lastScaleT,
	l35::readCountT, 2, biasT, l35::combinationType, l35::testResT, resT, true, false, 14, 184, true, 9, 103, 143> layer35;

	layer<dInT, interWinT, l36::dwidthT, 14, kernT, interKernT, l36::kernDepthT, 80, l36::kernWidthT, 1, l36::bufWidthT, l36::perColResT, l36::numColT, 184, l36::strideT, 1,
	l36::inScaleT , l36::w8sScaleT, l36::combScaleT, l36::lastScaleT, l36::readCountT, 0, biasT, l36::combinationType, l36::testResT, resT, false, false, 14, 480, false, 7, 115, 128> layer36;

	layer<dInT, interWinT, l37::dwidthT, 14, kernT, interKernT, l37::kernDepthT, 480, l37::kernWidthT, 1, l37::bufWidthT, l37::perColResT, l37::numColT, 80,  l37::strideT, 1, l37::inScaleT, l37::w8sScaleT, l37::combScaleT, l37::lastScaleT ,
	l37::readCountT, 0, biasT, l37::combinationType, l37::testResT, resT, true, false, 14, 480, false, 121, 83, 124> layer37;

	layer<dInT, interWinT, l38::dwidthT, 14, kernT, interKernT, l38::kernDepthT, 1, l38::kernWidthT, 3, l38::bufWidthT, l38::perColResT, l38::numColT, 480,  l38::strideT, 1, l38::inScaleT, l38::w8sScaleT, l38::combScaleT, l38::lastScaleT,
	l38::readCountT, 2, biasT, l38::combinationType, l38::testResT, resT, true, false, 14, 480, true, 5, 118, 107> layer38;

	layer<dInT, interWinT, l39::dwidthT, 1, kernT, interKernT, l39::kernDepthT, 120, l39::kernWidthT, 1, l39::bufWidthT, l39::perColResT, l39::numColT, 480,  l39::strideT, 1, l39::inScaleT, l39::w8sScaleT, l39::combScaleT, l39::lastScaleT,
	l39::readCountT, 0, biasT, l39::combinationType, l39::testResT, resT, false, true, 1, 120, false, 3, 151, 0> layer39;

	layer<dInT, interWinT, l40::dwidthT, 1, kernT, interKernT, l40::kernDepthT, 480, l40::kernWidthT, 1, l40::bufWidthT, l40::perColResT, l40::numColT, 120, l40::strideT, 1,
	l40::inScaleT , l40::w8sScaleT, l40::combScaleT, l40::lastScaleT, l40::readCountT, 0, biasT, l40::combinationType, l40::testResT, resT, false, true, 1, 480, false, 0, 118, 154> layer40;

	layer<dInT, interWinT, l41::dwidthT, 14, kernT, interKernT, l41::kernDepthT, 112, l41::kernWidthT, 1, l41::bufWidthT, l41::perColResT, l41::numColT, 480,  l41::strideT, 1, l41::inScaleT, l41::w8sScaleT, l41::combScaleT, l41::lastScaleT ,
	l41::readCountT, 0, biasT, l41::combinationType, l41::testResT, resT, false, false, 14, 112, false, 11, 120, 126> layer41;

	layer<dInT, interWinT, l42::dwidthT, 14, kernT, interKernT, l42::kernDepthT, 672, l42::kernWidthT, 1, l42::bufWidthT, l42::perColResT, l42::numColT, 112,  l42::strideT, 1, l42::inScaleT, l42::w8sScaleT, l42::combScaleT, l42::lastScaleT,
	l42::readCountT, 0, biasT, l42::combinationType, l42::testResT, resT, true, false, 14, 672, false, 126, 122, 104> layer42;

	layer<dInT, interWinT, l43::dwidthT, 14, kernT, interKernT, l43::kernDepthT, 1, l43::kernWidthT, 3, l43::bufWidthT, l43::perColResT, l43::numColT, 672,  l43::strideT, 1, l43::inScaleT, l43::w8sScaleT, l43::combScaleT, l43::lastScaleT,
	l43::readCountT, 2, biasT, l43::combinationType, l43::testResT, resT, true, false, 14, 672, true, 4, 131, 134> layer43;

	layer<dInT, interWinT, l44::dwidthT, 1, kernT, interKernT, l44::kernDepthT, 168, l44::kernWidthT, 1, l44::bufWidthT, l44::perColResT, l44::numColT, 672, l44::strideT, 1,
	l44::inScaleT , l44::w8sScaleT, l44::combScaleT, l44::lastScaleT, l44::readCountT, 0, biasT, l44::combinationType, l44::testResT, resT, false, true, 1, 168, false, 3, 122, 0> layer44;

	layer<dInT, interWinT, l45::dwidthT, 1, kernT, interKernT, l45::kernDepthT, 672, l45::kernWidthT, 1, l45::bufWidthT, l45::perColResT, l45::numColT, 168,  l45::strideT, 1, l45::inScaleT, l45::w8sScaleT, l45::combScaleT, l45::lastScaleT ,
	l45::readCountT, 0, biasT, l45::combinationType, l45::testResT, resT, false, true, 1, 672, false, 0, 102, 160> layer45;

	layer<dInT, interWinT, l46::dwidthT, 14, kernT, interKernT, l46::kernDepthT, 112, l46::kernWidthT, 1, l46::bufWidthT, l46::perColResT, l46::numColT, 672,  l46::strideT, 1, l46::inScaleT, l46::w8sScaleT, l46::combScaleT, l46::lastScaleT,
	l46::readCountT, 0, biasT, l46::combinationType, l46::testResT, resT, false, false, 14, 112, false, 14, 127, 139> layer46;

	layer<dInT, interWinT, l47::dwidthT, 14, kernT, interKernT, l47::kernDepthT, 672, l47::kernWidthT, 1, l47::bufWidthT, l47::perColResT, l47::numColT, 112, l47::strideT, 1, l47::inScaleT, l47::w8sScaleT, l47::combScaleT, l47::lastScaleT,
	l47::readCountT, 0, biasT, l47::combinationType, l47::testResT, resT, true, false, 14, 672, false, 133, 119, 124> layer47;

	layer<dInT, interWinT, l48::dwidthT, 14, kernT, interKernT, l48::kernDepthT, 1, l48::kernWidthT, 5, l48::bufWidthT, l48::perColResT, l48::numColT, 672, l48::strideT, 2,
	l48::inScaleT , l48::w8sScaleT, l48::combScaleT, l48::lastScaleT, l48::readCountT, 2, biasT, l48::combinationType, l48::testResT, resT, true, false, 7, 672, true, 4, 132, 9> layer48;

	layer<dInT, interWinT, l49::dwidthT, 1, kernT, interKernT, l49::kernDepthT, 168, l49::kernWidthT, 1, l49::bufWidthT, l49::perColResT, l49::numColT, 672,  l49::strideT, 1, l49::inScaleT, l49::w8sScaleT, l49::combScaleT, l49::lastScaleT ,
	l49::readCountT, 0, biasT, l49::combinationType, l49::testResT, resT, false, true, 1, 168, false, 3, 124, 0> layer49;

	layer<dInT, interWinT, l50::dwidthT, 1, kernT, interKernT, l50::kernDepthT, 672, l50::kernWidthT, 1, l50::bufWidthT, l50::perColResT, l50::numColT, 168,  l50::strideT, 1, l50::inScaleT, l50::w8sScaleT, l50::combScaleT, l50::lastScaleT,
	l50::readCountT, 0, biasT, l50::combinationType, l50::testResT, resT, false, true, 1, 672, false, 0, 88, 120> layer50;

	layer<dInT, interWinT, l51::dwidthT, 7, kernT, interKernT, l51::kernDepthT, 160, l54::kernWidthT, 1, l51::bufWidthT, l51::perColResT, l51::numColT, 672,  l51::strideT, 1, l51::inScaleT, l51::w8sScaleT, l51::combScaleT, l51::lastScaleT,
	l51::readCountT, 0, biasT, l51::combinationType, l51::testResT, resT, false, false, 7, 160, false, 8, 131, 125> layer51;

	layer<dInT, interWinT, l52::dwidthT, 7, kernT, interKernT, l52::kernDepthT, 960, l52::kernWidthT, 1, l52::bufWidthT, l52::perColResT, l52::numColT, 160, l52::strideT, 1,
	l52::inScaleT , l52::w8sScaleT, l52::combScaleT, l52::lastScaleT, l52::readCountT, 0, biasT, l52::combinationType, l52::testResT, resT, true, false, 7, 960, false, 125, 110, 107> layer52;

	layer<dInT, interWinT, l53::dwidthT, 7, kernT, interKernT, l53::kernDepthT, 1, l53::kernWidthT, 5, l53::bufWidthT, l53::perColResT, l53::numColT, 960,  l53::strideT, 1, l53::inScaleT, l53::w8sScaleT, l53::combScaleT, l53::lastScaleT ,
	l53::readCountT, 2, biasT, l53::combinationType, l53::testResT, resT, true, false, 7, 960, true, 5, 126, 113> layer53;

	layer<dInT, interWinT, l54::dwidthT, 1, kernT, interKernT, l54::kernDepthT, 240, l54::kernWidthT, 1, l54::bufWidthT, l54::perColResT, l54::numColT, 960,  l54::strideT, 1, l54::inScaleT, l54::w8sScaleT, l54::combScaleT, l54::lastScaleT,
	l54::readCountT, 0, biasT, l54::combinationType, l54::testResT, resT, false, true, 1, 240, false, 2, 119, 0> layer54;

	layer<dInT, interWinT, l55::dwidthT, 1, kernT, interKernT, l55::kernDepthT, 960, l55::kernWidthT, 1, l55::bufWidthT, l55::perColResT, l55::numColT, 240,  l55::strideT, 1, l55::inScaleT, l55::w8sScaleT, l55::combScaleT, l55::lastScaleT,
	l55::readCountT, 0, biasT, l55::combinationType, l55::testResT, resT, false, true, 1, 960, false, 0, 76, 132> layer55;

	layer<dInT, interWinT, l56::dwidthT, 7, kernT, interKernT, l56::kernDepthT, 160, l56::kernWidthT, 1, l56::bufWidthT, l56::perColResT, l56::numColT, 960, l56::strideT, 1,
	l56::inScaleT , l56::w8sScaleT, l56::combScaleT, l56::lastScaleT, l56::readCountT, 0, biasT, l56::combinationType, l56::testResT, resT, false, false, 7, 160, false, 6, 131, 124> layer56;

	layer<dInT, interWinT, l57::dwidthT, 7, kernT, interKernT, l57::kernDepthT, 960, l57::kernWidthT, 1, l57::bufWidthT, l57::perColResT, l57::numColT, 160,  l57::strideT, 1, l57::inScaleT, l57::w8sScaleT, l57::combScaleT, l57::lastScaleT ,
	l57::readCountT, 0, biasT, l57::combinationType, l57::testResT, resT, true, false, 7, 960, false, 125, 110, 142> layer57;

	layer<dInT, interWinT, l58::dwidthT, 7, kernT, interKernT, l58::kernDepthT, 1, l58::kernWidthT, 5, l58::bufWidthT, l58::perColResT, l58::numColT, 960,  l58::strideT, 1, l58::inScaleT, l58::w8sScaleT, l58::combScaleT, l58::lastScaleT,
	l58::readCountT, 2, biasT, l58::combinationType, l59::testResT, resT, true, false, 7, 960, true, 5, 131, 80> layer58;

	layer<dInT, interWinT, l59::dwidthT, 1, kernT, interKernT, l59::kernDepthT, 240, l59::kernWidthT, 1, l59::bufWidthT, l59::perColResT, l59::numColT, 960,  l59::strideT, 1, l59::inScaleT, l59::w8sScaleT, l59::combScaleT, l59::lastScaleT,
	l59::readCountT, 0, biasT, l59::combinationType, l59::testResT, resT, false, true, 1, 240, false, 1, 113, 0> layer59;

	layer<dInT, interWinT, l60::dwidthT, 1, kernT, interKernT, l60::kernDepthT, 960, l60::kernWidthT, 1, l60::bufWidthT, l60::perColResT, l60::numColT, 240, l60::strideT, 1,
	l60::inScaleT , l60::w8sScaleT, l60::combScaleT, l60::lastScaleT, l60::readCountT, 0, biasT, l60::combinationType, l60::testResT, resT, false, true, 1, 960, false, 0, 90, 126> layer60;

	layer<dInT, interWinT, l61::dwidthT, 7, kernT, interKernT, l61::kernDepthT, 160, l61::kernWidthT, 1, l61::bufWidthT, l61::perColResT, l61::numColT, 960,  l61::strideT, 1, l61::inScaleT, l61::w8sScaleT, l61::combScaleT, l61::lastScaleT ,
	l61::readCountT, 0, biasT, l61::combinationType, l61::testResT, resT, false, false, 7, 160, false, 2, 141, 128> layer61;

	layer<dInT, interWinT, l62::dwidthT, 7, kernT, interKernT, l62::kernDepthT, 960, l62::kernWidthT, 1, l62::bufWidthT, l62::perColResT, l62::numColT, 160,  l62::strideT, 1, l62::inScaleT, l62::w8sScaleT, l62::combScaleT, l62::lastScaleT ,
	l62::readCountT, 0, biasT, l62::combinationType, l62::testResT, resT, true, false, 7, 960, false, 127, 126, 129> layer62;

	layer<dInT, interWinT, l63::dwidthT, 1, kernT, interKernT, l63::kernDepthT, 1280, l63::kernWidthT, 1, l63::bufWidthT, l63::perColResT, l63::numColT, 960,  l63::strideT, 1, l63::inScaleT, l63::w8sScaleT, l63::combScaleT, l63::lastScaleT,
	l63::readCountT, 0, biasT, l63::combinationType, l63::testResT, resT, true, false, 1, 1280, false, 1, 128, 175> layer63;//bias

	layer<dInT, interWinT, l64::dwidthT, 1, kernT, interKernT, l64::kernDepthT, 1001, l64::kernWidthT, 1, l64::bufWidthT, l64::perColResT, l64::numColT, 1280,  l64::strideT, 1, l64::inScaleT, l64::w8sScaleT, l64::combScaleT, l64::lastScaleT,
	l64::readCountT, 0, biasT, l64::combinationType, l64::testResT, resT, false, false, 1, 1001, false, 22, 94, 77> layer64;

