#include "Functions.h"
#include "convol.h"
#include "scales.h"

void block1(volatile dInT *myIMG, volatile kernT *model_filters, biasInType *model_biases, volatile resT *return_addr);
void block2(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block3(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block4(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block3(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block4(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block5(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block6(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block7a(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block7b(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block8(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block9(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block10(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block11(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block12(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block13(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);
void block14(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases);

void conv(volatile dInT *myIMG, volatile kernT *model_filters, biasInType *model_biases, finalResultType *returned_values
){
#pragma HLS INTERFACE m_axi depth=200704 port=myIMG offset=slave bundle=inputs //max not calculated yet
#pragma HLS INTERFACE m_axi depth=832 port=model_filters offset=slave bundle=filters //max is 1281280 @ conv 17
#pragma HLS INTERFACE m_axi depth=12200 port=model_biases offset=slave bundle=biases
#pragma HLS INTERFACE m_axi depth=200704 port=returned_values offset=slave bundle=res //max not calculated yet
#pragma HLS INTERFACE s_axilite register port=return

	block1(myIMG, model_filters, model_biases, returned_values);
//	block2(returned_values, model_filters, model_biases);

//	block3(returned_values, model_filters, model_biases);
//	block4(returned_values, model_filters, model_biases);
//	block5(returned_values, model_filters, model_biases);
//	block6(returned_values, model_filters, model_biases);
//	block7a(returned_values, model_filters, model_biases);
//	block7b(returned_values, model_filters, model_biases);
//	block8(returned_values, model_filters, model_biases);
//	block9(returned_values, model_filters, model_biases);
//	block10(returned_values, model_filters, model_biases);
//	block11(returned_values, model_filters, model_biases);
//	block12(returned_values, model_filters, model_biases);
//	block13(returned_values, model_filters, model_biases);
//	block14(returned_values, model_filters, model_biases);

return;
}

void block1(volatile dInT *myIMG, volatile kernT *model_filters, biasInType *model_biases, volatile resT *return_addr){//L2Res and L3Res can be merged for memory savings
	//ap_fixed<25, 13> L1results[112][112][16];
	volatile resT L1results[16][112][112], L2results[16][112][112];//, L3results[16][112][112];
//	resT temp1;

	layer1.conv_stride2(myIMG, model_filters, L1results, model_biases, scale1, scale2, scale3);
	layer2.conv_stride2(&L1results[0][0][0], model_filters + 432, L2results, model_biases + 16, scale3, scale4, scale5);
	layer3.conv_stride2(&L2results[0][0][0], model_filters + 576, L2results, model_biases + 32, scale5, scale6, scale7);

	combine<ap_uint<18>, 200704, ap_uint<8>, l2::inScaleT, l3::lastScaleT, l4::inScaleT, ap_int<9> >(&L1results[0][0][0], &L2results[0][0][0], &L1results[0][0][0], scale3, scale7, scale8, 3, 118, 110);

	return_res <ap_uint<18>, 200704 >(&L1results[0][0][0], return_addr);

}

void block2(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){//L5Res and L6Res can be merged for memory savings
	volatile resT L4results[16][112][112], L5results[64][56][56], L6results[24][56][56];

	layer4.conv_stride2(inputs, model_filters + 832, L4results, model_biases + 48, scale8, scale9, scale10); //l4bias needs to be non volatile!
	layer5.conv_stride2(&L4results[0][0][0], model_filters + 1856, L5results, model_biases + 112, scale10, scale11, scale12);
	layer6.conv_stride2(&L5results[0][0][0], model_filters + 2432, L6results, model_biases + 176, scale12, scale13, scale14);
	return_res <ap_uint<17> , 75264>(&L6results[0][0][0], inputs);
}

void block3(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){//L7Res and L8Res can be merged for memory savings
	volatile resT L7results[72][56][56], L8results[72][56][56], L9results[24][56][56];
//	resT temp1[75264];
//	for(ap_uint<18> i = 0; i < 75264; i++) temp1[i] = inputs[i]; 

	layer7.conv_stride2(inputs, model_filters + 3968, L7results, model_biases + 200, scale14, scale15, scale16); //l4bias needs to be non volatile!
	layer8.conv_stride2(&L7results[0][0][0], model_filters + 5696, L8results, model_biases + 272, scale16, scale17, scale18);
	layer9.conv_stride2(&L8results[0][0][0], model_filters + 6344, L9results, model_biases + 344, scale18, scale19, scale20);

	combine<ap_uint<18>, 75264, ap_uint<8>, l9::inScaleT, l3::lastScaleT, l14::inScaleT, ap_uint<8> >(&L9results[0][0][0], inputs, &L9results[0][0][0], scale20, scale14, scale8, 128, 98, 133);

	return_res <ap_uint<17>, 75264 >(&L9results[0][0][0], inputs);
}

void block4(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){
	volatile resT L10results[72][56][56], lastResults[72][28][28], L12results[24][1][1], L13results[72][1][1], pooled[72*28*28];
//	resT temp1;

	layer10.conv_stride2(inputs, model_filters + 8072, L10results, model_biases + 368, scale21, scale22, scale23);
	layer11.conv_stride2(&L10results[0][0][0], model_filters + 9800, lastResults, model_biases + 440, scale23, scale24, scale25);
	// <sizeWT, sizeW, depthT, depth, sumT, dataT> (*inAddr, *resAddr)
	avg_pool_fn<ap_uint<10>, 784, ap_uint<7>, 72, ap_uint<14>, ap_uint<8> >(&lastResults[0][0][0], pooled);
	layer12.conv_stride2(pooled, model_filters + 11600, L12results, model_biases + 512, scale25, scale26, scale27);
	layer13.conv_stride2(&L12results[0][0][0], model_filters + 13328, L13results, model_biases + 536, scale27, scale28, scale29);

	BN<ap_uint<7>, 72, ap_uint<8>, l13::lastScaleT>(&L13results[0][0][0], scale29, 0);
	SE<ap_uint<10>, 784, ap_uint<7>, 72, ap_uint<8>, ap_ufixed<2,-6>, l13::lastScaleT, l14::inScaleT>
	(&lastResults[0][0][0], &L13results[0][0][0], 0.003921507392078638, scale25, scale30);

	return_res <ap_uint<16>, 56448 >(&lastResults[0][0][0], inputs);
}

void block5(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){
	//L15Res and L6Res cannot be merged because L15 is used for the later mult
	//L14Res and L19Res canno be merged because they are combined
	volatile resT L14results[40][28][28], L15results[120][28][28], L16results[120][28][28], pooled[32], L17results[32][1][1], L18results[120][1][1], L19results[40][28][28];
//	resT temp1;

	layer14.conv_stride2(inputs, model_filters + 15056, L14results, model_biases + 608, scale30, scale31, scale32); //l4bias needs to be non volatile!
	layer15.conv_stride2(&L14results[0][0][0], model_filters + 17936, L15results, model_biases + 648, scale32, scale33, scale34);
	layer16.conv_stride2(&L15results[0][0][0], model_filters + 22736, L16results, model_biases + 768, scale34, scale35, scale36);
	avg_pool_fn<ap_uint<10>, 784, ap_uint<7>, 120, ap_uint<14>, ap_uint<8> >(&L16results[0][0][0], pooled); // ap_uint<14> is correct if the ops are performed as uints, else I need to calculate

	layer17.conv_stride2(pooled, model_filters + 25736, L17results, model_biases + 888, scale36, scale37, scale38);
	layer18.conv_stride2(&L17results[0][0][0], model_filters + 29576, L18results, model_biases + 920, scale38, scale39, scale40);

	BN<ap_uint<7>, 120, ap_uint<8>, l18::lastScaleT>(&L18results[0][0][0], scale40, 146);
	SE<ap_uint<10>, 784, ap_uint<7>, 120, ap_uint<8>, l17::inScaleT, ap_ufixed<2,-6>, l19::inScaleT>(&L16results[0][0][0], &L18results[0][0][0], scale36, 0.003921507392078638, scale41);
	// Combine the results to make the input for the next layer .This can be its own function. The MUL

	layer19.conv_stride2(&L15results[0][0][0], model_filters + 33416, L19results, model_biases + 1040, scale41, scale42, scale43);

	combine<ap_uint<15>, 31360, ap_uint<8>, l15::inScaleT, l19::lastScaleT, l20::inScaleT , ap_uint<8> >(&L14results[0][0][0], &L19results[0][0][0], &L14results[0][0][0], scale32, scale43, scale44, 138, 123, 120);

	return_res <ap_uint<15>, 31360 >(&L14results[0][0][0], inputs);
} // Up to here I checked and the general form looks good

void block6(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){ //L20Res and L21Res can be merged for memory savings
	volatile resT L21results[120][28][28], L22results[32][1][1], lastResults[40][28][28], L23results[120][1][1], pooled[120][1][1];
//	resT temp1[37632];

//	for(ap_uint<18> i = 0; i < 94080; i++) temp1[i] = inputs[i]; // If it doesn't fit replace it with input...

	layer20.conv_stride2(inputs, model_filters + 38216, L21results, model_biases + 1080, scale44, scale45, scale46); //l4bias needs to be non volatile!
	layer21.conv_stride2(&L21results[0][0][0], model_filters + 43016, L21results, model_biases + 1200, scale46, scale47, scale48);
	avg_pool_fn<ap_uint<10>, 784, ap_uint<7>, 120, ap_uint<14>, ap_uint<8> >(&L21results[0][0][0], &pooled[0][0][0]); // ap_uint<14> is correct if the ops are performed as uints, else I need to calculate
	layer22.conv_stride2(&pooled[0][0][0], model_filters + 46016, L22results, model_biases + 1320, scale48, scale49, scale50);
	layer23.conv_stride2(&L22results[0][0][0], model_filters + 49856, L23results, model_biases + 1352, scale50, scale51, scale52);


	BN<ap_uint<7>, 120, ap_uint<8>,  l23::lastScaleT>(&L23results[0][0][0], scale25, 161);
	SE<ap_uint<10>, 784, ap_uint<7>, 120, ap_uint<8>, l21::lastScaleT, ap_ufixed<2,-6>, l24::inScaleT>(&L21results[0][0][0], &L23results[0][0][0], scale48, 0.003921507392078638, scale53);


	layer24.conv_stride2(&L21results[0][0][0], model_filters + 53696, lastResults, model_biases + 1472, scale53, scale54, scale55);
	combine<ap_uint<15>, 31360, ap_uint<8>, l20::inScaleT, l24::lastScaleT, l25::inScaleT, ap_uint<8> >(inputs, &lastResults[0][0][0], &lastResults[0][0][0], scale44, scale55, scale56, 120, 113, 118);
	return_res <ap_uint<15>, 31360 >(&lastResults[0][0][0], inputs);
}


void block7a(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){
	volatile resT L25results[240][28][28], L26results[240][14][14], L27results[80][14][14];
//	volatile resT temp1[37632];

//	for(ap_uint<16> i = 0; i < 37632; i++) temp1[i] = inputs[i];
	layer25.conv_stride2(inputs, model_filters + 58496, L25results, model_biases + 1510, scale56, scale57, scale58); //l4bias needs to be non volatile!
	layer26.conv_stride2(&L25results[0][0][0], model_filters + 68096, L26results, model_biases + 1752, scale58, scale59, scale60);
	layer27.conv_stride2(&L26results[0][0][0], model_filters + 70256, L27results, model_biases + 1992, scale60, scale61, scale62);

	return_res <ap_uint<14>, 15680 >(&L27results[0][0][0], inputs);
}

void block7b(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){
	volatile resT L28results[200][14][14], L30results[80][14][14];
//	volatile resT temp1[15680];

//	for(ap_uint<14> i = 0; i < 15680; i++) temp1[i] = inputs[i];

	layer28.conv_stride2(inputs, model_filters + 89456, L28results, model_biases + 2072, scale62, scale63, scale64);

	layer29.conv_stride2(&L28results[0][0][0], model_filters + 105456, L28results, model_biases + 2272, scale64, scale65, scale66);
	layer30.conv_stride2(&L28results[0][0][0], model_filters + 107256, L30results, model_biases + 2472, scale66, scale67, scale68);
	combine<ap_uint<14>, 15680, ap_uint<8>, l28::inScaleT, l30::lastScaleT, l31::inScaleT, ap_uint<8> >(inputs, &L30results[0][0][0], &L30results[0][0][0], scale62, scale68, scale69, 143, 147, 143);
	return_res <ap_uint<14>, 15680 >(&L30results[0][0][0], inputs);
}

void block8(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){
	volatile resT L31results[184][14][14], L32results[184][14][14], L33results[80][14][14], L34results[184][14][14], lastResults[80][14][14], pooled[80][14][14];
//	volatile resT temp1[15680];

//	for(ap_uint<14> i = 0; i < 15680; i++) temp1[i] = inputs[i];
	layer31.conv_stride2(inputs, model_filters + 123256, L31results, model_biases + 2552, scale69, scale70, scale71); //l4bias needs to be non volatile!
	layer32.conv_stride2(&L31results[0][0][0], model_filters + 138076, L32results, model_biases + 2736, scale71, scale72, scale73);
	layer33.conv_stride2(&L32results[0][0][0], model_filters + 139732, L33results, model_biases + 2920, scale73, scale74, scale75);
	combine<ap_uint<14>, 15680, ap_uint<8>, l31::inScaleT, l33::lastScaleT, l34::inScaleT, ap_uint<8> >(inputs, &L33results[0][0][0], &L33results[0][0][0], scale69, scale75, scale76, 143, 118, 129);
	layer34.conv_stride2(&L33results[0][0][0], model_filters + 154452, L34results, model_biases + 3000, scale76, scale76, scale78);
	layer35.conv_stride2(&L34results[0][0][0], model_filters + 169172, L34results, model_biases + 3184, scale78, scale79, scale80);
	layer36.conv_stride2(&L34results[0][0][0], model_filters + 170828, lastResults, model_biases + 3368, scale80, scale81, scale82);
	combine<ap_uint<14>, 15680, ap_uint<8>,l33::lastScaleT, l36::lastScaleT, l37::inScaleT, ap_uint<8> >(&L33results[0][0][0], &lastResults[0][0][0], &lastResults[0][0][0], scale75, scale82, scale83, 129, 128, 1121);
	return_res <ap_uint<14>, 15680 >(&lastResults[0][0][0], inputs);
}


void block9(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){ //L37Res and L38Res are combined, or it wouldn't fit for any reason...
	volatile resT L37results[480][14][14], L40results[480][1][1], pooled[120][1][1];
//	resT temp1;

	layer37.conv_stride2(inputs, model_filters + 185548, L37results, model_biases + 3448, scale83, scale84, scale85); //l4bias needs to be non volatile!
	layer38.conv_stride2(&L37results[0][0][0], model_filters + 223948, L37results, model_biases + 3928, scale85, scale86, scale87);
	avg_pool_fn<ap_uint<8>, 196, ap_uint<9>, 480, ap_uint<14>, ap_uint<8> >(&L37results[0][0][0], &pooled[0][0][0]); // ap_uint<14> is correct if the ops are performed as uints, else I need to calculate
	layer39.conv_stride2(&pooled[0][0][0], model_filters + 228268, pooled, model_biases + 4408, scale87, scale88, scale89);
	layer40.conv_stride2(&pooled[0][0][0], model_filters + 285868, L40results, model_biases + 4528, scale89, scale90, scale91);

	BN<ap_uint<7>, 480, ap_uint<8>, l40::lastScaleT>(&L40results[0][0][0], scale91, 154);
	SE<ap_uint<8>, 196, ap_uint<7>, 480, ap_uint<8>, l39::inScaleT, ap_ufixed<2,-6>, l41::inScaleT>(&L37results[0][0][0], &L40results[0][0][0], scale88, 0.003921507392078638, scale92);

	return_res <ap_uint<17>, 94080 >(&L37results[0][0][0], inputs);
}

void block10(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){
	volatile resT L41results[112][14][14], L42results[672][14][14], L44results[168][1][1], lastResults[112][14][14], pooled[672][1][1];

	layer41.conv_stride2(inputs, model_filters + 343468, L41results, model_biases + 5008, scale92, scale93, scale94);
	layer42.conv_stride2(&L41results[0][0][0], model_filters + 397228, L42results, model_biases + 5120, scale94, scale95, scale96); //l4bias needs to be non volatile!
	layer43.conv_stride2(&L42results[0][0][0], model_filters + 472492, L42results, model_biases + 5972, scale96, scale97, scale98);
	avg_pool_fn<ap_uint<8>, 196, ap_uint<10>, 672, ap_uint<14>, ap_uint<8> >(&L42results[0][0][0], &pooled[0][0][0]); // ap_uint<14> is correct if the ops are performed as uints, else I need to calculate
	layer44.conv_stride2(&pooled[0][0][0], model_filters + 478540, L44results, model_biases + 6464, scale98, scale99, scale100);
	layer45.conv_stride2(&L44results[0][0][0], model_filters + 591436, pooled, model_biases + 6632, scale100, scale101, scale102);

	BN<ap_uint<10>, 672, ap_uint<8>, l45::lastScaleT>(&pooled[0][0][0], scale102, 160);
	SE<ap_uint<8>, 196, ap_uint<10>, 672, ap_uint<8>, l44::inScaleT, ap_ufixed<2,-6>, l46::inScaleT>(&L42results[0][0][0], &pooled[0][0][0], scale98, 0.003921507392078638, scale103);

	layer46.conv_stride2(&L42results[0][0][0], model_filters + 704332, lastResults, model_biases + 7304, scale103, scale104, scale105);
	combine<ap_uint<15>, 21952, ap_uint<8>, l41::lastScaleT, l46::lastScaleT, l47::inScaleT, ap_uint<8> >(&L41results[0][0][0], &lastResults[0][0][0], &L41results[0][0][0], scale94, scale105, scale106, 126, 139, 133);
	return_res <ap_uint<15>, 21952 >(&L41results[0][0][0], inputs);
}

void block11(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){
	volatile resT L47results[672][14][14], L48results[672][7][7], L49results[168][1][1], lastResults[672][7][7], pooled[672][1][1];
//	resT temp1;

	layer47.conv_stride2(inputs, model_filters + 779596, L47results, model_biases + 7416, scale106, scale107, scale108); //l4bias needs to be non volatile!
	layer48.conv_stride2(&L47results[0][0][0], model_filters + 854860, L48results, model_biases + 8088, scale108, scale109, scale110);
	avg_pool_fn<ap_uint<6>, 49, ap_uint<10>, 672, ap_uint<14>, ap_uint<8> >(&L48results[0][0][0], &pooled[0][0][0]); // ap_uint<14> is correct if the ops are performed as uints, else I need to calculate
	layer49.conv_stride2(&pooled[0][0][0], model_filters + 871660, L49results, model_biases + 8760, scale110, scale111, scale112);
	layer50.conv_stride2(&L49results[0][0][0], model_filters + 984556, pooled, model_biases + 8928, scale112, scale113, scale114);

	BN<ap_uint<10>, 672, ap_uint<8>, l50::lastScaleT>(&pooled[0][0][0], scale114, 120);
	SE<ap_uint<8>, 196, ap_uint<10>, 672, ap_uint<8>, l49::inScaleT, ap_ufixed<2,-6>, l51::inScaleT>(&L48results[0][0][0], &pooled[0][0][0], scale110, 0.003921507392078638, scale115);

	return_res <ap_uint<16>, 32928 >(&L48results[0][0][0], inputs);
}

void block12(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){
	volatile resT L51results[160][7][7], L52results[960][7][7], L54results[240][1][1], lastResults[160][7][7], pooled[960][1][1];
//	volatile resT temp1;

	layer51.conv_stride2(inputs, model_filters + 1097452, L51results, model_biases + 9600, scale115, scale116, scale117);
	layer52.conv_stride2(&L51results[0][0][0], model_filters + 1204972, L52results, model_biases + 9760, scale117, scale118, scale119); //l4bias needs to be non volatile!
	layer53.conv_stride2(&L52results[0][0][0], model_filters + 1358572, L52results, model_biases + 10720, scale119, scale120, scale121);
	avg_pool_fn<ap_uint<6>,49, ap_uint<10>, 960, ap_uint<14>, ap_uint<8> >(&L52results[0][0][0], &pooled[0][0][0]); // ap_uint<14> is correct if the ops are performed as uints, else I need to calculate
	layer54.conv_stride2(&pooled[0][0][0], model_filters + 1382572, L54results, model_biases + 11680, scale121, scale122, scale123);//SE
	layer55.conv_stride2(&L54results[0][0][0], model_filters + 1612972, pooled, model_biases + 11920, scale123, scale124, scale125); // SE

	BN<ap_uint<10>, 960, ap_uint<8>, l55::lastScaleT>(&pooled[0][0][0], scale125, 132);
	SE<ap_uint<3>, 7, ap_uint<10>, 960, ap_uint<8>, l54::inScaleT, ap_ufixed<2,-6>, l56::inScaleT>(&L52results[0][0][0], &pooled[0][0][0], scale121, 0.003921507392078638, scale126);


	layer56.conv_stride2(&L52results[0][0][0], model_filters + 1843372, lastResults, model_biases + 12880, scale126, scale127, scale128);
	combine<ap_uint<13>, 7840, ap_uint<8>, l56::lastScaleT, l52::inScaleT, l57::inScaleT, ap_uint<8> >(&lastResults[0][0][0], &L51results[0][0][0], &lastResults[0][0][0], scale128, scale117, scale129, 124, 125, 125);
	return_res <ap_uint<13>, 7840 >(&lastResults[0][0][0], inputs);
}

void block13(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){
	volatile resT L57results[960][7][7], L59results[240][1][1], lastResults[160][7][7], pooled[960][1][1];
//	volatile resT temp1[7840];

//	for(ap_uint<13> i = 0; i < 7840; i++) temp1[i] = inputs[i];
	layer57.conv_stride2(inputs, model_filters + 1996972, L57results, model_biases + 1304, scale129, scale130, scale131); //l4bias needs to be non volatile!
	layer58.conv_stride2(&L57results[0][0][0], model_filters + 2150572, L57results, model_biases + 14000, scale131, scale132, scale133);
	avg_pool_fn<ap_uint<6>, 49, ap_uint<10>, 960, ap_uint<14>, ap_uint<8> >(&L57results[0][0][0], &pooled[0][0][0]); // ap_uint<14> is correct if the ops are performed as uints, else I need to calculate
	layer59.conv_stride2(&pooled[0][0][0], model_filters + 2150572, L59results, model_biases + 14960, scale133, scale134, scale135);//SE
	layer60.conv_stride2(&L59results[0][0][0], model_filters + 2404972, pooled, model_biases + 15200, scale135, scale136, scale137); // SE

	BN<ap_uint<10>, 960, ap_uint<8>, l60::lastScaleT>(&pooled[0][0][0], scale137, 126);
	SE<ap_uint<3>, 7, ap_uint<10>, 960, ap_uint<8>, l59::inScaleT, ap_ufixed<2,-6>, l61::inScaleT>(&L57results[0][0][0], &pooled[0][0][0], scale134, 0.003921507392078638, scale141);

	layer61.conv_stride2(&L57results[0][0][0], model_filters + 2635372, lastResults, model_biases + 16160, scale138, scale139, scale140);
	combine<ap_uint<13>, 7840, ap_uint<8>, l61::lastScaleT, l57::inScaleT, l62::inScaleT, ap_uint<8> >(&lastResults[0][0][0], inputs, &lastResults[0][0][0], scale140, scale129, scale141, 128, 125, 127);
	return_res <ap_uint<13>, 7840 >(&lastResults[0][0][0], inputs);
}

void block14(volatile dInT *inputs, volatile kernT *model_filters, biasInType *model_biases){
	volatile resT L62results[960][7][7], L63results[1280][1][1], L64results[1001][1][1], pooled[960][1][1]; //lastResults[72],
//	resT temp1, temp2;

	layer62.conv_stride2(inputs, model_filters + 2788972, L62results, model_biases + 16320, scale141, scale142, scale143); //l4bias needs to be non volatile!
	avg_pool_fn<ap_uint<6>, 49, ap_uint<7>, 960, ap_uint<14>, ap_uint<8> >(&L62results[0][0][0], &pooled[0][0][0]); // ap_uint<14> is correct if the ops are performed as uints, else I need to calculate
	layer63.conv_stride2(&pooled[0][0][0], model_filters + 2942572, L63results, model_biases + 17280, scale143, scale144, scale145); // SE

	avg_pool_fn<ap_uint<7>, 72, ap_uint<5>, 28, ap_uint<14>, ap_uint<8> >(&L63results[0][0][0], &L63results[0][0][0]); // ap_uint<14> is correct if the ops are performed as uints, else I need to calculate
	layer64.conv_stride2(&L63results[0][0][0], model_filters + 4171372, L64results, model_biases + 18560, scale145, scale146, scale147);

//	softmax
	return_res <ap_uint<10>, 1001 >(&L64results[0][0][0], inputs);
}
