#include "ap_fixed.h" 

#define outD 16
#define outW 112

namespace l1{
	typedef ap_uint<8> dwidthT;
	typedef ap_uint<5> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<3> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<3> numColT;
	typedef ap_uint<2> strideT;
	typedef ap_ufixed<9,-5> inScaleT;
	typedef ap_ufixed<9,-3> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,-1> lastScaleT;
	typedef ap_uint<16> readCountT;
	typedef ap_int<33> combinationType;
	typedef ap_fixed<33, 14> testResT;
}

namespace l2{
	typedef ap_uint<7> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<3> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<5> numColT;
	typedef ap_uint<1> strideT;
	typedef l1::lastScaleT inScaleT;
	typedef ap_ufixed<9,1> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<14> readCountT;
	typedef ap_int<33> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l3{
	typedef ap_uint<7> dwidthT;
	typedef ap_uint<5> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<5> numColT;
	typedef ap_uint<1> strideT;
	typedef l2::lastScaleT inScaleT;
	typedef ap_ufixed<9,-7> w8sScaleT;
	typedef ap_ufixed<9,-8> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,1> lastScaleT;
	typedef ap_uint<14> readCountT;
	typedef ap_int<33> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

namespace l4{
	typedef ap_uint<7> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<5> numColT;
	typedef ap_uint<1> strideT;
	typedef ap_ufixed<9,1> inScaleT;
	typedef ap_ufixed<9,-4> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,1> lastScaleT;
	typedef ap_uint<14> readCountT;
	typedef ap_int<33> combinationType;
	typedef ap_fixed<46, 38> testResT;
}


namespace l5{
	typedef ap_uint<7> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<3> bufWidthT;
	typedef ap_int<22> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<7> numColT;
	typedef ap_uint<2> strideT;
	typedef l4::lastScaleT inScaleT;
	typedef ap_ufixed<9,-4> w8sScaleT;
	typedef ap_ufixed<9,-4> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<14> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l6{
	typedef ap_uint<6> dwidthT;

	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<3> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<7> numColT;
	typedef ap_uint<2> strideT;
	typedef l5::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-6> combScaleT;
	typedef ap_ufixed<9,1> lastScaleT;
	typedef ap_uint<12> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l7{
	typedef ap_uint<6> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l6::lastScaleT inScaleT;
	typedef ap_ufixed<9,-7> w8sScaleT;
	typedef ap_ufixed<9,-7> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-1> lastScaleT;
	typedef ap_uint<12> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
	typedef ap_int<8> resT; //27,17
}

namespace l8{
	typedef ap_uint<6> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l7::lastScaleT inScaleT;
	typedef ap_ufixed<9,-3> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-1> lastScaleT;
	typedef ap_uint<12> readCountT;
	typedef ap_fixed<13,6> biasT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
	typedef ap_int<8> resT;
}

namespace l9{
	typedef ap_uint<6> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<7> numColT;
	typedef ap_uint<2> strideT;
	typedef l8::lastScaleT inScaleT;
	typedef ap_ufixed<9,-4> w8sScaleT;
	typedef ap_ufixed<9,-7> combScaleT;
	typedef ap_ufixed<9,1> lastScaleT;
	typedef ap_uint<13> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l10{
	typedef ap_uint<6> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l9::lastScaleT inScaleT;
	typedef ap_ufixed<9,-7> w8sScaleT;
	typedef ap_ufixed<9,-7> combScaleT;
	typedef ap_ufixed<9,-1> lastScaleT;
	typedef ap_uint<12> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l11{
	typedef ap_uint<6> dwidthT;
	typedef ap_uint<5> kernDepthT;
	typedef ap_uint<3> kernWidthT;
	typedef ap_uint<4> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<7> numColT;
	typedef ap_uint<2> strideT;
	typedef l10::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-7> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-1> lastScaleT;
	typedef ap_uint<12> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

namespace l12{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<5> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l11::lastScaleT inScaleT;
	typedef ap_ufixed<9,-10> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}


namespace l13{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<5> numColT;
	typedef ap_uint<2> strideT;
	typedef l12::lastScaleT inScaleT;
	typedef ap_ufixed<9,-9> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<13> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l14{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<6> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef ap_ufixed<9,-1> inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<10> readCountT;
	typedef ap_int<8> biasT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24

}

namespace l15{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<6> numColT;
	typedef ap_uint<1> strideT;
	typedef ap_ufixed<9, -1> inScaleT;
	typedef ap_ufixed<9,-8> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<10> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
	typedef ap_int<8> resT; //27,17
}

namespace l16{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<4> kernWidthT;
	typedef ap_uint<4> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l15::lastScaleT inScaleT;
	typedef ap_ufixed<9,-3> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<10> readCountT;
	typedef ap_fixed<13,6> biasT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
	typedef ap_int<8> resT;
}

namespace l17{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<6> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l16::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,-4> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l18{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<6> numColT;
	typedef ap_uint<1> strideT;
	typedef l17::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l19{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<6> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l18::lastScaleT inScaleT;
	typedef ap_ufixed<9,-2> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<10> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

namespace l20{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<6> numColT;
	typedef ap_uint<1> strideT;
	typedef l19::lastScaleT inScaleT;
	typedef ap_ufixed<9,-8> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<10> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}


namespace l21{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<3> kernWidthT;
	typedef ap_uint<4> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<7> numColT;
	typedef ap_uint<2> strideT;
	typedef l20::lastScaleT inScaleT;
	typedef ap_ufixed<9,-3> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<10> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l22{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<6> numColT;
	typedef ap_uint<1> strideT;
	typedef l21::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,-4> lastScaleT;
	typedef ap_uint<10> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l23{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<6> numColT;
	typedef ap_uint<1> strideT;
	typedef l22::lastScaleT inScaleT;
	typedef ap_ufixed<9,-6> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
	typedef ap_int<8> resT; //27,17
}

namespace l24{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<6> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l23::lastScaleT inScaleT;
	typedef ap_ufixed<9,-3> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<10> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}

namespace l25{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<6> numColT;
	typedef ap_uint<1> strideT;
	typedef ap_ufixed<9,1> inScaleT;
	typedef ap_ufixed<9,-8> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,2> lastScaleT;
	typedef ap_uint<10> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l26{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<4> kernWidthT;
	typedef ap_uint<3> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<8> numColT;
	typedef ap_uint<2> strideT;
	typedef l25::lastScaleT inScaleT;
	typedef ap_ufixed<9,3> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<10> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l27{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<8> numColT;
	typedef ap_uint<1> strideT;
	typedef l26::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT;
}

namespace l28{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l27::lastScaleT inScaleT;
	typedef ap_ufixed<9,-8> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
	typedef ap_int<8> resT;
}


namespace l29{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<3> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<8> numColT;
	typedef ap_uint<2> strideT;
	typedef l28::lastScaleT inScaleT;
	typedef ap_ufixed<9,-2> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,-4> lastScaleT;
	typedef ap_uint<9> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l30{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<8> numColT;
	typedef ap_uint<1> strideT;
	typedef l29::lastScaleT inScaleT;
	typedef ap_ufixed<9,-3> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24

}

namespace l31{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef ap_ufixed<9,0> inScaleT;
	typedef ap_ufixed<9,-8> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
	typedef ap_int<8> resT; //27,17
}

namespace l32{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<3> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<8> numColT;
	typedef ap_uint<2> strideT;
	typedef l31::lastScaleT inScaleT;
	typedef ap_ufixed<9,-2> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<9> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}

namespace l33{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<8> numColT;
	typedef ap_uint<2> strideT;
	typedef l32::lastScaleT inScaleT;
	typedef ap_ufixed<9,-3> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,-1> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l34{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<7> numColT;
	typedef ap_uint<2> strideT;
	typedef ap_ufixed<9,0> inScaleT;
	typedef ap_ufixed<9,-9> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l35{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<3> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<8> numColT;
	typedef ap_uint<2> strideT;
	typedef l34::lastScaleT inScaleT;
	typedef ap_ufixed<9,-1> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

namespace l36{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<8> numColT;
	typedef ap_uint<2> strideT;
	typedef l34::lastScaleT inScaleT;
	typedef ap_ufixed<9,-4> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-1> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}


namespace l37{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<9> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<7> numColT;
	typedef ap_uint<2> strideT;
	typedef ap_ufixed<9,0> inScaleT;
	typedef ap_ufixed<9,-7> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l38{
	typedef ap_uint<5> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<3> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<9> numColT;
	typedef ap_uint<2> strideT;
	typedef l37::lastScaleT inScaleT;
	typedef ap_ufixed<9,-1> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<9> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l39{
	typedef ap_uint<2> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<9> numColT;
	typedef ap_uint<2> strideT;
	typedef l38::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-4> lastScaleT;
	typedef ap_uint<2> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

namespace l40{
	typedef ap_uint<2> dwidthT;
	typedef ap_uint<9> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l39::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}

namespace l41{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<9> numColT;
	typedef ap_uint<2> strideT;
	typedef ap_ufixed<9,-3> inScaleT;
	typedef ap_ufixed<9,-4> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l42{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<10> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef l41::lastScaleT inScaleT;
	typedef ap_ufixed<9,-7> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l43{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<2> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<10> numColT;
	typedef ap_uint<1> strideT;
	typedef l42::lastScaleT inScaleT;
	typedef ap_ufixed<9,-2> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-6> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

namespace l44{
	typedef ap_uint<2> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<10> numColT;
	typedef ap_uint<1> strideT;
	typedef l43::lastScaleT inScaleT;
	typedef ap_ufixed<9,-4> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-5> lastScaleT;
	typedef ap_uint<2> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}


namespace l45{
	typedef ap_uint<2> dwidthT;
	typedef ap_uint<10> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<8> numColT;
	typedef ap_uint<2> strideT;
	typedef l43::lastScaleT inScaleT;
	typedef ap_ufixed<9,-6> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,-4> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l46{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<7> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<10> numColT;
	typedef ap_uint<1> strideT;
	typedef l45::lastScaleT inScaleT;
	typedef ap_ufixed<9,-2> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24

}

namespace l47{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<10> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<7> numColT;
	typedef ap_uint<1> strideT;
	typedef ap_ufixed<9,0> inScaleT;
	typedef ap_ufixed<9,-7> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

namespace l48{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<3> kernWidthT;
	typedef ap_uint<4> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<10> numColT;
	typedef ap_uint<2> strideT;
	typedef l47::lastScaleT inScaleT;
	typedef ap_ufixed<9,-3> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<8> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}

namespace l49{
	typedef ap_uint<2> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<10> numColT;
	typedef ap_uint<2> strideT;
	typedef l48::lastScaleT inScaleT;
	typedef ap_ufixed<9,-6> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,4> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l50{
	typedef ap_uint<2> dwidthT;
	typedef ap_uint<10> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<8> numColT;
	typedef ap_uint<2> strideT;
	typedef l49::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<2> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l51{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<10> numColT;
	typedef ap_uint<2> strideT;
	typedef ap_ufixed<9,-3> inScaleT;
	typedef ap_ufixed<9,-4> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<6> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}


namespace l52{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<10> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<8> numColT;
	typedef ap_uint<2> strideT;
	typedef l51::lastScaleT inScaleT;
	typedef ap_ufixed<9,-3> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<6> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l53{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<3> kernWidthT;
	typedef ap_uint<3> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<10> numColT;
	typedef ap_uint<2> strideT;
	typedef l52::lastScaleT inScaleT;
	typedef ap_ufixed<9,-7> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<6> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24

}

namespace l54{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<10> numColT;
	typedef ap_uint<1> strideT;
	typedef l53::lastScaleT inScaleT;
	typedef ap_ufixed<9,-1> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-1> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

namespace l55{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<10> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<8> numColT;
	typedef ap_uint<1> strideT;
	typedef l54::lastScaleT inScaleT;
	typedef ap_ufixed<9,-6> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-4> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;	typedef ap_int<8> resT;
}

namespace l56{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<10> numColT;
	typedef ap_uint<2> strideT;
	typedef l55::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<6> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l57{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<10> kernDepthT;
	typedef ap_uint<2> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<8> numColT;
	typedef ap_uint<1> strideT;
	typedef ap_ufixed<9,0> inScaleT;
	typedef ap_ufixed<9,-8> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<6> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l58{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<2> kernDepthT;
	typedef ap_uint<3> kernWidthT;
	typedef ap_uint<3> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<10> numColT;
	typedef ap_uint<1> strideT;
	typedef l57::lastScaleT inScaleT;
	typedef ap_ufixed<9,-1> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<6> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

namespace l59{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<10> numColT;
	typedef ap_uint<1> strideT;
	typedef l58::lastScaleT inScaleT;
	typedef ap_ufixed<9,-5> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-3> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}


namespace l60{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<10> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT; // Can't be global because later we have 5x5 kernels
	typedef ap_uint<8> numColT;
	typedef ap_uint<2> strideT;
	typedef l59::lastScaleT inScaleT;
	typedef ap_ufixed<9,-6> w8sScaleT;
	typedef ap_ufixed<9,-10> combScaleT;
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<33> combinationType; // Can't be global because later we have 5x5 kernels
	typedef ap_fixed<33, 14> testResT;
}

namespace l61{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<8> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<10> numColT;
	typedef ap_uint<1> strideT;
	typedef ap_ufixed<9,-1> inScaleT;
	typedef ap_ufixed<9,-3> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;
	typedef ap_ufixed<9,1> lastScaleT;
	typedef ap_uint<6> readCountT;
	typedef ap_int<21> combinationType;
	typedef ap_fixed<21, 11> testResT; //37,24
}

namespace l62{
	typedef ap_uint<4> dwidthT;
	typedef ap_uint<10> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<8> numColT;
	typedef ap_uint<1> strideT;
	typedef l2::lastScaleT inScaleT;
	typedef ap_ufixed<9,-7> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,0> lastScaleT;
	typedef ap_uint<6> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

namespace l63{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<11> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;
	typedef ap_uint<10> numColT;
	typedef ap_uint<1> strideT;
	typedef ap_ufixed<9,0> inScaleT;
	typedef ap_ufixed<9,-6> w8sScaleT;
	typedef ap_ufixed<9,-5> combScaleT;
	typedef ap_ufixed<9,-4> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<46, 38> testResT;
}

namespace l64{
	typedef ap_uint<1> dwidthT;
	typedef ap_uint<10> kernDepthT;
	typedef ap_uint<1> kernWidthT;
	typedef ap_uint<1> bufWidthT;
	typedef ap_int<21> perColResT;//37,24
	typedef ap_uint<11> numColT;
	typedef ap_uint<1> strideT;
	typedef l63::lastScaleT inScaleT;
	typedef ap_ufixed<9,-7> w8sScaleT;
	typedef ap_ufixed<9,-1> combScaleT;  // Check if I need more or less bits because this one is negative
	typedef ap_ufixed<9,-2> lastScaleT;
	typedef ap_uint<1> readCountT;
	typedef ap_int<26> combinationType;
	typedef ap_fixed<27,17> testResT; //37,24
}

typedef ap_uint<8> dInT;
typedef ap_int<9> interWinT;
typedef ap_uint<8> kernT;
typedef ap_int<9> interKernT;
typedef ap_int<32> biasT;
typedef ap_uint<8> resT;
typedef ap_int<32> biasInType;
typedef resT finalResultType;
