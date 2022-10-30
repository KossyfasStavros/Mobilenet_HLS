#ifndef CONVOL_H
#define CONVOL_H

#include "dTypes.h"
typedef dInT IMGtype;

void conv(volatile IMGtype *myIMG, volatile kernT *model_filters, biasInType *bodel_biases, finalResultType *returned_values);

#endif // for CONVOL_H

