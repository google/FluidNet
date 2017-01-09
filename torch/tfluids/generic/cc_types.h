// Copyright 2016 Google Inc, NYU.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Note: instead of calling THGenerateFloatTypes.h, we're going to hack into
// the torch build system a little bit. This makes the tfluids library
// compatible with the blaze build system (for reasons that aren't interesting,
// but are very annoying).

#ifndef SOURCE_FILE
  #error "You must define SOURCE_FILE before including cc_types.h"
#else

#include <TH.h>  // Includes THTensor and THTensor_ macros (form THTensor.h).

// Replicated THTensor macros for reference.
// #define THTensor TH_CONCAT_3(TH,Real,Tensor)
// #define THTensor_(NAME) TH_CONCAT_4(TH,Real,Tensor_,NAME)

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define tfluids_(NAME) TH_CONCAT_3(tfluids_, Real, NAME)


#define real float
#define accreal double
#define Real Float
#define THInf FLT_MAX
#define TH_REAL_IS_FLOAT
#line 1 SOURCE_FILE
#include SOURCE_FILE
#undef accreal
#undef real
#undef Real
#undef THInf
#undef TH_REAL_IS_FLOAT

#define real double
#define accreal double
#define Real Double
#define THInf DBL_MAX
#define TH_REAL_IS_DOUBLE
#line 1 SOURCE_FILE
#include SOURCE_FILE
#undef accreal
#undef real
#undef Real
#undef THInf
#undef TH_REAL_IS_DOUBLE

#endif  // #ifdef SOURCE_FILE
