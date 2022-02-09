#ifndef BUNSEN_TENSORCHECKTYPES_H
#define BUNSEN_TENSORCHECKTYPES_H

#include "TensorCheck.h"

namespace bunsen::typing {

static const TensorCheck ByteCheck = {{at::ScalarType::Byte}};
static const TensorCheck ShortCheck = {{at::ScalarType::Short}};
static const TensorCheck IntCheck = {{at::ScalarType::Int}};
static const TensorCheck LongCheck = {{at::ScalarType::Long}};

static const TensorCheck FloatCheck = {{at::ScalarType::Float}};
static const TensorCheck DoubleCheck = {{at::ScalarType::Double}};
static const TensorCheck HalfCheck = {{at::ScalarType::Half}};

static const TensorCheck FloatLikeCheck = {{
  at::ScalarType::Float,
  at::ScalarType::Double,
  at::ScalarType::Half
}};

static const TensorCheck IntLikeCheck = {{
  at::ScalarType::Int,
  at::ScalarType::Long,
  at::ScalarType::Short,
  at::ScalarType::Byte
}};

static const TensorCheck AnyCheck = {{}, {}, true};

static inline TensorCheck MakeByteCheck() { return ByteCheck; }
static inline TensorCheck MakeShortCheck() { return ShortCheck; }
static inline TensorCheck MakeIntCheck() { return IntCheck; }
static inline TensorCheck MakeLongCheck() { return LongCheck; }

static inline TensorCheck MakeFloatCheck() { return FloatCheck; }
static inline TensorCheck MakeDoubleCheck() { return DoubleCheck; }
static inline TensorCheck MakeHalfCheck() { return HalfCheck; }

static inline TensorCheck MakeFloatLikeCheck() { return FloatLikeCheck; }
static inline TensorCheck MakeIntLikeCheck() { return IntLikeCheck; }

static inline TensorCheck MakeAnyCheck() { return AnyCheck; }

}

#endif //BUNSEN_TENSORCHECKTYPES_H
