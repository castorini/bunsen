#ifndef BUNSEN_TENSORCHECK_H
#define BUNSEN_TENSORCHECK_H

#include <optional>
#include <string>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <ATen/ATen.h>

#include "bunsen/Types.h"
#include "bunsen/Exception.h"

namespace bunsen::typing {

BUNSEN_MAKE_EXCEPTION(SizeMismatchException, "Size or dimension mismatch in tensor type check.");
BUNSEN_MAKE_EXCEPTION(DTypeMismatchException, "Scalar type mismatch in tensor type check.");

using BoundData = absl::flat_hash_map<std::string, s64>;

struct TensorShape {
  int Dim;
  std::optional<s64> Shape = std::nullopt;
  std::optional<std::string> Name = std::nullopt;
};

enum CheckStatus : int {
  Match = 0,
  SizeMismatch = 1,
  DTypeMismatch = 2
};

class TensorCheck {
private:
  std::vector<TensorShape> m_AllowedShapes;
  std::vector<at::ScalarType> m_AllowedDTypes;
  bool m_IsBlacklist;

public:
  TensorCheck(std::vector<at::ScalarType> dtypes = {}, std::vector<TensorShape> shapes = {}, bool isBlacklist = false);

  CheckStatus Check(const at::Tensor &tensor, BoundData *data = nullptr) const;

  TensorCheck WithDimShapes(const std::vector<std::tuple<int, s64>> &shapes, bool isBlacklist = false) const;
  TensorCheck WithShapes(const std::vector<s64> &shapes, bool isBlacklist = false) const;

  TensorCheck WithDimNames(const std::vector<std::tuple<int, std::string>> &names, bool isBlacklist = false) const;
  TensorCheck WithNames(const std::vector<std::string> &names, bool isBlacklist = false) const;

  friend CheckStatus Check(const at::Tensor &tensor, const std::vector<TensorCheck> &checks);
};

CheckStatus Check(const at::Tensor &tensor, const std::vector<TensorCheck> &checks = {});

}

#endif //BUNSEN_TENSORCHECK_H
