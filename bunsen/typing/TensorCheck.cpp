#ifdef BUNSEN_PYTHON_MODULE
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

#include "TensorCheck.h"

using namespace bunsen::typing;

TensorCheck::TensorCheck(std::vector<at::ScalarType> dtypes, std::vector<TensorShape> shapes, bool isBlacklist)
: m_AllowedDTypes(std::move(dtypes))
, m_AllowedShapes(std::move(shapes))
, m_IsBlacklist(isBlacklist) {}

CheckStatus TensorCheck::Check(const at::Tensor &tensor, BoundData *data) const {
#ifndef BUNSEN_PRODUCTION // Disable entirely at production time.
    int checkShapeMatches = 0;
    u32 dimMatchMask = 0; // Matches up to 32-dimensional tensors.
    checkShapeMatches += int(0 == tensor.dim()) + int(1 == tensor.dim()); // Check trivial cases.

    // Check data type.
    auto findIt = std::find(this->m_AllowedDTypes.begin(), this->m_AllowedDTypes.end(), tensor.scalar_type());

    if ((findIt == this->m_AllowedDTypes.end()) == !this->m_IsBlacklist) {
      return DTypeMismatch;
    }

    // Check shapes.
    for (const auto &shape : this->m_AllowedShapes) {
      auto dim = shape.Dim;
      s64 checkSize = 0;
      checkShapeMatches += int(dim == tensor.dim() - 1);

      if (dim >= tensor.dim()) {
        return SizeMismatch;
      }

      if (shape.Shape.has_value()) {
        checkSize = *shape.Shape;
      } else if (shape.Name.has_value() && data) {
        auto it = data->find(*shape.Name);

        if (it != data->end()) {
          checkSize = it->second;
        } else {
          checkSize = tensor.size(dim);
          (*data)[*shape.Name] = checkSize;
        }
      }

      if (checkSize < 0) {
        checkSize = tensor.size(dim);
      }

      dimMatchMask |= (1 << dim) * int(tensor.size(dim) == checkSize) * int(!this->m_IsBlacklist);
    }

    if (checkShapeMatches == 0 || dimMatchMask ^ (0xffffffff >> (32 - tensor.dim()))) {
      // We didn't match one of the dimensions' shapes.
      return SizeMismatch;
    }
#endif

    return Match;
}

CheckStatus bunsen::typing::Check(const at::Tensor &tensor, const std::vector<TensorCheck> &checks) {
  BoundData data;
  CheckStatus lastStatus = Match;

#ifndef BUNSEN_PRODUCTION // Disable entirely at production time.
  for (const auto &check : checks) {
    lastStatus = check.Check(tensor, &data);

    if (lastStatus == Match) {
      break;
    }
  }
#endif

  return lastStatus;
}

TensorCheck TensorCheck::WithDimShapes(const std::vector<std::tuple<int, s64>> &shapes, bool isBlacklist) const {
  TensorCheck newCheck = *this;
  newCheck.m_IsBlacklist = isBlacklist;

  std::transform(shapes.begin(), shapes.end(), std::back_inserter(newCheck.m_AllowedShapes), [](const auto &pair) -> TensorShape {
    const auto &[idx, shape] = pair;
    return {idx, shape};
  });

  return newCheck;
}

TensorCheck TensorCheck::WithShapes(const std::vector<s64> &shapes, bool isBlacklist) const {
  TensorCheck newCheck = *this;
  newCheck.m_IsBlacklist = isBlacklist;

  for (int i = 0; i < shapes.size(); ++i) {
    newCheck.m_AllowedShapes.push_back({i, shapes[i]});
  }

  return newCheck;
}

TensorCheck TensorCheck::WithNames(const std::vector<std::string> &names, bool isBlacklist) const {
  TensorCheck newCheck = *this;
  newCheck.m_IsBlacklist = isBlacklist;

  for (int i = 0; i < names.size(); ++i) {
    newCheck.m_AllowedShapes.push_back({i, std::nullopt, names[i]});
  }

  return newCheck;
}

TensorCheck TensorCheck::WithDimNames(const std::vector<std::tuple<int, std::string>> &names, bool isBlacklist) const {
  TensorCheck newCheck = *this;
  newCheck.m_IsBlacklist = isBlacklist;

  std::transform(names.begin(), names.end(), std::back_inserter(newCheck.m_AllowedShapes), [](const auto &pair) -> TensorShape {
    const auto &[idx, name] = pair;
    return {idx, std::nullopt, name};
  });

  return newCheck;
}
