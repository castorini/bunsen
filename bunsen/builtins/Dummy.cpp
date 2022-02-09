#include <iostream>

#include <ATen/ATen.h>

#include "bunsen/typing/TensorCheckTypes.h"


int main() {
  // Do nothing.
  auto zeros = at::zeros({10, 10}).uniform_();
  const auto &[a, idxs] = zeros.max(1);
  std::cout << idxs[0].item<int>() << std::endl;

  auto check = bunsen::typing::IntLikeCheck.WithNames({"n", "n"});

  for (int i = 0; i < 1000000; ++i) {
    bunsen::typing::CheckStatus status = bunsen::typing::Check(idxs, {check});
    int x = int(status) + 1;
  }

  return 0;
}
