#pragma once

#include <string>
#include <utility>
#include <vector>

#include <tvm/ir/module.h>
#include <tvm/tir/function.h>

#include "../support/ffi_aliases.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangRISCV {
public:
  CodeGenTileLangRISCV() = default;

  void AddFunction(const GlobalVar &gvar, const tir::PrimFunc &func);
  std::string Finish() const;
  Array<String> GetFunctionNames() const { return function_names_; }

private:
  std::vector<std::pair<std::string, tir::PrimFunc>> functions_;
  Array<String> function_names_;
};

} // namespace codegen
} // namespace tvm
