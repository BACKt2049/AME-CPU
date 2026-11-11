#include "codegen_riscv.h"

#include <dmlc/memory_io.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>
#include <tvm/target/target_kind.h>

#include <fstream>
#include <algorithm>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {

class MLIRSourceModuleNode final : public ffi::ModuleObj {
public:
  explicit MLIRSourceModuleNode(const std::string &code,
                                const ffi::Array<ffi::String> &func_names)
      : code_(code), func_names_(func_names) {}

  const char *kind() const final { return "mlir"; }

  ffi::Optional<ffi::Function> GetFunction(const ffi::String &name) final {
    ObjectPtr<Object> sptr_to_self = ffi::GetObjectPtr<Object>(this);
    if (name == "get_symbol") {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any *rv) {
        *rv = this->func_names_[0];
      });
    }
    if (name == "get_func_names") {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any *rv) {
        *rv = this->func_names_;
      });
    }
    return ffi::Function(nullptr);
  }

  ffi::String InspectSource(const ffi::String &format) const final {
    ICHECK(format.empty() || format == "mlir");
    return code_;
  }

  ffi::Array<ffi::String> GetWriteFormats() const override { return {"mlir"}; }

  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    dmlc::MemoryStringStream ms(&buffer);
    dmlc::Stream *stream = &ms;
    stream->Write(code_);

    std::vector<std::string> func_names;
    for (const auto &func_name : func_names_) {
      func_names.push_back(func_name);
    }
    stream->Write(func_names);
    return ffi::Bytes(buffer);
  }

  static ffi::Module LoadFromBytes(const ffi::Bytes &bytes) {
    dmlc::MemoryFixedSizeStream ms(const_cast<char *>(bytes.data()), bytes.size());
    dmlc::Stream *stream = &ms;

    std::string code;
    ICHECK(stream->Read(&code)) << "Loading MLIR code failed";

    std::vector<std::string> tmp_func_names;
    ICHECK(stream->Read(&tmp_func_names)) << "Loading MLIR function names failed";

    ffi::Array<ffi::String> func_names;
    for (const auto &func_name : tmp_func_names) {
      func_names.push_back(ffi::String(func_name));
    }

    auto n = ffi::make_object<MLIRSourceModuleNode>(code, func_names);
    return ffi::Module(n);
  }

  void WriteToFile(const ffi::String &file_name, const ffi::String &format) const final {
    ICHECK(format.empty() || format == "mlir");
    std::ofstream os(file_name.operator std::string(), std::ios::binary);
    ICHECK(os.good()) << "Failed to open " << file_name << " for writing";
    os << code_;
  }

  int GetPropertyMask() const override { return ffi::Module::kBinarySerializable; }

  bool ImplementsFunction(const ffi::String &name) final {
    return std::find(func_names_.begin(), func_names_.end(), name) != func_names_.end();
  }

private:
  std::string code_;
  ffi::Array<ffi::String> func_names_;
};

ffi::Module MLIRSourceModuleCreate(const std::string &code,
                                   const ffi::Array<ffi::String> &func_names) {
  auto n = ffi::make_object<MLIRSourceModuleNode>(code, func_names);
  return ffi::Module(n);
}

ffi::Module BuildTileLangRISCV(IRModule mod, Target target) {
  (void)target;
  mod = tir::transform::LowerInitBlock()(mod);
  CodeGenTileLangRISCV cg;
  for (const auto &kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "CodeGenTileLangRISCV: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto func = Downcast<tir::PrimFunc>(kv.second);
    cg.AddFunction(gvar, func);
  }

  std::string code = cg.Finish();
  return MLIRSourceModuleCreate(code, cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.Module.load_from_bytes.mlir",
                        MLIRSourceModuleNode::LoadFromBytes);
  refl::GlobalDef().def("target.build.tilelang_riscv",
                        BuildTileLangRISCV);
}

TVM_REGISTER_TARGET_KIND("riscv", kDLCPU);

} // namespace codegen
} // namespace tvm
