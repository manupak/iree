// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "Passes/Passes.h"
// #include "mlir/InitRocMLIRPasses.h"
// #include "mlir/InitRocMLIRDialects.h"
// #include "torch-mlir/Conversion/Passes.h"
// #include "torch-mlir/Conversion/TorchOnnxToTorch/Passes.h"
// #include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
// #include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionDialect.h"
// #include "torch-mlir/Dialect/TorchConversion/Transforms/Passes.h"

namespace mlir::iree_compiler {

namespace {

struct RocmlirOptions {
  void bindOptions(OptionsBinder &binder) {
    // Implement me;
  }
};

// The torch plugin provides dialects, passes and opt-in options.
// Therefore, it is appropriate for default activation.
struct RocmlirSession
    : public PluginSession<RocmlirSession, RocmlirOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    mlir::iree_compiler::Rocmlir::registerRocmlirIREEPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<mlir::rock::RockDialect>();
  }
};

} // namespace

} // namespace mlir::iree_compiler

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::RocmlirOptions);

extern "C" bool iree_register_compiler_plugin_lib_rocmlir(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::RocmlirSession>("lib_rocmlir");
  return true;
}
