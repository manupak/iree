// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::Rocmlir {

namespace {
#define GEN_PASS_REGISTRATION
#include "compiler/plugins/lib/rocmlir/Passes/Passes.h.inc" // IWYU pragma: export
} // namespace

void registerRocmlirIREEPasses() {
  // Generated.
  registerPasses();
}

} // namespace mlir::iree_compiler::TorchInput
