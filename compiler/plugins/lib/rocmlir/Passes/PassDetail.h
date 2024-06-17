// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_LIB_ROCMLIR_PASSES_PASSDETAIL_H_
#define IREE_COMPILER_PLUGINS_LIB_ROCMLIR_PASSES_PASSDETAIL_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Rocmlir {

#define GEN_PASS_CLASSES
#include "compiler/plugins/lib/rocmlir/Passes/Passes.h.inc"

} // namespace mlir::iree_compiler::Rocmlir

#endif // IREE_COMPILER_PLUGINS_LIB_ROCMLIR_PASSES_PASSDETAIL_H_
