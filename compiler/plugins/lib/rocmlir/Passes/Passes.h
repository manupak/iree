// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_TORCH_INPUTCONVERSION_PASSES_H_
#define IREE_COMPILER_PLUGINS_INPUT_TORCH_INPUTCONVERSION_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Rocmlir {

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTestPass();


//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

void registerRocmlirIREEPasses();

} // namespace mlir::iree_compiler::TorchInput

#endif // IREE_COMPILER_PLUGINS_INPUT_TORCH_INPUTCONVERSION_PASSES_H_
