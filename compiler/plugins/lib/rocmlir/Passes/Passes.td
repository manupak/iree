// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_TORCH_INPUTCONVERSION_PASSES
#define IREE_COMPILER_PLUGINS_INPUT_TORCH_INPUTCONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TestPass :
    InterfacePass<"rocmlir-iree-test-pass", "mlir::FunctionOpInterface"> {
  let summary = "This is a rocMLIR IREE test pass.";
  let constructor = "mlir::iree_compiler::Rocmlir::createTestPass()";
}

#endif // IREE_COMPILER_PLUGINS_INPUT_TORCH_INPUTCONVERSION_PASSES
