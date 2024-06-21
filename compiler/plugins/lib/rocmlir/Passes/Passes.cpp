// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/lib/rocmlir/Passes/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Rock/RocMLIRConversions.h.inc"

#include "mlir/Dialect/Rock/Passes.h"

namespace mlir::iree_compiler::Rocmlir {

namespace {
#define GEN_PASS_REGISTRATION
#include "compiler/plugins/lib/rocmlir/Passes/Passes.h.inc" // IWYU pragma: export
} // namespace

void buildKernelPipeline(OpPassManager &pm) {
  // rock lowering (tuning, global to block)
  /* rocmlir-opt --rock-affix-params --rock-conv-to-gemm
   *   --rock-fold-broadcast --rock-affix-params --rock-gemm-to-gridwise
   *   --rock-regularize  --rock-gridwise-gemm-to-blockwise
   */
  auto &funcPm = pm.nest<func::FuncOp>();
  funcPm.addPass(rock::createRockAffixTuningParametersPass(
      rock::RockAffixTuningParametersPassOptions{false}));
  funcPm.addPass(rock::createRockConvToGemmPass());
  funcPm.addPass(rock::createRockGemmToGridwisePass());
  funcPm.addPass(rock::createRockRegularizePass());
  funcPm.addPass(rock::createRockGridwiseGemmToBlockwisePass());
  funcPm.addPass(rock::createRockBlockwiseGemmToThreadwisePass());

  // align linalg tiling
  /* rocmlir-opt --rock-linalg-align --canonicalize
    * --convert-linalg-to-affine-loops
    */
  funcPm.addPass(rock::createRockLinalgAlignPass());
  funcPm.addPass(rock::createRockPipelinePass());
  funcPm.addPass(createCanonicalizerPass());
  funcPm.addPass(createConvertLinalgToAffineLoopsPass());
  funcPm.addPass(rock::createRockVectorizeFusionsPass());
  // rock lowering for reductions
  /* rocmlir-opt --rock-lower-reduce
    */
  funcPm.addPass(rock::createRockLowerReducePass());

  // rock lowering (block to thread)
  /* rocmlir-opt --rock-lowering-blockwise-gemm-to-threadwise
    *   --canonicalize --rock-threadwise-gemm-lowering
    *   --rock-analyze-memory-use --rock-sugar-to-loops --rock-clean-math
    *   --math-legalize-to-f32 --rock-buffer-load-merge
    *   --rock-transform-to-memref --rock-loops-to-cf --convert-rock-to-gpu
    */
  funcPm.addPass(rock::createRockThreadwiseGemmLoweringPass());
  funcPm.addPass(rock::createRockAnalyzeMemoryUsePass());
  funcPm.addPass(rock::createRockSugarToLoopsPass());
  funcPm.addPass(rock::createRockCleanMathPass());
  funcPm.addPass(math::createMathLegalizeToF32());
  funcPm.addPass(rock::createRockBufferLoadMergePass());
  funcPm.addPass(rock::createRockTransformToMemrefPass());
  funcPm.addPass(rock::createRockLoopsToCfPass());
  // pm.addPass(createConvertRockToGPUPass());
}

void registerRocmlirIREEPasses() {
  // Generated.
  registerPasses();
}

} // namespace mlir::iree_compiler::TorchInput
