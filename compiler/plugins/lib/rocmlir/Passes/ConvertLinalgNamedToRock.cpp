// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/lib/rocmlir/Passes/PassDetail.h"

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"

namespace mlir::iree_compiler::Rocmlir {
      
namespace {

class FillRewritePattern : public OpRewritePattern<linalg::FillOp>{
  public:
    using OpRewritePattern<linalg::FillOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(linalg::FillOp op, PatternRewriter &rw) const final {
      Location loc = op.getLoc();
      Type outType = op.getOutputs()[0].getType();
      Value alloc = rw.create<bufferization::AllocTensorOp>(loc, outType, ValueRange{});
      rw.replaceOp(op, alloc);
      return success();
    }
};

class GemmRewritePattern : public OpRewritePattern<linalg::BatchMatmulOp>{
  public:
    using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(linalg::BatchMatmulOp op, PatternRewriter &rw) const final {
      Location loc = op.getLoc();
      Value inA = op.getInputs()[0];
      Value inB = op.getInputs()[1];
      Type outType = op.getOutputs()[0].getType();
      // Value output =
      //   rw.create<bufferization::AllocTensorOp>(loc, outType, ValueRange{});
      int64_t numCU = 110;
      auto numCUAttr = rw.getI32IntegerAttr(numCU);
      StringRef arch = "amdgcn-amd-amdhsa:gfx90a";
      rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
      rock::GemmFeatures features = archInfo.getDefaultFeatures(inA.getType());
      auto rockGemm = rw.create<rock::GemmOp>(
              loc, outType, inA, inB, op.getOutputs()[0], nullptr, nullptr, nullptr,
              rw.getStringAttr(arch), numCUAttr, rw.getAttr<rock::GemmFeaturesAttr>(features),
              rw.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set),
              /*blockSize=*/nullptr, /*gridSize=*/nullptr,
              /*params=*/nullptr);
      rw.replaceOp(op, rockGemm);
      return success();
    }
};


class ConvertLinalgNamedToRock
    : public ConvertLinalgNamedToRockBase<ConvertLinalgNamedToRock> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<mlir::rock::RockDialect>();
    registry.insert<bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext &context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);
    target.addIllegalOp<linalg::BatchMatmulOp, linalg::FillOp>();
    target.addLegalDialect<rock::RockDialect, bufferization::BufferizationDialect>();
    patterns.add<GemmRewritePattern, FillRewritePattern>(&context);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))){
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertLinalgNamedToRockPass() {
  return std::make_unique<ConvertLinalgNamedToRock>();
}

} // namespace mlir::iree_compiler::TorchInput
