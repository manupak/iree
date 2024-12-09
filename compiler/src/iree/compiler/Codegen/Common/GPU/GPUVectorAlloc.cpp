// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/Passes.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"

#define DEBUG_TYPE "iree-codegen-gpu-vector-alloc"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUVECTORALLOCPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

// For optimal performance we always want to copy 128 bits.
constexpr int copyVectorNumBits = 128;
using StrideOrder = std::pair<int64_t, int64_t>;

using StrideOrder = std::pair<int64_t, int64_t>;

/// Filter to decide which contraction ops need allocations.
static bool contractOpFilter(Operation *op) {
  auto contractOp = dyn_cast<vector::ContractionOp>(op);
  if (!contractOp) {
    return false;
  }
  SmallVector<unsigned> dims;
  for (auto [idx, type] : llvm::enumerate(contractOp.getIteratorTypesArray())) {
    if (type == vector::IteratorType::parallel) {
      dims.push_back(idx);
    }
  }
  SmallVector<int64_t> shapes;
  contractOp.getIterationBounds(shapes);
  // Don't promote vector*matrix kind of case.
  int numNonUnitParallelLoop = 0;
  for (unsigned parallelDim : dims) {
    if (shapes[parallelDim] != 1) {
      numNonUnitParallelLoop++;
    }
  }
  // TODO: Relax this constraint.
  return numNonUnitParallelLoop > 1 && dims.size() >= 2 && dims.size() <= 3;
}

static tensor::UnPackOp unPackLastLevelTiling(OpBuilder& b, Location loc, int64_t tilingDims, Value src, Value flatTensor, ArrayRef<int64_t> outerDimsPerm = {}){
    SmallVector<OpFoldResult> innerTileSizes;
    SmallVector<int64_t> innerDimsPos;
    innerTileSizes.reserve(tilingDims);

    int64_t rank = cast<TensorType>(src.getType()).getRank();
    ArrayRef<int64_t> shape = cast<TensorType>(src.getType()).getShape();
    assert(rank % tilingDims == 0);
    int64_t lastGroupIdx = (rank / tilingDims) - 1;
    for (int64_t i : llvm::seq<int64_t>(0, tilingDims)){
      int64_t tilingDim = lastGroupIdx * tilingDims + i;
      innerTileSizes.push_back(b.getIndexAttr(shape[tilingDim]));
      int64_t innerDim = ((rank / tilingDims) - 2) * tilingDims + i;
      innerDimsPos.push_back(innerDim);
    }
    llvm::errs() << "src=" << src << "\n";
    llvm::errs() << "innerTileSizes="; llvm::interleaveComma(innerTileSizes, llvm::errs()); llvm::errs() << "\n";
    llvm::errs() << "innerDimsPos="; llvm::interleaveComma(innerDimsPos, llvm::errs()); llvm::errs() << "\n";
    llvm::errs() << "outerDimsPerm="; llvm::interleaveComma(outerDimsPerm, llvm::errs()); llvm::errs() << "\n";
    auto empty = tensor::UnPackOp::createDestinationTensor(b, loc, src, innerTileSizes, innerDimsPos, outerDimsPerm);
    TensorType destType = cast<TensorType>(empty.getType());
    SmallVector<ReassociationIndices> reassociation;
    reassociation.push_back(llvm::to_vector(llvm::seq<int64_t>(0, destType.getRank())));
    auto srcReshape = b.create<tensor::ExpandShapeOp>(loc, destType, flatTensor, reassociation);
    // auto srcReshape = b.create<tensor::ReshapeOp>(loc, destType, src, destShapeVal);
    auto unpack = b.create<tensor::UnPackOp>(loc, src, srcReshape, innerDimsPos, innerTileSizes, outerDimsPerm);
    return unpack;
}

static SmallVector<int64_t> getDeinterleavedShape(ArrayRef<int64_t> threadContigousShape, int64_t preDistributedRank){
  SmallVector<int64_t> deinterleavedShape;
  deinterleavedShape.reserve(threadContigousShape.size());
  int64_t subDimGroups = threadContigousShape.size() / preDistributedRank;
  for (int64_t i : llvm::seq<int64_t>(0, preDistributedRank)){
    for (int64_t j : llvm::seq<int64_t>(0, subDimGroups)){
      deinterleavedShape.push_back(threadContigousShape[j*preDistributedRank + i]);
    }
  }
  return deinterleavedShape;
}

// [a1 x a2 x a3 x a4 x a5] x [b1 x b2 x b3 x b4 x b5] --> [a1 x b1] x [a2 x b2] x ... x [a5 x b5]
//  0    1    2     3   4      5    6     7    8   9        0     5     1     6          4      9
static SmallVector<int64_t> getInterleavingPerm(int64_t preDistributedRank){
  SmallVector<int64_t> interleavingPerm;
  for(int64_t i : llvm::seq<int64_t>(0, 5)){
    for(int64_t j : llvm::seq<int64_t>(0, preDistributedRank)){
      interleavingPerm.push_back(i + 5*j);
    }
  }
  return interleavingPerm;
}

// [a1 x b1] x [a2 x b2] x ... x [a5 x b5] --> [a1 x a2 x a3 x a4 x a5] x [b1 x b2 x b3 x b4 x b5]
//  0    1       2    3           8     9       0     2    4   6     8     1    3    5    7     9
static SmallVector<int64_t> getDeInterleavingPerm(int64_t preDistributedRank){
  SmallVector<int64_t> deinterleavingPerm;
  for(int64_t j : llvm::seq<int64_t>(0, preDistributedRank)){
    for(int64_t i : llvm::seq<int64_t>(0, 5)){
      deinterleavingPerm.push_back(j + preDistributedRank*i);
    }
  }
  return deinterleavingPerm;
}

// Allocates a tensor to copy the vector into a la bufferization.alloc_tensor.
// This allocation is always static as vectors are currently always static
// where this is used.
static FailureOr<Value> allocateTensorForVector(OpBuilder &b, Location loc,
                                                Value vector) {
  VectorType vectorType = llvm::cast<VectorType>(vector.getType());
  if (vectorType.isScalable()) {
    return failure();
  }

  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  RankedTensorType tensorType =
      RankedTensorType::get(vectorType.getShape(), vectorType.getElementType(),
                            sharedMemoryAddrSpace);
  // Vectors are always statically shaped.
  auto allocTensorOp = b.create<bufferization::AllocTensorOp>(
      loc, tensorType, ValueRange{}, Value());
  allocTensorOp.setMemorySpaceAttr(sharedMemoryAddrSpace);

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value copied = b.create<vector::TransferWriteOp>(loc, vector, allocTensorOp,
                                                   indices, inBounds)
                     .getResult();
  return copied;
}

static Value readVectorFromTensor(OpBuilder &b, VectorType vectorType,
                                  Value written) {
  Location loc = written.getLoc();
  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  auto tensorType = cast<TensorType>(written.getType());
  auto writtenVectorType = VectorType::get(tensorType.getShape(), tensorType.getElementType());

  SmallVector<Value> indices(writtenVectorType.getRank(), c0);
  SmallVector<bool> inBounds(writtenVectorType.getRank(), true);
  auto read = b.create<vector::TransferReadOp>(loc, writtenVectorType, written, indices, inBounds);
  return read;
}

static SmallVector<int64_t> getInvertPerm(ArrayRef<int64_t> perm){
  SmallVector<int64_t> ret(perm.size(), -1);
  for(auto [original, permed] : llvm::enumerate(perm)){
    ret[permed] = original;
  }
  return ret;
}

struct GPUVectorAllocPass final
    : impl::GPUVectorAllocPassBase<GPUVectorAllocPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    // Run the analysis and determine the layouts.
    LLVM_DEBUG(llvm::dbgs() << "Running Layout Analysis\n");
    VectorLayoutAnalysis analysis(funcOp);
    if (failed(analysis.run()))
      return signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "Layout Analysis Succeded\n");
    LLVM_DEBUG(llvm::dbgs() << "\n\n");

    SmallVector<IREE::VectorExt::ToLayoutOp> opsToPromote;
    funcOp.walk([&](IREE::VectorExt::ToLayoutOp op) {
      if (op.getSharedMemoryConversion()) {
        opsToPromote.push_back(op);
      }
    });

    for (IREE::VectorExt::ToLayoutOp op : opsToPromote) {
      OpBuilder builder(op);

      // HACK: Until proper barrier placement is handled later we have to
      // synchronize explicitly in this pass.

      // Synchronize before the write to shared memory to avoid stepping over
      // reads in the previous iteration of a loop. We set this barrier
      // at the start of this block.
      builder.setInsertionPointToStart(op->getBlock());
      builder.create<gpu::BarrierOp>(op->getLoc());

      // Promote both of the input operands, excluding the accumulator.
      builder.setInsertionPoint(op);
      OpOperand &operand = op.getInputMutable();

      // Maybe transpose the input; so that it stores
      // to LDS in a read friendly manner
      IREE::VectorExt::NestedLayoutAttr vectorLayout =
        dyn_cast<IREE::VectorExt::NestedLayoutAttr>(op.getLayoutAttr());
      SmallVector<int64_t> readThreadTileOrder = vectorLayout.getThreadTileOrder();

      // auto wrTransposed = builder.create<vector::TransposeOp>(op.getLoc(), op.getOperand(), readThreadTileOrder);

      // vectorLayout.getBa

      FailureOr<Value> ret =
          allocateTensorForVector(builder, op->getLoc(), op.getOperand());
      if (failed(ret)) {
        return signalPassFailure();
      }

      // Synchronize after the write to shared memory before we read from it.
      auto synced =
      builder.create<IREE::GPU::ValueBarrierOp>(op->getLoc(), *ret);
      // builder.create<gpu::BarrierOp>(op->getLoc());

      VectorType inputTy = cast<VectorType>(op.getType());
      Value read = readVectorFromTensor(builder, inputTy, synced.getResult(0));
      // auto rdTransposed = builder.create<vector::TransposeOp>(op.getLoc(), read, getInvertPerm(readThreadTileOrder));

      SmallVector<int64_t> elementTile = llvm::to_vector(vectorLayout.getElementTile());
      SmallVector<int64_t> batchTile = llvm::to_vector(vectorLayout.getBatchTile());
      int64_t& elementTileLen = elementTile.back();
      int64_t& batchTileLen = batchTile.back();
      if(elementTileLen < 8){
        elementTileLen = elementTileLen * batchTileLen;
        batchTileLen = 1;
      }
    
      auto newLayout = IREE::VectorExt::NestedLayoutAttr::get(
        op.getContext(), vectorLayout.getSubgroupTile(),  batchTile, vectorLayout.getOuterTile(),
        vectorLayout.getThreadTile(), elementTile, vectorLayout.getSubgroupStrides(), vectorLayout.getThreadStrides()
      );
      auto newLayoutOp = builder.create<IREE::VectorExt::ToLayoutOp>(op.getLoc(), read, newLayout);
      operand.set(newLayoutOp);

      // Remove the shared_memory_conversion attribute from the to_layout
      // operation.
      op.setSharedMemoryConversion(false);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
