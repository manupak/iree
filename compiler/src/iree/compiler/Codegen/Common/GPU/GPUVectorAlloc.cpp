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

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUVECTORALLOCPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {

// For optimal performance we always want to copy 128 bits.
constexpr int copyVectorNumBits = 128;

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

// Allocates a tensor to copy the vector into a la bufferization.alloc_tensor.
// This allocation is always static as vectors are currently always static
// where this is used.
static FailureOr<memref::AllocOp> allocateTensorForVector(OpBuilder &b, Location loc,
                                                Value vector, ArrayRef<int64_t> threadContigousShape, AffineMap transposeMap) {
  VectorType vectorType = llvm::cast<VectorType>(vector.getType());
  if (vectorType.isScalable()) {
    return failure();
  }

  Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());

  MemRefType packedWriteType =
      MemRefType::get(threadContigousShape, vectorType.getElementType(), AffineMap{}, sharedMemoryAddrSpace);
  // MemRefType unpackedType =
  //     MemRefType::get(vectorType.getShape(), vectorType.getElementType(), AffineMap{}, sharedMemoryAddrSpace);
  // Vectors are always statically shaped.
  auto allocOp = b.create<memref::AllocOp>(loc, packedWriteType);
  auto transposedAllocOp = b.create<memref::TransposeOp>(loc, allocOp, AffineMapAttr::get(transposeMap));
  auto c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  // // auto viewOp = b.create<memref::ViewOp>(loc, unpackedType, transposedAllocOp, c0, ArrayRef<Value>({}));
  // SmallVector<ReassociationIndices> reassociationFlat {llvm::to_vector(llvm::seq<int64_t>(0, threadContigousShape.size()))};
  // for(ReassociationIndices reassoc : reassociationFlat){
  //   llvm::interleaveComma(reassoc, llvm::errs());
  //   llvm::errs() << ";";
  // }
  // llvm::errs() << "\n";
  // llvm::errs() << "tpose=" << transposedAllocOp << "\n";
  // auto flatOp = b.create<memref::CollapseShapeOp>(loc, transposedAllocOp, reassociationFlat);
  // SmallVector<ReassociationIndices> reassociationExpand {llvm::to_vector(llvm::seq<int64_t>(0, vectorType.getRank()))};
  // auto expandOp = b.create<memref::ExpandShapeOp>(loc, unpackedType, flatOp, reassociationExpand);

  // auto transposedAllocTensorOp = b.create<bufferization::ToTensorOp>(loc, transposedAllocOp, /*restrict=*/true, /*writable=*/true);

  //Shape cast to packed type for write
  VectorType flatType = VectorType::get({vectorType.getNumElements()}, vectorType.getElementType());
  auto flatShapeCastOp = b.create<vector::ShapeCastOp>(loc, flatType, vector);
  ArrayRef<int64_t> writePackedShape = transposedAllocOp.getType().getShape();
  VectorType writePackedType = VectorType::get(writePackedShape, vectorType.getElementType());
  auto shapeCastOp = b.create<vector::ShapeCastOp>(loc, writePackedType, flatShapeCastOp);

  SmallVector<Value> indices(writePackedType.getRank(), c0);
  SmallVector<bool> inBounds(writePackedType.getRank(), true);
  b.create<vector::TransferWriteOp>(loc, shapeCastOp, transposedAllocOp,indices, inBounds);
  return allocOp;
}

static Value readVectorFromTensor(OpBuilder &b, VectorType vectorType,
                                  memref::AllocOp alloc, AffineMap inverseTransposeMap) {
  // Location loc = tensor.getLoc();
  // TensorType allocTensorType = cast<TensorType>(tensor.getType());
  // Attribute sharedMemoryAddrSpace = gpu::AddressSpaceAttr::get(
  //     b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  // MemRefType allocMemRefType = MemRefType::get(allocTensorType.getShape(), allocTensorType.getElementType(), AffineMap{}, sharedMemoryAddrSpace);
  // auto allocMemrefOp = b.create<bufferization::ToMemrefOp>(loc, allocMemRefType, tensor);
  // auto transposedAllocOp = b.create<memref::TransposeOp>(loc, allocMemrefOp, AffineMapAttr::get(inverseTransposeMap));
  // auto transposedAllocTensorOp = b.create<bufferization::ToTensorOp>(loc, transposedAllocOp, /*restrict=*/true, /*writable=*/true);

  // TensorType transposedAllocTensorType = transposedAllocTensorOp.getType();
  // VectorType transposedAllocVectorType = VectorType::get(transposedAllocTensorType.getShape(), transposedAllocTensorType.getElementType());

  Location loc = alloc.getLoc();
  MemRefType allocType = cast<MemRefType>(alloc.getType());
  VectorType allocVectorType = VectorType::get(allocType.getShape(), allocType.getElementType());

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(allocType.getRank(), c0);
  SmallVector<bool> inBounds(allocType.getRank(), true);
  auto read = b.create<vector::TransferReadOp>(loc, allocVectorType, alloc, indices, inBounds);

  VectorType flatType = VectorType::get({vectorType.getNumElements()}, vectorType.getElementType());
  auto flatShapeCastOp = b.create<vector::ShapeCastOp>(loc, flatType, read);
  auto shapeCastOp = b.create<vector::ShapeCastOp>(loc, vectorType, flatShapeCastOp);
  return shapeCastOp;
}

struct GPUVectorAllocPass final
    : impl::GPUVectorAllocPassBase<GPUVectorAllocPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

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
      IREE::VectorExt::NestedLayoutAttr vectorLayout =
        dyn_cast<IREE::VectorExt::NestedLayoutAttr>(op.getLayoutAttr());
      if(!vectorLayout){
        return signalPassFailure();
      }

      SmallVector<StrideOrder> threadStrides;
      threadStrides.reserve(vectorLayout.getRank());

      for(auto[idx, stride] : llvm::enumerate(vectorLayout.getThreadStrides())){
        threadStrides.push_back({idx, stride});
      }
      llvm::sort(threadStrides, [](const StrideOrder& lhs, const StrideOrder& rhs){
        return lhs.second > rhs.second;
      });

      SmallVector<int64_t> packedShape = vectorLayout.getUndistributedPackedShape();
      SmallVector<int64_t> threadContigousShape = packedShape;

      int64_t threadTileOffset = 3 * vectorLayout.getRank();
      SmallVector<int64_t> threadTilePerm = llvm::to_vector(llvm::seq<int64_t>(0, vectorLayout.getRank() * 5));
      SmallVector<int64_t> inverseThreadTilePerm = llvm::to_vector(llvm::seq<int64_t>(0, vectorLayout.getRank() * 5));
      for(auto[idx, strideOrder] : llvm::enumerate(threadStrides)){
        threadContigousShape[threadTileOffset + idx] = packedShape[threadTileOffset + strideOrder.first];
        threadTilePerm[threadTileOffset + idx] = threadTileOffset + strideOrder.first;
        inverseThreadTilePerm[threadTileOffset + strideOrder.first] = threadTileOffset + idx;
      }
      AffineMap transposeMap = AffineMap::getPermutationMap(threadTilePerm, op.getContext());
      AffineMap inverseTransposeMap = AffineMap::getPermutationMap(inverseThreadTilePerm, op.getContext());

      FailureOr<memref::AllocOp> ret =
          allocateTensorForVector(builder, op->getLoc(), operand.get(), threadContigousShape, transposeMap);
      if (failed(ret)) {
        return signalPassFailure();
      }

      // Synchronize after the write to shared memory before we read from it.
      // auto synced =
      //     builder.create<IREE::GPU::ValueBarrierOp>(op->getLoc(), *ret);
      builder.create<gpu::BarrierOp>(op->getLoc());

      VectorType inputTy = cast<VectorType>(op.getType());
      Value read = readVectorFromTensor(builder, inputTy, ret.value(), inverseTransposeMap);
      operand.set(read);

      // Remove the shared_memory_conversion attribute from the to_layout
      // operation.
      op.setSharedMemoryConversion(false);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
