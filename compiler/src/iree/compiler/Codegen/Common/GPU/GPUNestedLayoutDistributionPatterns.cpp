// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/Common/GPU/GPUVectorDistribution.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Utils/Permutation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/Transforms/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir::iree_compiler {

using namespace mlir::iree_compiler::IREE::VectorExt;
using VectorValue = TypedValue<VectorType>;

static VectorValue getPackedVector(RewriterBase &rewriter,
                                   NestedLayoutAttr layout,
                                   VectorValue vector) {
  // Pack the vector to [subgroup, batch, outer, thread, element]
  SmallVector<int64_t> packedShape;

  ArrayRef<int64_t> subgroupShape = layout.getSubgroupTile();
  ArrayRef<int64_t> batchShape = layout.getBatchTile();
  ArrayRef<int64_t> outerShape = layout.getOuterTile();
  ArrayRef<int64_t> threadShape = layout.getThreadTile();
  ArrayRef<int64_t> elementShape = layout.getElementTile();

  packedShape.append(subgroupShape.begin(), subgroupShape.end());
  packedShape.append(batchShape.begin(), batchShape.end());
  packedShape.append(outerShape.begin(), outerShape.end());
  packedShape.append(threadShape.begin(), threadShape.end());
  packedShape.append(elementShape.begin(), elementShape.end());

  VectorType packedType =
      VectorType::get(packedShape, vector.getType().getElementType());
  return rewriter.create<vector::ShapeCastOp>(vector.getLoc(), packedType,
                                              vector);
}

/// Helper to linearize the given |ids| with maximum values given as |sizes|.
/// Gets the element ID in terms of |elementCount| and adds the element
/// |offset|. For example,
///
/// IDs = [d0, d1, d2, d3]
/// sizes = [s0, s1, s2, s3]
/// linear_index = d0 * (s1 * s2 * s3)
///              + d1 * (s2 * s3)
///              + d2 * (s3)
///              + d3
/// return element_index = linear_index * |elementCount| + |offset|;
static Value linearizeIndex(OpBuilder &builder, Value offset,
                            ArrayRef<OpFoldResult> ids, ArrayRef<int64_t> sizes,
                            int64_t elementCount) {
  SmallVector<AffineExpr> exprs(ids.size() + 1);
  bindSymbolsList(builder.getContext(), MutableArrayRef{exprs});
  AffineExpr idExpr = builder.getAffineConstantExpr(0);

  for (int i = 0, e = ids.size(); i < e; ++i) {
    if (sizes[i] > 1) {
      // Multiply by the residual threads along this dimension (which must be
      // faster changing than all previous dimensions) and add the id for this
      // dimension.
      idExpr = idExpr * builder.getAffineConstantExpr(sizes[i]) + exprs[i];
    }
  }
  idExpr = idExpr * builder.getAffineConstantExpr(elementCount);
  idExpr = idExpr + exprs.back();
  SmallVector<OpFoldResult> mapArgs(ids);
  mapArgs.push_back(offset);
  return affine::makeComposedAffineApply(
             builder, offset.getLoc(),
             AffineMap::get(0, mapArgs.size(), idExpr), mapArgs)
      .getResult();
}

/// Given a set of base transfer |indices|, |offsets| for the batch/outer
/// dimensions, and distributed warp and thread indices, computes the indices
/// of the distributed transfer operation based on the |vectorLayout|.
static SmallVector<Value> getTransferIndicesFromNestedLayout(
    OpBuilder &b, ValueRange indices, ArrayRef<int64_t> offsets,
    NestedLayoutAttr vectorLayout, AffineMap permutationMap,
    ArrayRef<Value> warpIndices, ArrayRef<Value> threadIndices) {
  auto isBroadcast = [](AffineExpr expr) {
    if (auto constExpr = dyn_cast<AffineConstantExpr>(expr))
      return constExpr.getValue() == 0;
    return false;
  };
  int64_t rank = vectorLayout.getRank();
  // Permute the batch and outer vector offsets to match the order of
  // the vector dimensions using the inverse of the batch/offset order.
  ArrayRef<int64_t> batchOffsets(offsets.begin(), rank);
  ArrayRef<int64_t> outerVectorOffsets(offsets.begin() + rank, rank);

  SmallVector<Value> slicedIndices(indices.begin(), indices.end());
  for (const auto &[i, dim] : llvm::enumerate(permutationMap.getResults())) {
    // Broadcasted dimension offsets can be used as-is; the read index is
    // invariant of the thread in such cases (and illegal for writes).
    if (isBroadcast(dim)) {
      continue;
    }
    unsigned pos = cast<AffineDimExpr>(dim).getPosition();
    SmallVector<OpFoldResult> ids = {
        warpIndices[i], b.getIndexAttr(batchOffsets[i]),
        b.getIndexAttr(outerVectorOffsets[i]), threadIndices[i]};
    // The order in which a vector dimension is "tiled" is
    // subgroups -> batches -> outer vectors -> threads -> elements
    SmallVector<int64_t> sizes = {
        vectorLayout.getSubgroupTile()[i], vectorLayout.getBatchTile()[i],
        vectorLayout.getOuterTile()[i], vectorLayout.getThreadTile()[i]};
    slicedIndices[pos] = linearizeIndex(b, indices[pos], ids, sizes,
                                        vectorLayout.getElementTile()[i]);
  }
  return slicedIndices;
}

static SmallVector<int64_t>
getElementVectorTileShape(NestedLayoutAttr vectorLayout) {
  int64_t rank = vectorLayout.getRank();
  SmallVector<int64_t> tileShape = vectorLayout.getDistributedShape();
  // We tile to a vector with BATCH, OUTER, and ELEMENT dimensions. So to access
  // the subvector only containing elements, we need indices in all BATCH and
  // OUTER (rank * 2) dimensions to have tile size 1.
  for (int i = 0, e = rank * 2; i < e; ++i) {
    tileShape[i] = 1;
  }
  return tileShape;
}

/// Computes the warp and thread indices for the given vector layout from a
/// single linearized thread ID.
static void populateWarpAndThreadIndices(RewriterBase &rewriter, Value threadId,
                                         int64_t subgroupSize,
                                         NestedLayoutAttr vectorLayout,
                                         SmallVector<Value> &warpIndices,
                                         SmallVector<Value> &threadIndices) {
  // The delinearized thread IDs are returned from outer most to inner most,
  // i.e. before applying the layout described dimensions ordering.
  int64_t rank = vectorLayout.getRank();
  SmallVector<Value> threadIds =
      vectorLayout.computeThreadIds(threadId, subgroupSize, rewriter);
  warpIndices = SmallVector<Value>(threadIds.begin(), threadIds.begin() + rank);
  threadIndices = SmallVector<Value>(threadIds.begin() + rank,
                                     threadIds.begin() + 2 * rank);
}

namespace {

/// Pattern to distribute `vector.transfer_read` ops with nested layouts.
struct DistributeTransferRead final
    : OpDistributionPattern<vector::TransferReadOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferRead(MLIRContext *context, Value threadId,
                         int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // TODO: Support masking.
    if (readOp.getMask()) {
      return rewriter.notifyMatchFailure(readOp, "unimplemented: masked read");
    }
    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[readOp.getResult()]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(readOp,
                                         "non-nested transfer_read layout");
    }

    // Guard on memrefs for distribution. In isolation this pattern is agnostic
    // to tensors or memrefs.
    if (!isa<MemRefType>(readOp.getSource().getType())) {
      return rewriter.notifyMatchFailure(readOp,
                                         "distribution expects memrefs");
    }

    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(vectorLayout);
    int64_t rank = vectorLayout.getRank();

    Type elementType = readOp.getSource().getType().getElementType();
    auto vectorType = VectorType::get(distShape, elementType);
    // The shape of the vector we read is pre-permutation. The permutation is
    // a transpose on the resulting read vector.
    auto innerVectorType =
        VectorType::get(vectorLayout.getElementTile(), elementType);

    // Initialize the full distributed vector for unrolling the batch/outer
    // vector dimensions.
    Value zero = rewriter.create<arith::ConstantOp>(
        readOp.getLoc(), vectorType, rewriter.getZeroAttr(vectorType));
    VectorValue acc = cast<VectorValue>(zero);

    SmallVector<Value> warpIndices, threadIndices;
    populateWarpAndThreadIndices(rewriter, threadId, subgroupSize, vectorLayout,
                                 warpIndices, threadIndices);

    ValueRange indices = readOp.getIndices();
    SmallVector<int64_t> strides(rank, 1);
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, tileShape)) {
      SmallVector<Value> slicedIndices = getTransferIndicesFromNestedLayout(
          rewriter, indices, offsets, vectorLayout, readOp.getPermutationMap(),
          warpIndices, threadIndices);

      Value slicedRead = rewriter.create<vector::TransferReadOp>(
          readOp.getLoc(), innerVectorType, readOp.getSource(), slicedIndices,
          readOp.getPermutationMapAttr(), readOp.getPadding(), readOp.getMask(),
          readOp.getInBoundsAttr());

      acc = rewriter.create<vector::InsertStridedSliceOp>(
          readOp.getLoc(), slicedRead, acc, offsets, strides);
    }

    replaceOpWithDistributedValues(rewriter, readOp, acc);
    return success();
  }

  Value threadId;
  int64_t subgroupSize;
};

/// Pattern to distribute `vector.transfer_write` ops with nested layouts.
struct DistributeTransferWrite final
    : OpDistributionPattern<vector::TransferWriteOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeTransferWrite(MLIRContext *context, Value threadId,
                          int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    // TODO: Support masking.
    if (writeOp.getMask()) {
      return rewriter.notifyMatchFailure(writeOp,
                                         "unimplemented: masked write");
    }
    NestedLayoutAttr vectorLayout =
        dyn_cast<NestedLayoutAttr>(signature[writeOp.getVector()]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(writeOp,
                                         "non-nested transfer_write layout");
    }

    if (!isa<MemRefType>(writeOp.getSource().getType())) {
      return rewriter.notifyMatchFailure(writeOp,
                                         "distribution expects memrefs");
    }

    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    SmallVector<int64_t> tileShape = getElementVectorTileShape(vectorLayout);
    int64_t rank = vectorLayout.getRank();

    SmallVector<Value> warpIndices, threadIndices;
    populateWarpAndThreadIndices(rewriter, threadId, subgroupSize, vectorLayout,
                                 warpIndices, threadIndices);

    Value distributedVector =
        getDistributed(rewriter, writeOp.getVector(), vectorLayout);

    ValueRange indices = writeOp.getIndices();
    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, tileShape)) {
      SmallVector<Value> slicedIndices = getTransferIndicesFromNestedLayout(
          rewriter, indices, offsets, vectorLayout, writeOp.getPermutationMap(),
          warpIndices, threadIndices);

      // Extract the "element vector" from the inner most dimensions. All outer
      // dimensions are either unrolled or distributed such that this is a
      // contiguous slice.
      ArrayRef<int64_t> offsetArray(offsets);
      Value slicedVector = rewriter.create<vector::ExtractOp>(
          writeOp.getLoc(), distributedVector,
          offsetArray.take_front(rank * 2));
      rewriter.create<vector::TransferWriteOp>(
          writeOp.getLoc(), slicedVector, writeOp.getSource(), slicedIndices,
          writeOp.getPermutationMapAttr(), writeOp.getMask(),
          writeOp.getInBoundsAttr());
    }

    rewriter.eraseOp(writeOp);
    return success();
  }

  Value threadId;
  int64_t subgroupSize;
};

struct DistributeBroadcast final : OpDistributionPattern<vector::BroadcastOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp broadcastOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = broadcastOp.getLoc();
    VectorValue dstVector = broadcastOp.getVector();
    auto vectorLayout = dyn_cast<NestedLayoutAttr>(signature[dstVector]);
    if (!vectorLayout) {
      return rewriter.notifyMatchFailure(broadcastOp,
                                         "non-nested result vector layout");
    }
    SmallVector<int64_t> distShape = vectorLayout.getDistributedShape();
    Type elementType =
        llvm::cast<ShapedType>(dstVector.getType()).getElementType();
    auto vectorType = VectorType::get(distShape, elementType);

    VectorValue srcVector = dyn_cast<VectorValue>(broadcastOp.getSource());
    if (!srcVector) {
      // The way distribution currently works, there is no partial thread
      // distribution, so a scalar is available to all threads. Scalar
      // distribution is simply a broadcast from scalar to the distributed
      // result shape.
      Value source = broadcastOp.getSource();
      VectorValue accumulator =
          rewriter.create<vector::BroadcastOp>(loc, vectorType, source);
      replaceOpWithDistributedValues(rewriter, broadcastOp, accumulator);
      return success();
    }

    auto sourceLayout = dyn_cast<NestedLayoutAttr>(signature[srcVector]);
    if (!sourceLayout) {
      return rewriter.notifyMatchFailure(broadcastOp,
                                         "non-nested source vector layout");
    }

    Value accumulator = rewriter.create<arith::ConstantOp>(
        loc, vectorType, rewriter.getZeroAttr(vectorType));

    int64_t rank = vectorLayout.getRank();
    // We unroll along both the batch and outer dimensions for a similar reason
    // to the transfer ops. `vector.broadcast` can only broadcast along outer
    // dims, so mixing broadcasted and un-broadcasted element/outer dims can't
    // be represented with a single `vector.broadcast`.
    SmallVector<int64_t> resultVectorUnrollShape =
        getElementVectorTileShape(vectorLayout);

    Value distributedSource = getDistributed(rewriter, srcVector, sourceLayout);

    VectorType broadcastTargetType =
        VectorType::get(vectorLayout.getElementTile(), elementType);

    int64_t sourceRank = sourceLayout.getRank();

    for (SmallVector<int64_t> offsets :
         StaticTileOffsetRange(distShape, resultVectorUnrollShape)) {
      ArrayRef<int64_t> offsetsRef(offsets);

      // Slice out the last |sourceRank| dimensions which is the inner
      // broadcasted shape.
      ArrayRef<int64_t> batchSourceOffsets =
          offsetsRef.slice(rank - sourceRank, sourceRank);
      ArrayRef<int64_t> outerSourceOffsets =
          offsetsRef.slice(2 * rank - sourceRank, sourceRank);

      // Construct the list of source offsets based on the batch/outer order of
      // the broadcasted vector. This is because we need to compute the offsets
      // into the distributed source vector with the distributed permutation.
      SmallVector<int64_t> sourceOffsets;
      sourceOffsets.append(batchSourceOffsets.begin(),
                           batchSourceOffsets.end());
      sourceOffsets.append(outerSourceOffsets.begin(),
                           outerSourceOffsets.end());

      // Extract a slice of the input to be broadcasted.
      Value slice = rewriter.create<vector::ExtractOp>(loc, distributedSource,
                                                       sourceOffsets);
      // TODO: Support non-trivial element orders.
      if (vector::isBroadcastableTo(slice.getType(), broadcastTargetType) !=
          vector::BroadcastableToResult::Success) {
        return rewriter.notifyMatchFailure(
            broadcastOp,
            "unimplemented: non-trivial broadcast source element order");
      }
      Value broadcastedSlice =
          rewriter.create<vector::BroadcastOp>(loc, broadcastTargetType, slice);
      // Insert into the broadcasted destination vector.
      accumulator = rewriter.create<vector::InsertOp>(
          loc, broadcastedSlice, accumulator, offsetsRef.take_front(rank * 2));
    }

    replaceOpWithDistributedValues(rewriter, broadcastOp, accumulator);
    return success();
  }
};

static int64_t getShuffleOffset(NestedLayoutAttr layout, int64_t dim) {
  return layout.getThreadStrides()[dim];
}

static int64_t getShuffleWidth(NestedLayoutAttr layout, int64_t dim) {
  return layout.getThreadTile()[dim];
}

/// The lowering for multi_reduction is done in two steps:
///   1. Local Reduce: Each thread reduces all elements carried by it along
///      the reduction dimensions. This is the batch, outer and element dims.
///   2. Thread Reduce: Each thread reduces result of step 1 across threads
///      by doing a butterfly shuffle.
///   3. Accumulator Reduce: Each thread reduces it's intermediate reduced
///      results with the accumulator it holds.
/// Currently, reduction across warps is not supported, but it would just add
/// another step, Warp Reduce, where threads do an atomic addition on a buffer.
struct DistributeMultiReduction final
    : OpDistributionPattern<vector::MultiDimReductionOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeMultiReduction(MLIRContext *context, int64_t subgroupSize,
                           int64_t maxBitsPerShuffle, int64_t benefit = 1)
      : OpDistributionPattern(context, benefit), subgroupSize(subgroupSize),
        maxBitsPerShuffle(maxBitsPerShuffle) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp multiReduceOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue srcVector = multiReduceOp.getSource();
    auto accVector = dyn_cast<VectorValue>(multiReduceOp.getAcc());
    if (!accVector) {
      return rewriter.notifyMatchFailure(
          multiReduceOp, "unimplemented: scalar accumulator distribution");
    }
    auto resVector = dyn_cast<VectorValue>(multiReduceOp.getResult());
    if (!resVector) {
      return rewriter.notifyMatchFailure(
          multiReduceOp, "unimplemented: scalar result distribution");
    }

    auto srcLayout = dyn_cast_or_null<NestedLayoutAttr>(signature[srcVector]);
    if (!srcLayout) {
      return rewriter.notifyMatchFailure(multiReduceOp,
                                         "expected nested layout attr");
    }

    Type elemTy = srcVector.getType().getElementType();
    unsigned elemBitwidth = elemTy.getIntOrFloatBitWidth();
    if (elemBitwidth != maxBitsPerShuffle) {
      return rewriter.notifyMatchFailure(
          multiReduceOp, llvm::formatv("unimplemented: packed shuffle",
                                       elemBitwidth, maxBitsPerShuffle));
    }

    VectorValue disSrc =
        getDistributed(rewriter, srcVector, signature[srcVector]);
    VectorValue disAcc =
        getDistributed(rewriter, accVector, signature[accVector]);

    Location loc = multiReduceOp.getLoc();

    SmallVector<bool> reducedDims = multiReduceOp.getReductionMask();
    int64_t rank = srcVector.getType().getRank();

    // Do thread local reduce.

    // The distributed reduction mask is simply the same mask appended
    // thrice.
    SmallVector<bool> distributedReductionMask;
    distributedReductionMask.reserve(3 * rank);
    for (int i = 0; i < 3; ++i) {
      distributedReductionMask.append(reducedDims.begin(), reducedDims.end());
    }
    Value localInit = getCombiningIdentityValue(
        loc, rewriter, multiReduceOp.getKind(), disAcc.getType());
    auto localReduction = rewriter.create<vector::MultiDimReductionOp>(
        loc, disSrc, localInit, distributedReductionMask,
        multiReduceOp.getKind());
    auto locallyReduced = dyn_cast<VectorValue>(localReduction.getResult());

    assert(locallyReduced && "result should have been a vector");

    // Flatten the locally reduced value.
    VectorType shaped = locallyReduced.getType();
    int64_t numElements = shaped.getNumElements();
    SmallVector<int64_t> flatShape(1, numElements);
    VectorType flatVecType = VectorType::get(flatShape, elemTy);
    VectorValue flat =
        rewriter.create<vector::ShapeCastOp>(loc, flatVecType, locallyReduced);

    // Do inter-thread/warp reduce.
    FailureOr<VectorValue> threadReduced = doThreadReduction(
        rewriter, srcLayout, flat, multiReduceOp.getKind(), reducedDims);
    if (failed(threadReduced)) {
      return failure();
    }

    // Do reduction against accumulator, which needs to be done after thread
    // reduction.
    VectorValue unflattened = rewriter.create<vector::ShapeCastOp>(
        loc, shaped, threadReduced.value());
    Value accReduction = vector::makeArithReduction(
        rewriter, loc, multiReduceOp.getKind(), unflattened, disAcc);
    auto accReduced = dyn_cast<VectorValue>(accReduction);
    if (!accReduced) {
      return failure();
    }
    replaceOpWithDistributedValues(rewriter, multiReduceOp, accReduced);

    return failure();
  }

  FailureOr<VectorValue> doThreadReduction(RewriterBase &rewriter,
                                           NestedLayoutAttr layout,
                                           VectorValue flat,
                                           vector::CombiningKind kind,
                                           ArrayRef<bool> reductionMask) const {
    VectorType flatVecType = flat.getType();
    int64_t numElements = flatVecType.getNumElements();
    Location loc = flat.getLoc();

    auto constOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(flatVecType));
    auto res = llvm::cast<VectorValue>(constOp.getResult());

    for (unsigned i = 0; i < numElements; ++i) {
      Value extracted = rewriter.create<vector::ExtractOp>(loc, flat, i);

      // Reduce across all reduction dimensions 1-by-1.
      for (unsigned i = 0, e = reductionMask.size(); i != e; ++i) {
        if (reductionMask[i]) {
          int64_t offset = getShuffleOffset(layout, i);
          int64_t width = getShuffleWidth(layout, i);
          assert(offset <= std::numeric_limits<uint32_t>::max() &&
                 width <= std::numeric_limits<uint32_t>::max());

          extracted = rewriter.create<gpu::SubgroupReduceOp>(
              loc, extracted, combiningKindToAllReduce(kind),
              /*uniform=*/false, /*cluster_size=*/width,
              /*cluster_stride=*/offset);
        }
      }

      res = rewriter.create<vector::InsertOp>(loc, extracted, res, i);
    }

    return res;
  }

  int64_t subgroupSize;
  int64_t maxBitsPerShuffle;
};

struct DistributeTranspose final : OpDistributionPattern<vector::TransposeOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(vector::TransposeOp transposeOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    VectorValue value = transposeOp.getVector();
    VectorLayoutInterface layout = dyn_cast<NestedLayoutAttr>(signature[value]);
    if (!layout) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "layout must be NestedLayoutAttr");
    }

    /// Transpose only changes the notion of where the data carried by each
    /// thread comes from in the transposed SIMD vector. The data carried by
    /// each thread is still the same, transposed as requested by the operation.
    /// So, for distributed dimensions (thread and subgroup) transpose is a
    /// no-op.
    ///
    /// Example (indices [0-3] represent ids of the threads carrying the data):
    ///
    /// input: vector<2x4xf16>
    ///
    /// 0 0 1 1
    /// 2 2 3 3
    ///
    /// after transpose,
    ///
    /// transp: vector<4x2xf16>
    ///
    /// 0 2
    /// 0 2
    /// 1 3
    /// 1 3
    ///
    /// As it can be seen, each thread is still carrying the same data but
    /// just holds a transposed version of it.

    VectorValue input = getDistributed(rewriter, value, layout);
    // Permute batch, outer and element based on the given permutation.
    int64_t rank = value.getType().getRank();
    SmallVector<int64_t> permutation;
    for (int i = 0; i < 3; ++i) {
      for (auto it : transposeOp.getPermutation()) {
        permutation.push_back(it + (i * rank));
      }
    }
    VectorValue transposed = rewriter.create<vector::TransposeOp>(
        transposeOp.getLoc(), input, permutation);
    replaceOpWithDistributedValues(rewriter, transposeOp, transposed);
    return success();
  }
};

struct DistributeBatchOuterToLayoutConversions final
    : OpDistributionPattern<IREE::VectorExt::ToLayoutOp> {
  using OpDistributionPattern::OpDistributionPattern;

  LogicalResult matchAndRewrite(IREE::VectorExt::ToLayoutOp toLayoutOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = toLayoutOp.getLoc();
    auto input = cast<VectorValue>(toLayoutOp.getInput());
    auto output = cast<VectorValue>(toLayoutOp.getOutput());
    auto layoutA = dyn_cast<NestedLayoutAttr>(signature[input]);
    auto layoutB = dyn_cast<NestedLayoutAttr>(signature[output]);

    if (!layoutA || !layoutB) {
      return rewriter.notifyMatchFailure(toLayoutOp, "non-nested layout");
    }

    // Check if everything other than batch and outer tile matches.
    if (layoutA.getSubgroupTile() != layoutB.getSubgroupTile()) {
      return failure();
    }
    if (layoutA.getSubgroupStrides() != layoutB.getSubgroupStrides()) {
      return failure();
    }
    if (layoutA.getThreadTile() != layoutB.getThreadTile()) {
      return failure();
    }
    if (layoutA.getThreadStrides() != layoutB.getThreadStrides()) {
      return failure();
    }
    if (layoutA.getElementTile() != layoutB.getElementTile()) {
      return failure();
    }

    auto batchTileA = SmallVector<int64_t>(layoutA.getBatchTile());
    auto outerTileA = SmallVector<int64_t>(layoutA.getOuterTile());
    auto batchTileB = SmallVector<int64_t>(layoutB.getBatchTile());
    auto outerTileB = SmallVector<int64_t>(layoutB.getOuterTile());

    // Check if there is a batch/outer tile mismatch.
    if (batchTileA == batchTileB && outerTileA == outerTileB) {
      return rewriter.notifyMatchFailure(toLayoutOp,
                                         "trivial layout conversion");
    }

    SmallVector<int64_t> shapeA = layoutA.getDistributedShape();
    SmallVector<int64_t> shapeB = layoutB.getDistributedShape();
    int64_t rank = layoutA.getRank();

    // Interleave batch and outer dims by transposing.

    // Build a permutation for interleaving.
    auto interleavePermutation =
        llvm::to_vector(llvm::seq<int64_t>(shapeA.size()));
    for (int i = 0; i < rank; ++i) {
      // Batch tile : [0...rank]
      // OuterTile : [rank+1...2*rank]
      // Interleave : [batch0, outer0, batch1, outer1,...]
      interleavePermutation[2 * i] = i;
      interleavePermutation[2 * i + 1] = i + rank;
    }

    auto interleaved = rewriter.create<vector::TransposeOp>(
        loc, getDistributed(rewriter, input, layoutA), interleavePermutation);

    // Shape cast to match the new layout.

    SmallVector<int64_t> transposedShapeB(shapeB);
    applyPermutationToVector(transposedShapeB, interleavePermutation);
    Type reshapedType = VectorType::get(
        transposedShapeB, interleaved.getResultVectorType().getElementType());

    auto reshaped =
        rewriter.create<vector::ShapeCastOp>(loc, reshapedType, interleaved);

    // Inverse transpose to preserve original order.
    SmallVector<int64_t> invertedPermutation =
        invertPermutationVector(interleavePermutation);

    auto layouted = rewriter.create<vector::TransposeOp>(loc, reshaped,
                                                         invertedPermutation);

    replaceOpWithDistributedValues(rewriter, toLayoutOp, layouted.getResult());
    return success();
  }
};

struct DistributeDenseConstant final
    : OpDistributionPattern<arith::ConstantOp> {
  using OpDistributionPattern::OpDistributionPattern;

  DistributeDenseConstant(MLIRContext *context, Value threadId,
                          int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {

    auto valueAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
    if (!valueAttr) {
      return failure();
    }

    auto vector = dyn_cast<VectorValue>(constantOp.getResult());
    if (!vector) {
      return failure();
    }

    auto layout = dyn_cast<NestedLayoutAttr>(signature[vector]);
    if (!layout) {
      return rewriter.notifyMatchFailure(constantOp,
                                         "non-nested transfer_read layout");
    }

    // Create an alloc for the constant.
    VectorType vectorTy = vector.getType();
    Location loc = constantOp.getLoc();
    auto addressSpaceAttr = gpu::AddressSpaceAttr::get(rewriter.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
    auto memrefType =
        MemRefType::get(vectorTy.getShape(), vectorTy.getElementType(), MemRefLayoutAttrInterface{}, addressSpaceAttr);

    // Allocate a buffer for this vector.
    auto alloc = rewriter.create<memref::AllocaOp>(loc, memrefType);

    // Zero indices.
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(rewriter.getIndexType()));
    SmallVector<Value> indices(vectorTy.getRank(), zero);

    // Write to the alloc.
    auto clonedVector = cast<arith::ConstantOp>(rewriter.clone(*constantOp));
    rewriter.create<vector::TransferWriteOp>(
        loc, clonedVector.getResult(), alloc, indices,
        SmallVector<bool>(vectorTy.getRank(), true));

    // auto memrefValueAttr =
    //     DenseElementsAttr::get(memrefType, valueAttr.reshape(memrefType));

    // // Convert the constant op to a memref constant op and transfer_read from
    // // it in the right layout.
    // valueAttr.getType();
    // auto memrefConstant =
    //     rewriter.create<arith::ConstantOp>(loc, memrefType, memrefValueAttr);

    // Create a transfer_read from this memref constant.
    auto read = rewriter.create<vector::TransferReadOp>(
        loc, vector.getType(), alloc, indices,
        SmallVector<bool>(vectorTy.getRank(), true));

    auto unitAttr = UnitAttr::get(rewriter.getContext());
    ArrayAttr readOperandsAttr = ArrayAttr::get(
        rewriter.getContext(),
        SmallVector<Attribute>(read->getNumOperands(), unitAttr));
    ArrayAttr readResultsAttr = ArrayAttr::get(rewriter.getContext(), {layout});
    setSignatureForRedistribution(rewriter, read, readOperandsAttr,
                                  readResultsAttr);

    rewriter.replaceOp(constantOp, read);
    return success();

    // // Clone the constant op.
    // auto clone = cast<arith::ConstantOp>(rewriter.clone(*constantOp));
    // auto clonedVector = cast<VectorValue>(clone.getResult());

    // int64_t rank = layout.getRank();
    // VectorValue packedVector = getPackedVector(rewriter, layout,
    // clonedVector); VectorType packedType = packedVector.getType();

    // // Transpose the packed vector from:
    // // [subgroup, batch, thread, outer, element] to
    // // [subgroup, thread, batch, outer, element].
    // SmallVector<int64_t> permutation =
    //     llvm::to_vector(llvm::seq<int64_t>(packedType.getRank()));
    // // Transpose batch and thread.
    // for (int64_t i = 0; i < rank; ++i) {
    //   std::swap(permutation[i + rank], permutation[i + rank * 2]);
    // }
    // VectorValue transposed = rewriter.create<vector::TransposeOp>(
    //     vector.getLoc(), packedVector, permutation);

    // SmallVector<Value> warpIndices, threadIndices;
    // populateWarpAndThreadIndices(rewriter, threadId, subgroupSize,
    // layout,
    //                              warpIndices, threadIndices);
    // SmallVector<Value> indices;
    // indices.append(warpIndices);
    // indices.append(threadIndices);

    // Value result = rewriter.create<vector::ExtractOp>(
    //     vector.getLoc(), transposed, indices,
    //     SmallVector<int64_t>(indices.size(), ShapedType::kDynamic));
    // replaceOpWithDistributedValues(rewriter, constantOp, result);

    return success();
  }

  Value threadId;
  int64_t subgroupSize;
};



struct DistributeStep final : OpDistributionPattern<vector::StepOp> {
  using OpDistributionPattern::OpDistributionPattern;

  VectorValue generateSlicedStep(OpBuilder& builder, Location loc, ArrayRef<int64_t> dimStrides, ArrayRef<int64_t> dimLens, ArrayRef<Value> dimIdxs, int64_t distributedLen) const {
    SmallVector<APInt> offsets;
    VectorType offsetType = VectorType::get({distributedLen}, builder.getIndexType());
    offsets.reserve(distributedLen);
    for(int64_t i=0; i<distributedLen; i++){
      int64_t offset = 0;
      for(auto[dimStride, dimLen] : zip(dimStrides, dimLens)){
        if(dimStride != 0){
          offset += (i % dimStride) + (i/dimStride)*(dimStride*dimLen); 
        }
      }
      offsets.push_back(APInt(/*width=*/64, offset));
    }
    auto constOffset = builder.create<arith::ConstantOp>(loc, DenseElementsAttr::get(offsetType, offsets));
    Value finalOffset = constOffset;
    for(auto[dimStride, dimIdx] : zip(dimStrides, dimIdxs)){
      if(dimStride != 0){
        auto strideVal = builder.create<arith::ConstantIndexOp>(loc, dimStride);
        auto dimIdxOffsetPerElem = builder.create<arith::MulIOp>(loc, strideVal, dimIdx);
        auto dimIdxOffset = builder.create<vector::BroadcastOp>(loc, offsetType, dimIdxOffsetPerElem);
        finalOffset = builder.create<arith::AddIOp>(loc, finalOffset, dimIdxOffset);
      }
    }
    return cast<VectorValue>(finalOffset);
  }

  DistributeStep(MLIRContext *context, Value threadId,
                          int64_t subgroupSize)
      : OpDistributionPattern(context), threadId(threadId),
        subgroupSize(subgroupSize) {}
  LogicalResult matchAndRewrite(vector::StepOp stepOp,
                                DistributionSignature &signature,
                                PatternRewriter &rewriter) const override {
    Location loc = stepOp.getLoc();
    VectorValue result = stepOp.getResult();
    NestedLayoutAttr resultLayout =
        dyn_cast<NestedLayoutAttr>(signature[result]);
    if (!resultLayout) {
      return rewriter.notifyMatchFailure(
          stepOp, "missing nested layout for step op result");
    }
    SmallVector<Value> subgroupIndices, threadIndices;
    populateWarpAndThreadIndices(rewriter, threadId, subgroupSize, resultLayout,
                                 subgroupIndices, threadIndices);
    ArrayRef<int64_t> subgroupStrides = resultLayout.getSubgroupStrides();
    ArrayRef<int64_t> subgroupLengths = resultLayout.getSubgroupTile();
    ArrayRef<int64_t> threadStrides = resultLayout.getThreadStrides();
    ArrayRef<int64_t> threadLengths = resultLayout.getThreadTile();
    assert(subgroupIndices.size() == 1);
    assert(threadIndices.size() == 1);
    assert(subgroupLengths.size() == 1);
    assert(threadLengths.size() == 1);
    assert(subgroupStrides.size() == 1);
    assert(threadStrides.size() == 1);
    auto distributedShape = signature[result].getDistributedShape();

    int64_t distributedElements = ShapedType::getNumElements(distributedShape);

    VectorValue slicedStepOp = generateSlicedStep(rewriter, 
                                                  loc, 
                                                  {subgroupStrides[0], threadStrides[0]}, 
                                                  {subgroupLengths[0], threadLengths[0]},
                                                  {subgroupIndices[0], threadIndices[0]},
                                                  distributedElements);
    VectorType finalSlicedStepOpType = VectorType::get({distributedShape}, result.getType().getElementType());                                              
    auto finalSlicedStepOp = rewriter.create<vector::ShapeCastOp>(loc, finalSlicedStepOpType, slicedStepOp);
    replaceOpWithDistributedValues(rewriter, stepOp, {finalSlicedStepOp});
    return success();
  }

  Value threadId;
  int64_t subgroupSize;
};

} // namespace

void populateGPUDistributeNestedLayoutAttrPatterns(RewritePatternSet &patterns,
                                                   Value threadId,
                                                   int64_t subgroupSize,
                                                   int64_t maxBitsPerShuffle) {
  patterns.add<DistributeTransferRead, DistributeTransferWrite,
               DistributeDenseConstant>(patterns.getContext(), threadId,
                                        subgroupSize);
  patterns.add<DistributeBroadcast, DistributeTranspose>(patterns.getContext());
  patterns.add<DistributeMultiReduction>(patterns.getContext(), subgroupSize,
                                         maxBitsPerShuffle);
  patterns.add<DistributeBatchOuterToLayoutConversions>(patterns.getContext());
  patterns.add<DistributeStep>(patterns.getContext(), threadId, subgroupSize);
}

}; // namespace mlir::iree_compiler
