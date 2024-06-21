# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(ROCMLIR_ROOT_DIR "${IREE_SOURCE_DIR}/third_party/rocmlir")
set(IREE_COMPILER_TABLEGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(rocmlir_compiler_defs INTERFACE)
target_include_directories(rocmlir_compiler_defs
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Configures all iree_cc_* targets to take this implicit dep,
# which provides common includes and copts for the tree.
set(IREE_IMPLICIT_DEFS_CC_DEPS rocmlir_compiler_defs)

list(APPEND IREE_COMPILER_TABLEGEN_INCLUDE_DIRS
    "${ROCMLIR_ROOT_DIR}/mlir/include")

iree_cc_library(
  NAME
    defs
  INCLUDES
    "${ROCMLIR_ROOT_DIR}/mlir/include"
)

###############################################################################
# Rock dialect
###############################################################################

file(GLOB _RockDialectIR_SRCS "${ROCMLIR_ROOT_DIR}/mlir/lib/Dialect/Rock/IR/*.cpp")
iree_cc_library(
  NAME
    RockDialectIR
  SRCS
    ${_RockDialectIR_SRCS}
  DEPS
    ::defs
    ::RockDialectGen
    ::RockAttrDefsIncGen
    ::RockTuningParamAttrInterfaceIncGen
    ::RockAccelTuningParamAttrInterfaceIncGen
    ::RockGemmWrapperInterfaceIncGen
    ::RockConvInterfaceIncGen
    ::RockAcceptingViewOpInterfaceIncGen
    ::RockWriterOpInterfaceIncGen
    MLIRBytecodeOpInterface
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRFuncDialect
    MLIRIR
    MLIRInferIntRangeInterface
    MLIRSupport
    MLIRControlFlowInterfaces
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
)

file(GLOB _RockDialectUtility_SRCS "${ROCMLIR_ROOT_DIR}/mlir/lib/Dialect/Rock/utility/*.cpp")
iree_cc_library(
  NAME
    RockDialectUtility
  SRCS
    ${_RockDialectUtility_SRCS}
  DEPS
    ::defs
    ::RockDialectGen
    ::RockAttrDefsIncGen
    ::RockTuningParamAttrInterfaceIncGen
    ::RockAccelTuningParamAttrInterfaceIncGen
    ::RockGemmWrapperInterfaceIncGen
    ::RockConvInterfaceIncGen
    ::RockAcceptingViewOpInterfaceIncGen
    ::RockWriterOpInterfaceIncGen
    MLIRBytecodeOpInterface
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRFuncDialect
    MLIRIR
    MLIRInferIntRangeInterface
    MLIRSupport
    MLIRControlFlowInterfaces
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
)

file(GLOB _RockDialectTuning_SRCS "${ROCMLIR_ROOT_DIR}/mlir/lib/Dialect/Rock/Tuning/*.cpp")
iree_cc_library(
  NAME
    RockDialectTuning
  SRCS
    ${_RockDialectTuning_SRCS}
  DEPS
    ::defs
    ::RockDialectGen
    ::RockAttrDefsIncGen
    ::RockTuningParamAttrInterfaceIncGen
    ::RockAccelTuningParamAttrInterfaceIncGen
    ::RockGemmWrapperInterfaceIncGen
    ::RockConvInterfaceIncGen
    ::RockAcceptingViewOpInterfaceIncGen
    ::RockWriterOpInterfaceIncGen
    ::RockPassesIncGen
    MLIRBytecodeOpInterface
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRFuncDialect
    MLIRIR
    MLIRInferIntRangeInterface
    MLIRSupport
    MLIRControlFlowInterfaces
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
)

file(GLOB _RockDialectTransforms_SRCS "${ROCMLIR_ROOT_DIR}/mlir/lib/Dialect/Rock/Transforms/*.cpp")
list(REMOVE_ITEM _RockDialectTransforms_SRCS "${ROCMLIR_ROOT_DIR}/mlir/lib/Dialect/Rock/Transforms/ViewToTransform.cpp")
list(REMOVE_ITEM _RockDialectTransforms_SRCS "${ROCMLIR_ROOT_DIR}/mlir/lib/Dialect/Rock/Transforms/CheckResidency.cpp")
iree_cc_library(
  NAME
    RockDialectTransforms
  SRCS
    ${_RockDialectTransforms_SRCS}
  DEPS
    ::defs
    ::RockDialectGen
    ::RockAttrDefsIncGen
    ::RockTuningParamAttrInterfaceIncGen
    ::RockAccelTuningParamAttrInterfaceIncGen
    ::RockGemmWrapperInterfaceIncGen
    ::RockConvInterfaceIncGen
    ::RockAcceptingViewOpInterfaceIncGen
    ::RockWriterOpInterfaceIncGen
    ::RockPassesIncGen
    ::RockDialectTuning
    MLIRBytecodeOpInterface
    MLIRBytecodeReader
    MLIRBytecodeWriter
    MLIRFuncDialect
    MLIRIR
    MLIRInferIntRangeInterface
    MLIRSupport
    MLIRControlFlowInterfaces
    MLIRInferTypeOpInterface
    MLIRSideEffectInterfaces
)


# iree_cc_library(
#   NAME
#     RockDialectConversions
#   SRCS
#     "${ROCMLIR_ROOT_DIR}/mlir/lib/Conversion/RockToGPU/RockToGPU.cpp"
#   DEPS
#     ::defs
#     ::RockDialectGen
#     ::RockAttrDefsIncGen
#     ::RockTuningParamAttrInterfaceIncGen
#     ::RockAccelTuningParamAttrInterfaceIncGen
#     ::RockGemmWrapperInterfaceIncGen
#     ::RockConvInterfaceIncGen
#     ::RockAcceptingViewOpInterfaceIncGen
#     ::RockWriterOpInterfaceIncGen
#     ::RockConversionPassesIncGen
#     MLIRBytecodeOpInterface
#     MLIRBytecodeReader
#     MLIRBytecodeWriter
#     MLIRFuncDialect
#     MLIRIR
#     MLIRInferIntRangeInterface
#     MLIRSupport
#     MLIRControlFlowInterfaces
#     MLIRInferTypeOpInterface
#     MLIRSideEffectInterfaces
# )

iree_tablegen_library(
  NAME
    RockDialectGen
  TD_FILE
    "${ROCMLIR_ROOT_DIR}/mlir/include/mlir/Dialect/Rock/IR/RockOps.td"
  OUTS
    -gen-dialect-decls mlir/Dialect/Rock/IR/RockOpsDialect.h.inc
    -gen-dialect-defs mlir/Dialect/Rock/IR/RockOpsDialect.cpp.inc
    -gen-op-decls mlir/Dialect/Rock/IR/RockOps.h.inc
    -gen-op-defs mlir/Dialect/Rock/IR/RockOps.cpp.inc
)

iree_tablegen_library(
  NAME
    RockAttrDefsIncGen
  TD_FILE
    "${ROCMLIR_ROOT_DIR}/mlir/include/mlir/Dialect/Rock/IR/RockAttrDefs.td"
  OUTS
    -gen-enum-decls mlir/Dialect/Rock/IR/RockTypes.h.inc
    -gen-enum-defs mlir/Dialect/Rock/IR/RockTypes.cpp.inc
    -gen-attrdef-decls mlir/Dialect/Rock/IR/RockAttrDefs.h.inc
    -gen-attrdef-defs mlir/Dialect/Rock/IR/RockAttrDefs.cpp.inc
)

iree_tablegen_library(
  NAME
    RockTuningParamAttrInterfaceIncGen
  TD_FILE
    "${ROCMLIR_ROOT_DIR}/mlir/include/mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.td"
  OUTS
    -gen-attr-interface-decls mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h.inc
    -gen-attr-interface-defs mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.cpp.inc
)

iree_tablegen_library(
  NAME
    RockAccelTuningParamAttrInterfaceIncGen
  TD_FILE
    "${ROCMLIR_ROOT_DIR}/mlir/include/mlir/Dialect/Rock/IR/RockAccelTuningParamAttrInterface.td"
  OUTS
    -gen-attr-interface-decls mlir/Dialect/Rock/IR/RockAccelTuningParamAttrInterface.h.inc
    -gen-attr-interface-defs mlir/Dialect/Rock/IR/RockAccelTuningParamAttrInterface.cpp.inc
)

iree_tablegen_library(
  NAME
    RockPassesIncGen
  TD_FILE
    "${ROCMLIR_ROOT_DIR}/mlir/include/mlir/Dialect/Rock/Passes.td"
  OUTS
    -gen-pass-decls mlir/Dialect/Rock/Passes.h.inc
)

iree_tablegen_library(
  NAME
    RockConversionPassesIncGen
  TD_FILE
    "${ROCMLIR_ROOT_DIR}/mlir/include/mlir/Conversion/RocMLIRPasses.td"
  OUTS
    -gen-pass-decls mlir/Dialect/Rock/RocMLIRConversions.h.inc
)

function(add_iree_tablegen_rock_interface interface)
  iree_tablegen_library(
    NAME
      ${interface}IncGen
    TD_FILE
      "${ROCMLIR_ROOT_DIR}/mlir/include/mlir/Dialect/Rock/IR/${interface}.td"
    OUTS
      -gen-op-interface-decls mlir/Dialect/Rock/IR/${interface}.h.inc
      -gen-op-interface-defs mlir/Dialect/Rock/IR/${interface}.cpp.inc
  )
endfunction()

add_iree_tablegen_rock_interface(RockGemmWrapperInterface)
add_iree_tablegen_rock_interface(RockConvInterface)
add_iree_tablegen_rock_interface(RockAcceptingViewOpInterface)
add_iree_tablegen_rock_interface(RockWriterOpInterface)



add_subdirectory(Passes)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    MLIRIR
    MLIRPass
    iree::compiler::PluginAPI
    iree::compiler::plugins::lib::rocmlir::RockDialectIR
    iree::compiler::plugins::lib::rocmlir::RockDialectUtility
    iree::compiler::plugins::lib::rocmlir::RockDialectTransforms
    iree::compiler::plugins::lib::rocmlir::Passes::IREERocmlirPasses
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    lib_rocmlir
  TARGET
    ::registration
)
