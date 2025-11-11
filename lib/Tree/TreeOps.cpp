//===- TreeOps.cpp - Tree dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tree/TreeOps.h"
#include "Tree/TreeDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "Tree/TreeOps.cpp.inc"

namespace mlir {
// Copied and modified from LLVMDialect.cpp
static Operation *parentLLVMModule(Operation *op) {
    Operation *module = op->getParentOp();
    while (module && !LLVM::satisfiesLLVMModule(module))
        module = module->getParentOp();
    assert(module && "unexpected operation outside of a module");
    return module;
}

LogicalResult tree::FloatFeatureOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    if (!getGlobalNameAttr()) {
        return success();
    }
    Operation *symbol = symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr());
    LLVM::GlobalOp global = dyn_cast_or_null<LLVM::GlobalOp>(symbol);
    if (!global) {
        return emitOpError("must reference a global defined by 'llvm.mlir.global', "
                           "'llvm.mlir.alias' or 'llvm.func'");
    }
    /*else if (global.getAddrSpace() != getType().getAddressSpace()) {
        return emitOpError("pointer address space must match address space of the "
                           "referenced global or alias");
    }*/
    return success();
}

LogicalResult tree::IntFeatureOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
    if (!getGlobalNameAttr()) {
        return success();
    }
    Operation *symbol = symbolTable.lookupSymbolIn(parentLLVMModule(*this), getGlobalNameAttr());
    LLVM::GlobalOp global = dyn_cast_or_null<LLVM::GlobalOp>(symbol);
    if (!global) {
        return emitOpError("must reference a global defined by 'llvm.mlir.global', "
                           "'llvm.mlir.alias' or 'llvm.func'");
    }
    /*else if (global.getAddrSpace() != getType().getAddressSpace()) {
        return emitOpError("pointer address space must match address space of the "
                           "referenced global or alias");
    }*/
    return success();
}
} // namespace mlir
