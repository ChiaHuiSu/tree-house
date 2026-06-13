//===- TreeDialect.cpp - Tree dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Tree/TreeDialect.h"
#include "Tree/TreeOps.h"
#include "Tree/TreeTypes.h"

using namespace mlir;
using namespace mlir::tree;

#include "Tree/TreeOpsDialect.cpp.inc"

//#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

struct TreeDialectInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *call,
                       Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }
  bool isLegalToInline(Operation *,
                       Region *,
                       bool,
                       IRMapping &) const final {
    return true;
  }
  void handleTerminator(Operation *op,
                        ValueRange valuesToReplace) const final {
    if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
      for (auto [operand, result] :
           llvm::zip(returnOp.getOperands(), valuesToReplace)) {
        result.replaceAllUsesWith(operand);
      }
    }
  }
  void handleTerminator(Operation *op, Block *newDest) const final {}
};

//===----------------------------------------------------------------------===//
// Tree dialect.
//===----------------------------------------------------------------------===//

void TreeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Tree/TreeOps.cpp.inc"
      >();
  registerTypes();
  addInterfaces<TreeDialectInlinerInterface>();
}
