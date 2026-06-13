//===- TreeOps.td - Tree dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TREE_OPS
#define TREE_OPS

include "Tree/TreeTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/IR/SymbolInterfaces.td"

def Tree_FooOp : Tree_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `tree.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = tree.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Tree_FloatFeatureOp : Tree_Op<"floatFeature",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "A F32 feature operator for a tree node.";

    let description = [{
        Provide feature for a tree node.
        Support two types of arguments.
        //LLVM::LoadOp supports basic flow without load-code-hoisting optimization.
        For regular feature, input F32 pointer and arith::ConstantIntOp as feature address and feature index.
        For feature with pinned register, input LLVM::GlobalOp as global address.
    }];

    let arguments = (ins LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$featureAddr, OptionalAttr<IndexAttr>:$featureIndex, OptionalAttr<FlatSymbolRefAttr>:$globalName);

    //let results = (outs LLVM_AnyPointer:$result);
    let results = (outs F32:$result);

    /*let builders = [
        OpBuilder<(ins "Value":$featureAddr, "int64_t":$featureIndex)>,
        OpBuilder<(ins "Value":$featureAddr, "int64_t":$featureIndex, "LLVM::GlobalOp":$global),
        [{
            build($_builder, $_state,
                  featureAddr, featureIndex,
                  //LLVM::LLVMPointerType::get($_builder.getContext(), global.getAddrSpace()),
                  global.getSymName());
        }]>,
        OpBuilder<(ins "Value":$featureAddr, "int64_t":$featureIndex, "StringRef":$globalName),
        [{
            //odsState.getOrAddProperties<Properties>().globalName = globalName;
            odsState.addAttribute("globalName", FlatSymbolRefAttr::get($_builder.getContext(), globalName));
            //odsState.addTypes(result);
        }]>
    ];*/

    let extraClassDeclaration = [{
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value featureAddr, int64_t featureIndex,
                          LLVM::GlobalOp global) {
            odsState.addOperands(featureAddr);
            odsState.addAttribute("featureIndex", builder.getIndexAttr(featureIndex));
            odsState.addAttribute("globalName", FlatSymbolRefAttr::get(builder.getContext(), global.getSymName()));
            //odsState.addTypes(LLVM::LLVMPointerType::get(builder.getContext(), global.getAddrSpace()));
            odsState.addTypes(builder.getF32Type());
        }
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value featureAddr, int64_t featureIndex) {
            odsState.addOperands(featureAddr);
            odsState.addAttribute("featureIndex", builder.getIndexAttr(featureIndex));
            odsState.addTypes(builder.getF32Type());
        }
    }];
}

def Tree_IntFeatureOp : Tree_Op<"intFeature",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "A I32 feature operator for a tree node.";

    let description = [{
        Provide feature for a tree node.
        Support two types of arguments.
        //LLVM::LoadOp supports basic flow without load-code-hoisting optimization.
        For regular feature, input I32 pointer and arith::ConstantIntOp as feature address and feature index.
        For feature with pinned register, input LLVM::GlobalOp as global address.
    }];

    let arguments = (ins LLVM_ScalarOrVectorOf<LLVM_AnyPointer>:$featureAddr, OptionalAttr<IndexAttr>:$featureIndex, OptionalAttr<FlatSymbolRefAttr>:$globalName);

    let results = (outs I32:$result);

    let extraClassDeclaration = [{
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value feature, int threshold,
                          arith::CmpIPredicate predicate,
                          double trueProbability, double falseProbability,
                          Block *trueDest, Block *falseDest) {
            odsState.addOperands(feature);
            odsState.addAttribute("threshold", builder.getI32IntegerAttr(threshold));
            odsState.addAttribute("thresholdIndex", builder.getIndexAttr(-1));
            odsState.addAttribute("predicate", arith::CmpIPredicateAttr::get(builder.getContext(), predicate));
            odsState.addAttribute("trueProbability", builder.getF64FloatAttr(trueProbability));
            odsState.addAttribute("falseProbability", builder.getF64FloatAttr(falseProbability));
            odsState.addSuccessors(trueDest);
            odsState.addSuccessors(falseDest);
        }
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value feature, int64_t featureIndex,
                          LLVM::GlobalOp global) {
            odsState.addOperands(feature);
            odsState.addAttribute("featureIndex", builder.getIndexAttr(featureIndex));
            odsState.addAttribute("globalName", FlatSymbolRefAttr::get(builder.getContext(), global.getSymName()));
            odsState.addTypes(builder.getI32Type());
        }
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value feature, int64_t featureIndex) {
            odsState.addOperands(feature);
            odsState.addAttribute("featureIndex", builder.getIndexAttr(featureIndex));
            odsState.addTypes(builder.getI32Type());
        }
    }];
}

def Tree_FloatNodeOp : Tree_Op<"floatNode", [BranchOpInterface, Terminator]> {
    let summary = "Decision tree node with F32 comparison";

    let description = [{
        WIP `tree.floatNode`
        //LLVM::LoadOp
        arith::ConstantOp
        arith::CmpFPredicate
        True successor False successor.
    }];

    let arguments = (ins F32:$feature, F32Attr:$threshold, IndexAttr:$thresholdIndex, Arith_CmpFPredicateAttr:$predicate, F64Attr:$trueProbability, F64Attr:$falseProbability);
    //let regions = (region AnyRegion:$trueRegion, AnyRegion:$falseRegion);
    let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

    let extraClassDeclaration = [{
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value feature, float threshold,
                          arith::CmpFPredicate predicate,
                          double trueProbability, double falseProbability,
                          Block *trueDest, Block *falseDest) {
            odsState.addOperands(feature);
            odsState.addAttribute("threshold", builder.getF32FloatAttr(threshold));
            odsState.addAttribute("thresholdIndex", builder.getIndexAttr(-1));
            odsState.addAttribute("predicate", arith::CmpFPredicateAttr::get(builder.getContext(), predicate));
            odsState.addAttribute("trueProbability", builder.getF64FloatAttr(trueProbability));
            odsState.addAttribute("falseProbability", builder.getF64FloatAttr(falseProbability));
            odsState.addSuccessors(trueDest);
            odsState.addSuccessors(falseDest);
        }
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value feature, float threshold,
                          arith::CmpFPredicate predicate,
                          Block *trueDest, Block *falseDest) {
            odsState.addOperands(feature);
            odsState.addAttribute("threshold", builder.getF32FloatAttr(threshold));
            odsState.addAttribute("thresholdIndex", builder.getIndexAttr(-1));
            odsState.addAttribute("predicate", arith::CmpFPredicateAttr::get(builder.getContext(), predicate));
            odsState.addAttribute("trueProbability", builder.getF64FloatAttr(0));
            odsState.addAttribute("falseProbability", builder.getF64FloatAttr(0));
            odsState.addSuccessors(trueDest);
            odsState.addSuccessors(falseDest);
        }
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value feature, float threshold, int64_t thresholdIndex,
                          arith::CmpFPredicate predicate,
                          double trueProbability, double falseProbability,
                          Block *trueDest, Block *falseDest) {
            odsState.addOperands(feature);
            odsState.addAttribute("threshold", builder.getF32FloatAttr(threshold));
            odsState.addAttribute("thresholdIndex", builder.getIndexAttr(thresholdIndex));
            odsState.addAttribute("predicate", arith::CmpFPredicateAttr::get(builder.getContext(), predicate));
            odsState.addAttribute("trueProbability", builder.getF64FloatAttr(trueProbability));
            odsState.addAttribute("falseProbability", builder.getF64FloatAttr(falseProbability));
            odsState.addSuccessors(trueDest);
            odsState.addSuccessors(falseDest);
        }
        SuccessorOperands getSuccessorOperands(unsigned index) {
            return SuccessorOperands(0, MutableOperandRange(*this, 0, 0));
        }
    }];
}

def Tree_IntNodeOp : Tree_Op<"intNode", [BranchOpInterface, Terminator]> {
    let summary = "Decision tree node with I32 comparison";

    let description = [{
        WIP `tree.intNode`
        True successor False successor.
    }];

    let arguments = (ins I32:$feature, I32Attr:$threshold, IndexAttr:$thresholdIndex, Arith_CmpIPredicateAttr:$predicate, F64Attr:$trueProbability, F64Attr:$falseProbability);
    //let regions = (region AnyRegion:$trueRegion, AnyRegion:$falseRegion);
    let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

    /*let builders = [OpBuilder<(ins "mlir::Value":$feature, "mlir::Value":$threshold, "mlir::arith::CmpIPredicate":$predicate,
        "mlir::Value":$trueProbability, "mlir::Value":$falseProbability, "mlir::Block*":$trueDest, "mlir::Block*":$falseDest)
    >];*/

    let extraClassDeclaration = [{
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value feature, int threshold,
                          arith::CmpIPredicate predicate,
                          double trueProbability, double falseProbability,
                          Block *trueDest, Block *falseDest) {
            odsState.addOperands(feature);
            odsState.addAttribute("threshold", builder.getI32IntegerAttr(threshold));
            odsState.addAttribute("thresholdIndex", builder.getIndexAttr(-1));
            odsState.addAttribute("predicate", arith::CmpIPredicateAttr::get(builder.getContext(), predicate));
            odsState.addAttribute("trueProbability", builder.getF64FloatAttr(trueProbability));
            odsState.addAttribute("falseProbability", builder.getF64FloatAttr(falseProbability));
            odsState.addSuccessors(trueDest);
            odsState.addSuccessors(falseDest);
        }
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value feature, int threshold,
                          arith::CmpIPredicate predicate,
                          Block *trueDest, Block *falseDest) {
            odsState.addOperands(feature);
            odsState.addAttribute("threshold", builder.getI32IntegerAttr(threshold));
            odsState.addAttribute("thresholdIndex", builder.getIndexAttr(-1));
            odsState.addAttribute("predicate", arith::CmpIPredicateAttr::get(builder.getContext(), predicate));
            odsState.addAttribute("trueProbability", builder.getF64FloatAttr(0));
            odsState.addAttribute("falseProbability", builder.getF64FloatAttr(0));
            odsState.addSuccessors(trueDest);
            odsState.addSuccessors(falseDest);
        }
        static void build(OpBuilder &builder, OperationState &odsState,
                          Value feature, int threshold, int64_t thresholdIndex,
                          arith::CmpIPredicate predicate,
                          double trueProbability, double falseProbability,
                          Block *trueDest, Block *falseDest) {
            odsState.addOperands(feature);
            odsState.addAttribute("threshold", builder.getI32IntegerAttr(threshold));
            odsState.addAttribute("thresholdIndex", builder.getIndexAttr(thresholdIndex));
            odsState.addAttribute("predicate", arith::CmpIPredicateAttr::get(builder.getContext(), predicate));
            odsState.addAttribute("trueProbability", builder.getF64FloatAttr(trueProbability));
            odsState.addAttribute("falseProbability", builder.getF64FloatAttr(falseProbability));
            odsState.addSuccessors(trueDest);
            odsState.addSuccessors(falseDest);
        }
        SuccessorOperands getSuccessorOperands(unsigned index) {
            return SuccessorOperands(0, MutableOperandRange(*this, 0, 0));
        }
    }];
}

#endif // TREE_OPS
