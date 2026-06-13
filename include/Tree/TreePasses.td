//===- TreePsss.td - Tree dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TREE_PASS
#define TREE_PASS

include "mlir/Pass/PassBase.td"
/*
def TreeSwitchBarFoo: Pass<"tree-switch-bar-foo", "::mlir::ModuleOp"> {
    let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
    let description = [{
        Switches the name of a FuncOp named `bar` to `foo` and folds.
        ```
        func.func @bar() {
            return
        }
        // Gets transformed to:
        func.func @foo() {
            return
        }
        ```
    }];
}
*/
def TreeReorder: Pass<"tree-reorder", "::mlir::ModuleOp"> {
    let summary = "Reorder tree_ functions based on the probability distribution of features.";
    let description = [{
        Reorder tree_ functions based on the probability distribution of features or/and thresholds.
        Calculate distance using N-dimension Euclidean distance, Jensen–Shannon divergence, or Manhattan distance.
        Minimize Hamiltonian path using customized LKH-2 TSP solver.
    }];
    let options = [
        Option<"isFLIntEnabled", "is-flint-enabled", "bool", /*default=*/"false",
               "If FLInt is enabled, this option should be set to true to have correct thresholds.">,
        Option<"useFeature", "use-feature", "bool", /*default=*/"false",
               "Reorder trees based on features.">,
        Option<"useThreshold", "use-threshold", "bool", /*default=*/"false",
               "Reorder trees based on thresholds by assuming thresholdOrdering is enabled.">
    ];
}

def TreeInline: Pass<"tree-inline", "::mlir::ModuleOp"> {
    let summary = "Inline tree_ functions.";
    let description = [{
        Inline tree_ functions.
        Replace func::ReturnOP with cf::BranchOp.
        Remove func::FuncOp and func::CallOp.
    }];
    let options = [
        Option<"pinningAmount", "pinning-amount", "size_t", /*default=*/"0",
               "Hoist feature load of k (0~32) most used features to the top of the program.">
    ];
}

def TreeSwap: Pass<"tree-swap", "::mlir::ModuleOp"> {
    let summary = "Swap left child and right child based on probability.";
    let description = [{
        Swap left child and right child based on probability.
        ```
        ```
    }];
}

def TreeFLInt: Pass<"tree-flint", "::mlir::ModuleOp"> {
    let summary = "Transform comparison of tree nodes from floating point to integer.";
    let description = [{
        Transform comparison of tree nodes from floating point to integer.
        ```
        %0 = "arith.constant"() <{value = 1043073073 : i32}> : () -> i32
        %1 = "arith.constant"() <{value = 7 : i32}> : () -> i32
        %2 = "llvm.getelementptr"(%arg0, %1) <{elem_type = i32, rawConstantIndices = array<i32: -2147483648>}> : (!llvm.ptr, i32) -> !llvm.ptr
        %3 = "llvm.load"(%2) <{ordering = 0 : i64}> : (!llvm.ptr) -> i32
        "tree.floatNode"(%3, %0)[^bb1, ^bb20] {predicate = 4 : i64} : (i32, i32) -> ()
        // Transforms into: WIP
        ```
    }];
}

def TreeLoadHoisting: Pass<"load-hoisting", "::mlir::ModuleOp"> {
    let summary = "Hoist feature load OPs with high probability to previous nodes.";
    let description = [{
        Hoist feature load OPs with high probability to previous nodes.
        If the probability of a child node divided by the parent node is greater than the specified threshold, 
        the feature load OP of the child node would be hoisted to the parent node.
    }];
    let options = [
        Option<"probabilityThreshold", "threshold", "float", /*default=*/"0.8",
               "Specify probability threshold (0.5~1).">
    ];
}

def TreeLowering: Pass<"tree-lowering", "::mlir::ModuleOp"> {
    let summary = "Lower Tree to arith, memref, and LLVM.";
    let description = [{
        Lower Tree to arith, memref, and LLVM.
        Enable the ordering option to order thresholds of nodes into a memref global OP.
        Also, each tree would start with a global OP and have a load OP for every comparison instead.
    }];
    let options = [
        Option<"thresholdOrdering", "ordering", "bool", /*default=*/"false",
               "Order thresholds in the memory block.">
    ];
}

#endif // TREE_PASS
