//===- TreePasses.cpp - Tree passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/IRMapping.h"

#include <tuple>
#include <vector>
#include <algorithm>
#include <set>

#include "Tree/TreePasses.h"

namespace mlir::tree {
//#define GEN_PASS_DEF_TREESWITCHBARFOO
#define GEN_PASS_DEF_TREEINLINE
#define GEN_PASS_DEF_TREESWAP
#define GEN_PASS_DEF_TREEFLINT
#define GEN_PASS_DEF_TREELOADHOISTING
#define GEN_PASS_DEF_TREELOWERING
#include "Tree/TreePasses.h.inc"

namespace {
/*class TreeSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
    using OpRewritePattern<func::FuncOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(func::FuncOp op,
                                  PatternRewriter &rewriter) const final {
        if (op.getSymName() == "bar") {
            rewriter.updateRootInPlace(op, [&op]() { op.setSymName("foo"); });
            return success();
        }
        return failure();
    }
};

class TreeSwitchBarFoo : public impl::TreeSwitchBarFooBase<TreeSwitchBarFoo> {
public:
    using impl::TreeSwitchBarFooBase<TreeSwitchBarFoo>::TreeSwitchBarFooBase;
    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        patterns.add<TreeSwitchBarFooRewriter>(&getContext());
        FrozenRewritePatternSet patternSet(std::move(patterns));
        //if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
            signalPassFailure();
        }
    }
};*/
// TreeInline
class TreeInline : public impl::TreeInlineBase<TreeInline> {
public:
    using impl::TreeInlineBase<TreeInline>::TreeInlineBase;
    void runOnOperation() final {
        ModuleOp moduleOp = getOperation();
        OpBuilder builder(moduleOp.getContext());
        SymbolTable symbolTable(moduleOp);
        std::vector<func::CallOp> callOps;
        moduleOp.walk([&](func::CallOp callOp) {
            callOps.push_back(callOp);
        });
        for (func::CallOp callOp : callOps) {
            auto funcOp = symbolTable.lookup<func::FuncOp>(callOp.getCallee());
            // No such funcOp or only a declaration without implementation
            if (!funcOp || funcOp.isDeclaration()) {
                return;
            }
            IRMapping mapper;
            for (auto [argument, operand] : zip(funcOp.getArguments(), callOp.getOperands())) {
                mapper.map(argument, operand);
            }
            Block *callBlock = callOp.getOperation()->getBlock();
            Block *afterCallBlock = callBlock->splitBlock(callOp);
            builder.setInsertionPointToStart(afterCallBlock);

            auto resultTypes = callOp.getResultTypes();
            afterCallBlock->addArguments(resultTypes, SmallVector<Location>(resultTypes.size(), callOp.getLoc()));

            for (auto [result, argument] : zip(callOp.getResults(), afterCallBlock->getArguments())) {
                result.replaceAllUsesWith(argument);
            }

            DenseMap<Block*, Block*> blockMap;
            for (Block &funcBlock : funcOp.getBody().getBlocks()) {
                Block *newBlock = builder.createBlock(callBlock->getParent());
                blockMap[&funcBlock] = newBlock;
                mapper.map(&funcBlock, newBlock);
            }

            for (Block &funcBlock : funcOp.getBody().getBlocks()) {
                Block *newBlock = blockMap.lookup(&funcBlock);
                builder.setInsertionPointToStart(newBlock);
                for (Operation &op : funcBlock.getOperations()) {
                    if (func::ReturnOp returnOp = dyn_cast<func::ReturnOp>(op)) {
                        SmallVector<Value> mappedOperands;
                        for (Value operand : returnOp.getOperands()) {
                            mappedOperands.push_back(mapper.lookup(operand));
                        }
                        builder.setInsertionPointToEnd(newBlock);
                        builder.create<cf::BranchOp>(returnOp.getLoc(), afterCallBlock, mappedOperands);
                    }
                    else if (auto branchOp = dyn_cast<BranchOpInterface>(op)) {
                        Operation *clonedOp = op.clone(mapper);
                        for (unsigned index = 0; index < branchOp->getNumSuccessors(); ++index) {
                            clonedOp->setSuccessor(blockMap.lookup(branchOp->getSuccessor(index)), index);
                        }
                        newBlock->push_back(clonedOp);
                    }
                    else {
                        builder.clone(op, mapper);
                    }
                }
            }

            builder.setInsertionPointToEnd(callBlock);
            builder.create<cf::BranchOp>(callOp.getLoc(), blockMap[&funcOp.getBody().front()]);
            callOp.erase();
        }
    }
};

// TreeSwap
void treeSwapFloat(tree::FloatNodeOp &op) {
    if (op.getTrueProbability().convertToDouble() < op.getFalseProbability().convertToDouble()) {
        auto formerTrueProbability = op.getTrueProbability();
        auto formerFalseProbability = op.getFalseProbability();
        op.setTrueProbability(formerFalseProbability);
        op.setFalseProbability(formerTrueProbability);
        if (op.getPredicate() == arith::CmpFPredicate::OLT) {
            op.setPredicate(arith::CmpFPredicate::OGE);
        }
        else if (op.getPredicate() == arith::CmpFPredicate::OLE) {
            op.setPredicate(arith::CmpFPredicate::OGT);
        }
        else if (op.getPredicate() == arith::CmpFPredicate::OGT) {
            op.setPredicate(arith::CmpFPredicate::OLE);
        }
        else if (op.getPredicate() == arith::CmpFPredicate::OGE) {
            op.setPredicate(arith::CmpFPredicate::OLT);
        }
        Block* trueDest = op.getSuccessor(0);
        Block* falseDest = op.getSuccessor(1);
        op.setSuccessor(falseDest, 0);
        op.setSuccessor(trueDest, 1);
    }
}

void treeSwapInt(tree::IntNodeOp &op) {
    if (op.getTrueProbability().convertToDouble() < op.getFalseProbability().convertToDouble()) {
        auto formerTrueProbability = op.getTrueProbability();
        auto formerFalseProbability = op.getFalseProbability();
        op.setTrueProbability(formerFalseProbability);
        op.setFalseProbability(formerTrueProbability);
        if (op.getPredicate() == arith::CmpIPredicate::slt) {
            op.setPredicate(arith::CmpIPredicate::sge);
        }
        else if (op.getPredicate() == arith::CmpIPredicate::sle) {
            op.setPredicate(arith::CmpIPredicate::sgt);
        }
        else if (op.getPredicate() == arith::CmpIPredicate::sgt) {
            op.setPredicate(arith::CmpIPredicate::sle);
        }
        else if (op.getPredicate() == arith::CmpIPredicate::sge) {
            op.setPredicate(arith::CmpIPredicate::slt);
        }
        Block* trueDest = op.getSuccessor(0);
        Block* falseDest = op.getSuccessor(1);
        op.setSuccessor(falseDest, 0);
        op.setSuccessor(trueDest, 1);
    }
}

class TreeSwap : public impl::TreeSwapBase<TreeSwap> {
public:
    using impl::TreeSwapBase<TreeSwap>::TreeSwapBase;
    void runOnOperation() final {
        getOperation().walk([](Operation *op) {
            if (tree::FloatNodeOp nodeOp = dyn_cast<tree::FloatNodeOp>(op)) {
                treeSwapFloat(nodeOp);
            }
            else if (tree::IntNodeOp nodeOp = dyn_cast<tree::IntNodeOp>(op)) {
                treeSwapInt(nodeOp);
            }
        });
    }
};
// TreeFLInt
class TreeNodeFLIntRewriter : public OpRewritePattern<tree::FloatNodeOp> {
public:
    TreeNodeFLIntRewriter(MLIRContext *context, ModuleOp moduleOp)
        : OpRewritePattern<tree::FloatNodeOp>(context), moduleOp(moduleOp) {};
    using OpRewritePattern<tree::FloatNodeOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tree::FloatNodeOp op, PatternRewriter &rewriter) const final {
        tree::FloatFeatureOp floatFeature = op.getOperand().getDefiningOp<tree::FloatFeatureOp>();
        if (!floatFeature) {
            return failure();
        }
        Value intFeature;
        if (auto globalName = floatFeature->getAttrOfType<FlatSymbolRefAttr>("globalName")) {
            LLVM::GlobalOp globalOp = dyn_cast_or_null<LLVM::GlobalOp>(SymbolTable(moduleOp).lookup(globalName.getValue()));
            intFeature = rewriter.create<tree::IntFeatureOp>(floatFeature.getLoc(), floatFeature.getFeatureAddr(), floatFeature.getFeatureIndex()->getSExtValue(), globalOp);
        }
        else {
            intFeature = rewriter.create<tree::IntFeatureOp>(floatFeature.getLoc(), floatFeature.getFeatureAddr(), floatFeature.getFeatureIndex()->getSExtValue());
        }
        union {
            float f32;
            int i32;
        } threshold;
        threshold.f32 = op.getThreshold().convertToFloat();
        arith::CmpIPredicate compare;
        if (op.getPredicate() == arith::CmpFPredicate::OLT) {
            compare = threshold.f32 < 0 ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::slt;
        }
        else if (op.getPredicate() == arith::CmpFPredicate::OLE) {
            compare = threshold.f32 < 0 ? arith::CmpIPredicate::sge : arith::CmpIPredicate::sle;
        }
        else if (op.getPredicate() == arith::CmpFPredicate::OGT) {
            compare = threshold.f32 < 0 ? arith::CmpIPredicate::slt : arith::CmpIPredicate::sgt;
        }
        else if (op.getPredicate() == arith::CmpFPredicate::OGE) {
            compare = threshold.f32 < 0 ? arith::CmpIPredicate::sle : arith::CmpIPredicate::sge;
        }
        if (threshold.f32 < 0) {
            //Value mask = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0x80000000, rewriter.getI32Type());
            Value mask = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0x80000000, 32);
            intFeature = rewriter.create<mlir::arith::XOrIOp>(op.getLoc(), intFeature, mask);
            threshold.f32 = -threshold.f32;
        }
        rewriter.replaceOp(op, rewriter.create<tree::IntNodeOp>(op.getLoc(), intFeature, threshold.i32, compare, op.getTrueProbability().convertToDouble(), op.getFalseProbability().convertToDouble(), op.getSuccessor(0), op.getSuccessor(1)));
        return success();
    }
private:
    ModuleOp moduleOp;
};

class TreeFLInt : public impl::TreeFLIntBase<TreeFLInt> {
public:
    using impl::TreeFLIntBase<TreeFLInt>::TreeFLIntBase;
    void runOnOperation() final {
        RewritePatternSet patterns(&getContext());
        patterns.add<TreeNodeFLIntRewriter>(&getContext(), getOperation());
        FrozenRewritePatternSet patternSet(std::move(patterns));
        //if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        if (failed(applyPatternsGreedily(getOperation(), patternSet))) {
            signalPassFailure();
        }
    }
};
// TreeLoadHoisting
void treeFloatLoadHoisting(tree::FloatNodeOp &op, float probabilityThreshold) {
    Block *childBlock;
    if (op.getTrueProbability().convertToDouble() * (1.0f - probabilityThreshold) >= op.getFalseProbability().convertToDouble() * probabilityThreshold) {
        childBlock = op.getSuccessor(0);
    }
    else if (op.getTrueProbability().convertToDouble() * probabilityThreshold <= op.getFalseProbability().convertToDouble() * (1.0f - probabilityThreshold)) {
        childBlock = op.getSuccessor(1);
    }
    else {
        return;
    }
    tree::FloatFeatureOp childFeature;
    for (const Operation& childOp : *childBlock) {
        if ((childFeature = dyn_cast<tree::FloatFeatureOp>(childOp))) {
            break;
        }
    }
    if (!childFeature) {
        return;
    }
    tree::FloatFeatureOp feature = dyn_cast<tree::FloatFeatureOp>(op.getOperand().getDefiningOp());
    if (!feature) {
        return;
    }
    if (childFeature.getFeatureAddr() == feature.getFeatureAddr() && childFeature.getFeatureIndex() == feature.getFeatureIndex()) {
        childFeature.replaceAllUsesWith(op.getOperand().getDefiningOp());
        childFeature.erase();
    }
    else {
        childFeature->moveBefore(op);
    }
};

void treeIntLoadHoisting(tree::IntNodeOp &op, float probabilityThreshold) {
    Block *childBlock;
    if (op.getTrueProbability().convertToDouble() * (1.0f - probabilityThreshold) >= op.getFalseProbability().convertToDouble() * probabilityThreshold) {
        childBlock = op.getSuccessor(0);
    }
    else if (op.getTrueProbability().convertToDouble() * probabilityThreshold <= op.getFalseProbability().convertToDouble() * (1.0f - probabilityThreshold)) {
        childBlock = op.getSuccessor(1);
    }
    else {
        return;
    }
    tree::IntFeatureOp childFeature;
    for (const Operation& childOp : *childBlock) {
        if ((childFeature = dyn_cast<tree::IntFeatureOp>(childOp))) {
            break;
        }
    }
    if (!childFeature) {
        return;
    }
    tree::IntFeatureOp feature = dyn_cast<tree::IntFeatureOp>(op.getOperand().getDefiningOp());
    if (!feature) {
        return;
    }
    if (childFeature.getFeatureAddr() == feature.getFeatureAddr() && childFeature.getFeatureIndex() == feature.getFeatureIndex()) {
        childFeature.replaceAllUsesWith(op.getOperand().getDefiningOp());
        childFeature.erase();
    }
    else {
        childFeature->moveBefore(op);
    }
};

class TreeLoadHoisting : public impl::TreeLoadHoistingBase<TreeLoadHoisting> {
public:
    using impl::TreeLoadHoistingBase<TreeLoadHoisting>::TreeLoadHoistingBase;
    void runOnOperation() final {
        getOperation().walk([&](Operation *op) {
            if (tree::FloatNodeOp nodeOp = dyn_cast<tree::FloatNodeOp>(op)) {
                treeFloatLoadHoisting(nodeOp, probabilityThreshold);
            }
            else if (tree::IntNodeOp nodeOp = dyn_cast<tree::IntNodeOp>(op)) {
                treeIntLoadHoisting(nodeOp, probabilityThreshold);
            }
        });
    }
};
// TreeLowering
typedef std::tuple<double, long long, float> probabilityIndexFloatThreshold;
typedef std::tuple<double, long long, int> probabilityIndexIntThreshold;
/*
void TreeFloatThresholdOrdering(tree::FloatNodeOp &op, std::vector<probabilityIndexFloatThreshold> &floatThresholdTupleVector) {
    op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, floatThresholdTupleVector.size()));
    floatThresholdTupleVector.emplace_back(op.getTrueProbability().convertToDouble() + op.getFalseProbability().convertToDouble(), floatThresholdTupleVector.size(), op.getThreshold().convertToFloat());
}

void TreeIntThresholdOrdering(tree::IntNodeOp &op, std::vector<probabilityIndexIntThreshold> &intThresholdTupleVector) {
    op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, intThresholdTupleVector.size()));
    intThresholdTupleVector.emplace_back(op.getTrueProbability().convertToDouble() + op.getFalseProbability().convertToDouble(), intThresholdTupleVector.size(), op.getThreshold());
}

void TreeVectorInvertIndex(std::vector<probabilityIndexFloatThreshold> &floatThresholdTupleVector, std::vector<probabilityIndexIntThreshold> &intThresholdTupleVector) {
    typedef std::pair<long long, long long> inverseIndexPair;
    if (!floatThresholdTupleVector.empty()) {
        std::vector<inverseIndexPair> inverseIndexVector(floatThresholdTupleVector.size());
        for (unsigned index = 0; index < floatThresholdTupleVector.size(); ++index) {
            inverseIndexVector[index] = std::pair(std::get<1>(floatThresholdTupleVector[index]), index);
        }
        std::sort(inverseIndexVector.begin(), inverseIndexVector.end());
        for (unsigned index = 0; index < floatThresholdTupleVector.size(); ++index) {
            std::get<1>(floatThresholdTupleVector[index]) = inverseIndexVector[index].second;
        }
    }
    else if (!intThresholdTupleVector.empty()) {
        std::vector<inverseIndexPair> inverseIndexVector(intThresholdTupleVector.size());
        for (unsigned index = 0; index < intThresholdTupleVector.size(); ++index) {
            inverseIndexVector[index] = std::pair(std::get<1>(intThresholdTupleVector[index]), index);
        }
        std::sort(inverseIndexVector.begin(), inverseIndexVector.end());
        for (unsigned index = 0; index < intThresholdTupleVector.size(); ++index) {
            std::get<1>(intThresholdTupleVector[index]) = inverseIndexVector[index].second;
        }
    }
}
*/
void TreeBuildGlobal(ModuleOp moduleOp, std::vector<probabilityIndexFloatThreshold> &floatThresholdTupleVector, std::vector<probabilityIndexIntThreshold> &intThresholdTupleVector) {
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToStart(moduleOp.getBody());
    if (!floatThresholdTupleVector.empty()) {
        std::vector<float> floatThresholdVector(floatThresholdTupleVector.size());
        std::transform(floatThresholdTupleVector.cbegin(), floatThresholdTupleVector.cend(), floatThresholdVector.begin(), [](const probabilityIndexFloatThreshold tupleElement) {
            return std::get<2>(tupleElement);
        });
        std::unordered_map<float, int> thresholdMap;
        for (int index = 0; index < (int)floatThresholdTupleVector.size(); ++index) {
            ++thresholdMap[floatThresholdVector[index]];
        }
        llvm::outs() << "         Thresholds count: " << floatThresholdTupleVector.size() << '\n';
        llvm::outs() << "Distinct thresholds count: " << thresholdMap.size() << '\n';
        MemRefType type = MemRefType::get({static_cast<long>(floatThresholdTupleVector.size())}, builder.getF32Type());
        DenseFPElementsAttr value = DenseFPElementsAttr::get(RankedTensorType::get({static_cast<long>(floatThresholdTupleVector.size())}, builder.getF32Type()), llvm::ArrayRef(floatThresholdVector));
        builder.create<memref::GlobalOp>(moduleOp.getLoc(), builder.getStringAttr("global_float_thresholds"), builder.getStringAttr("private"), type, value, true, builder.getI64IntegerAttr(64));
    }
    else if (!intThresholdTupleVector.empty()) {
        std::vector<int> intThresholdVector(intThresholdTupleVector.size());
        std::transform(intThresholdTupleVector.cbegin(), intThresholdTupleVector.cend(), intThresholdVector.begin(), [](const probabilityIndexIntThreshold tupleElement) {
            return std::get<2>(tupleElement);
        });
        MemRefType type = MemRefType::get({static_cast<long>(intThresholdTupleVector.size())}, builder.getI32Type());
        DenseIntElementsAttr value = DenseIntElementsAttr::get(RankedTensorType::get({static_cast<long>(intThresholdTupleVector.size())}, builder.getI32Type()), llvm::ArrayRef(intThresholdVector));
        builder.create<memref::GlobalOp>(moduleOp.getLoc(), builder.getStringAttr("global_int_thresholds"), builder.getStringAttr("private"), type, value, true, builder.getI64IntegerAttr(64));
    }
}

void TreeBuildGetGlobal(func::FuncOp &funcOp, std::vector<probabilityIndexFloatThreshold> &floatThresholdTupleVector, std::vector<probabilityIndexIntThreshold> &intThresholdTupleVector) {
    Block &entryBlock = funcOp.front();
    OpBuilder builder(&entryBlock, entryBlock.begin());
    if (!floatThresholdTupleVector.empty()) {
        builder.create<memref::GetGlobalOp>(funcOp.getLoc(), MemRefType::get({static_cast<long>(floatThresholdTupleVector.size())}, builder.getF32Type()), builder.getStringAttr("global_float_thresholds"));
    }
    else if (!intThresholdTupleVector.empty()) {
        builder.create<memref::GetGlobalOp>(funcOp.getLoc(), MemRefType::get({static_cast<long>(intThresholdTupleVector.size())}, builder.getI32Type()), builder.getStringAttr("global_int_thresholds"));
    }
}
/*
void TreeFloatThresholdCompression(tree::FloatNodeOp &op, std::unordered_map<float, double> &floatThresholdMap) {
    op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, floatThresholdMap.size()));
    floatThresholdMap[op.getThreshold().convertToFloat()] += op.getTrueProbability().convertToDouble() + op.getFalseProbability().convertToDouble();
}*/
/*
void TreeFloatThresholdCompression(tree::FloatNodeOp &op, std::vector<probabilityIndexFloatThreshold> &floatThresholdTupleVector, std::unordered_map<float, int> &floatThresholdMap) {
    if (!floatThresholdMap.count(op.getThreshold().convertToFloat())) {
        floatThresholdTupleVector.emplace_back(op.getTrueProbability().convertToDouble() + op.getFalseProbability().convertToDouble(), floatThresholdMap.size(), op.getThreshold().convertToFloat());
        op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, floatThresholdMap.size()));
        floatThresholdMap[op.getThreshold().convertToFloat()] = floatThresholdMap.size();
    }
    else {
        op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, floatThresholdMap[op.getThreshold().convertToFloat()]));
    }
    //floatThresholdMap[op.getThreshold().convertToFloat()] += op.getTrueProbability().convertToDouble() + op.getFalseProbability().convertToDouble();
    // floatThresholdTupleVector +=
}*/

int64_t convertToKey(int index, float threshold) {
    int64_t key = static_cast<int64_t>(index) << 32;
    key |= *reinterpret_cast<int*>(&threshold);
    return key;
}

void TreeFloatThresholdCompression(tree::FloatNodeOp &op, std::vector<probabilityIndexFloatThreshold> &floatThresholdTupleVector, std::unordered_map<int64_t, int> &treeIndexFloatThresholdMap, int treeIndex) {
    if (!treeIndexFloatThresholdMap.count(convertToKey(treeIndex, op.getThreshold().convertToFloat()))) {
        floatThresholdTupleVector.emplace_back(op.getTrueProbability().convertToDouble() + op.getFalseProbability().convertToDouble(), treeIndexFloatThresholdMap.size(), op.getThreshold().convertToFloat());
        op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, treeIndexFloatThresholdMap.size()));
        treeIndexFloatThresholdMap[convertToKey(treeIndex, op.getThreshold().convertToFloat())] = treeIndexFloatThresholdMap.size();
    }
    else {
        op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, treeIndexFloatThresholdMap[convertToKey(treeIndex, op.getThreshold().convertToFloat())]));
    }
    //floatThresholdMap[op.getThreshold().convertToFloat()] += op.getTrueProbability().convertToDouble() + op.getFalseProbability().convertToDouble();
    // floatThresholdTupleVector +=
}

class TreeFloatFeatureLoweringRewriter : public OpRewritePattern<tree::FloatFeatureOp> {
public:
    TreeFloatFeatureLoweringRewriter(MLIRContext *context, ModuleOp moduleOp)
        : OpRewritePattern<tree::FloatFeatureOp>(context), moduleOp(moduleOp) {};
    using OpRewritePattern<tree::FloatFeatureOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tree::FloatFeatureOp op, PatternRewriter &rewriter) const final {
        if (auto globalName = op->getAttrOfType<FlatSymbolRefAttr>("globalName")) {
            LLVM::GlobalOp globalOp = dyn_cast_or_null<LLVM::GlobalOp>(SymbolTable(moduleOp).lookup(globalName.getValue()));
            Value globalAddr = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), globalOp);
            rewriter.replaceOp(op, rewriter.create<LLVM::LoadOp>(op.getLoc(), rewriter.getF32Type(), globalAddr));
        }
        else {
            LLVMTypeConverter converter(getContext());
            //Value featureIdx = rewriter.create<arith::ConstantIntOp>(op.getLoc(), op.getFeatureIndex()->getSExtValue(), rewriter.getI32Type());
            Value featureIdx = rewriter.create<arith::ConstantIntOp>(op.getLoc(), op.getFeatureIndex()->getSExtValue(), 32);
            //Value featurePtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), converter.getPointerType(rewriter.getF32Type()), rewriter.getF32Type(), op.getFeatureAddr(), featureIdx);
            // Newer MLIR has to use opaque pointer (no pointer type), and then specify the type when using LLVM::GEPOp
            Value featurePtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), LLVM::LLVMPointerType::get(getContext()), rewriter.getF32Type(), op.getFeatureAddr(), featureIdx);
            rewriter.replaceOp(op, rewriter.create<LLVM::LoadOp>(op.getLoc(), rewriter.getF32Type(), featurePtr));
        }
        return success();
    }
private:
    ModuleOp moduleOp;
};

class TreeIntFeatureLoweringRewriter : public OpRewritePattern<tree::IntFeatureOp> {
public:
    TreeIntFeatureLoweringRewriter(MLIRContext *context, ModuleOp moduleOp)
        : OpRewritePattern<tree::IntFeatureOp>(context), moduleOp(moduleOp) {};
    using OpRewritePattern<tree::IntFeatureOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tree::IntFeatureOp op, PatternRewriter &rewriter) const final {
        if (auto globalName = op->getAttrOfType<FlatSymbolRefAttr>("globalName")) {
            LLVM::GlobalOp globalOp = dyn_cast_or_null<LLVM::GlobalOp>(SymbolTable(moduleOp).lookup(globalName.getValue()));
            Value globalAddr = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), globalOp);
            rewriter.replaceOp(op, rewriter.create<LLVM::LoadOp>(op.getLoc(), rewriter.getI32Type(), globalAddr));
        }
        else {
            LLVMTypeConverter converter(getContext());
            //Value featureIdx = rewriter.create<arith::ConstantIntOp>(op.getLoc(), op.getFeatureIndex()->getSExtValue(), rewriter.getI32Type());
            Value featureIdx = rewriter.create<arith::ConstantIntOp>(op.getLoc(), op.getFeatureIndex()->getSExtValue(), 32);
            //Value featurePtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), converter.getPointerType(rewriter.getI32Type()), rewriter.getI32Type(), op.getFeatureAddr(), featureIdx);
            Value featurePtr = rewriter.create<LLVM::GEPOp>(op.getLoc(), LLVM::LLVMPointerType::get(getContext()), rewriter.getI32Type(), op.getFeatureAddr(), featureIdx);
            rewriter.replaceOp(op, rewriter.create<LLVM::LoadOp>(op.getLoc(), rewriter.getI32Type(), featurePtr));
        }
        return success();
    }
private:
    ModuleOp moduleOp;
};

static Value findGetGlobalOp(Operation *op) {
    while (op && !dyn_cast<func::FuncOp>(op))
        op = op->getParentOp();
    func::FuncOp funcOp = dyn_cast<func::FuncOp>(op);
    assert(funcOp && "unexpected operation outside of a funcOp");
    memref::GetGlobalOp getGlobalOp;
    for (auto &subOp : funcOp.getBody().front()) {
        if (dyn_cast<memref::GetGlobalOp>(subOp)) {
            getGlobalOp = dyn_cast<memref::GetGlobalOp>(subOp);
            break;
        }
    }
    assert(getGlobalOp && "cannot find getGlobalOp");
    return getGlobalOp;
}

class TreeFloatNodeLoweringRewriter : public OpRewritePattern<tree::FloatNodeOp> {
public:
    TreeFloatNodeLoweringRewriter(MLIRContext *context, std::vector<probabilityIndexFloatThreshold>& floatThresholdTupleVector)
        : OpRewritePattern<tree::FloatNodeOp>(context), floatThresholdTupleVector(floatThresholdTupleVector) {}
    using OpRewritePattern<tree::FloatNodeOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tree::FloatNodeOp op, PatternRewriter &rewriter) const final {
        ValueRange nullList = {};
        Value threshold;
        if (0 <= op.getThresholdIndex().getSExtValue() && op.getThresholdIndex().getSExtValue() < static_cast<int64_t>(floatThresholdTupleVector.size())) {
            Value index = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(std::get<1>(floatThresholdTupleVector[op.getThresholdIndex().getSExtValue()])));
            threshold = rewriter.create<memref::LoadOp>(op.getLoc(), findGetGlobalOp(op), index);
        }
        else {
            threshold = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getF32Type(), rewriter.getF32FloatAttr(op.getThreshold().convertToFloat()));
        }
        Value condition = rewriter.create<arith::CmpFOp>(op.getLoc(), op.getPredicate(), op.getOperand(), threshold);
        rewriter.create<LLVM::CondBrOp>(op.getLoc(), condition, op.getSuccessor(0), nullList, op.getSuccessor(1), nullList);
        rewriter.eraseOp(op);
        return success();
    }
private:
    std::vector<probabilityIndexFloatThreshold>& floatThresholdTupleVector;
};

class TreeIntNodeLoweringRewriter : public OpRewritePattern<tree::IntNodeOp> {
public:
    TreeIntNodeLoweringRewriter(MLIRContext *context, std::vector<probabilityIndexIntThreshold>& intThresholdTupleVector)
        : OpRewritePattern<tree::IntNodeOp>(context), intThresholdTupleVector(intThresholdTupleVector) {}
    using OpRewritePattern<tree::IntNodeOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tree::IntNodeOp op, PatternRewriter &rewriter) const final {
        ValueRange nullList = {};
        Value threshold;
        if (0 <= op.getThresholdIndex().getSExtValue() && op.getThresholdIndex().getSExtValue() < static_cast<int64_t>(intThresholdTupleVector.size())) {
            Value index = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(std::get<1>(intThresholdTupleVector[op.getThresholdIndex().getSExtValue()])));
            threshold = rewriter.create<memref::LoadOp>(op.getLoc(), findGetGlobalOp(op), index);
        }
        else {
            //threshold = rewriter.create<arith::ConstantIntOp>(op.getLoc(), op.getThreshold(), rewriter.getI32Type());
            threshold = rewriter.create<arith::ConstantIntOp>(op.getLoc(), op.getThreshold(), 32);
        }
        Value condition = rewriter.create<arith::CmpIOp>(op.getLoc(), op.getPredicate(), op.getOperand(), threshold);
        rewriter.create<LLVM::CondBrOp>(op.getLoc(), condition, op.getSuccessor(0), nullList, op.getSuccessor(1), nullList);
        rewriter.eraseOp(op);
        return success();
    }
private:
    std::vector<probabilityIndexIntThreshold>& intThresholdTupleVector;
};

class TreeLowering : public impl::TreeLoweringBase<TreeLowering> {
public:
    using impl::TreeLoweringBase<TreeLowering>::TreeLoweringBase;
    void runOnOperation() final {
        std::vector<probabilityIndexFloatThreshold> floatThresholdTupleVector;
        std::vector<probabilityIndexIntThreshold> intThresholdTupleVector;
        //std::unordered_map<float, double> floatThresholdMap;
        //std::unordered_map<float, int> floatThresholdMap;
        std::unordered_map<int64_t, int> treeIndexFloatThresholdMap;
        memref::GlobalOp floatThresholdOp;
        memref::GlobalOp intThresholdOp;
        /*if (thresholdOrdering) {
            getOperation().walk([&](Operation *op) {
                if (tree::FloatNodeOp nodeOp = dyn_cast<tree::FloatNodeOp>(op)) {
                    TreeFloatThresholdOrdering(nodeOp, floatThresholdTupleVector);
                }
                else if (tree::IntNodeOp nodeOp = dyn_cast<tree::IntNodeOp>(op)) {
                    TreeIntThresholdOrdering(nodeOp, intThresholdTupleVector);
                }
            });
            std::sort(floatThresholdTupleVector.begin(), floatThresholdTupleVector.end(), std::greater<>());
            std::sort(intThresholdTupleVector.begin(), intThresholdTupleVector.end(), std::greater<>());
            TreeVectorInvertIndex(floatThresholdTupleVector, intThresholdTupleVector);
            TreeBuildGlobal(getOperation(), floatThresholdTupleVector, intThresholdTupleVector);
            getOperation().walk([&](Operation *op) {
                if (func::FuncOp funcOp = dyn_cast<func::FuncOp>(op)) {
                    TreeBuildGetGlobal(funcOp, floatThresholdTupleVector, intThresholdTupleVector);
                }
            });
        }*/
        if (thresholdOrdering) {
            int treeIndex = 0;
            getOperation().walk([&](func::FuncOp funcOp) {
                funcOp->walk([&](tree::FloatNodeOp nodeOp) {
                    TreeFloatThresholdCompression(nodeOp, floatThresholdTupleVector, treeIndexFloatThresholdMap, treeIndex);
                });
                ++treeIndex;
            });
            /*
            getOperation().walk([&](Operation *op) {
                if (tree::FloatNodeOp nodeOp = dyn_cast<tree::FloatNodeOp>(op)) {
                    TreeFloatThresholdCompression(nodeOp, floatThresholdTupleVector, floatThresholdMap);
                }
            });
            */
            /*vector<double> probabilityVector;
            probabilityVector.reserve(floatThresholdMap.size());
            for (auto [threshold, probability] : floatThresholdMap) {
                floatThresholdTupleVector.emplace_back(probability, 0, threshold);
                probabilityVector.push_back(probability);
            }
            std::sort(floatThresholdTupleVector.begin(), floatThresholdTupleVector.end(), std::greater<>());
            std::unordered_map<float, int> thresholdIndexMap;
            for (int index = 0; index < floatThresholdTupleVector.size(); ++index) {
                thresholdIndexMap[std::get<2>(floatThresholdTupleVector[index])] = index;
            }*/
            TreeBuildGlobal(getOperation(), floatThresholdTupleVector, intThresholdTupleVector);
            treeIndex = 0;
            getOperation().walk([&](func::FuncOp funcOp) {
                TreeBuildGetGlobal(funcOp, floatThresholdTupleVector, intThresholdTupleVector);
            });
        }
        mlir::ConversionTarget target(getContext());
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        //target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect>();
        target.addIllegalDialect<TreeDialect>();
        RewritePatternSet patterns(&getContext());
        patterns.add<TreeFloatFeatureLoweringRewriter>(&getContext(), getOperation());
        patterns.add<TreeIntFeatureLoweringRewriter>(&getContext(), getOperation());
        patterns.add<TreeFloatNodeLoweringRewriter>(&getContext(), floatThresholdTupleVector);
        patterns.add<TreeIntNodeLoweringRewriter>(&getContext(), intThresholdTupleVector);
        FrozenRewritePatternSet patternSet(std::move(patterns));
        if (failed(applyFullConversion(getOperation(), target, patternSet))) {
            signalPassFailure();
        }
    }
};
} // namespace
} // namespace mlir::tree
