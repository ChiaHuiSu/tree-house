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
#include "mlir/Analysis/SliceAnalysis.h"

#include <tuple>
#include <vector>
#include <algorithm>
#include <set>
#include <fstream>

#include "Tree/TreePasses.h"

namespace mlir::tree {
//#define GEN_PASS_DEF_TREESWITCHBARFOO
#define GEN_PASS_DEF_TREEREORDER
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
// TreeReorder
class TreeReorder : public impl::TreeReorderBase<TreeReorder> {
public:
    using impl::TreeReorderBase<TreeReorder>::TreeReorderBase;
    //std::vector<std::vector<double>> distributions;
    std::unordered_map<int, size_t> thresholdToIndex;
    //std::vector<std::map<size_t, double>> distributions;
    std::vector<std::map<size_t, double>> thresholdDistributions;
    std::vector<std::map<size_t, double>> featureDistributions;
    void runOnOperation() final {
        assert((useFeature || useThreshold) && "At least one of useFeature and useThreshold should be enabled");
        getOperation().walk([&](func::FuncOp funcOp) {
            if (funcOp.getSymName().str().substr(0, 5) != "tree_")
                return;
            const int treeIndex = stoi(funcOp.getSymName().str().substr(5));
            if (treeIndex < 0)
                return;
            if ((int)thresholdDistributions.size() <= treeIndex)
                thresholdDistributions.resize(treeIndex + 1);
            if ((int)featureDistributions.size() <= treeIndex)
                featureDistributions.resize(treeIndex + 1);
            // Expected number of each feature access
            //std::vector<double>& distribution = distributions[treeIndex];
            //std::map<size_t, double>& distribution = distributions[treeIndex];
            funcOp.walk([&](tree::FloatNodeOp nodeOp) {
                if (useThreshold) {
                    union {
                        float f32;
                        int i32;
                    } threshold;
                    threshold.f32 = nodeOp.getThreshold().convertToFloat();
                    if (isFLIntEnabled && threshold.f32 < 0) {
                        threshold.f32 = -threshold.f32;
                    }
                    if (!thresholdToIndex.count(threshold.i32)) {
                        thresholdToIndex[threshold.i32] = thresholdToIndex.size();
                    }
                    thresholdDistributions[treeIndex][thresholdToIndex[threshold.i32]] += nodeOp.getTrueProbability().convertToDouble() + nodeOp.getFalseProbability().convertToDouble();
                }
                if (useFeature)
                    if (tree::FloatFeatureOp featureOp = nodeOp.getFeature().getDefiningOp<tree::FloatFeatureOp>()) {
                        const size_t featureIndex = featureOp.getFeatureIndex()->getZExtValue();
                        featureDistributions[treeIndex][featureIndex] += nodeOp.getTrueProbability().convertToDouble() + nodeOp.getFalseProbability().convertToDouble();
                    }
            });
            funcOp.walk([&](tree::IntNodeOp nodeOp) {
                if (useThreshold) {
                    int threshold = nodeOp.getThreshold();
                    if (!thresholdToIndex.count(threshold)) {
                        thresholdToIndex[threshold] = thresholdToIndex.size();
                    }
                    thresholdDistributions[treeIndex][thresholdToIndex[threshold]] += nodeOp.getTrueProbability().convertToDouble() + nodeOp.getFalseProbability().convertToDouble();
                }
                if (useFeature)
                    if (tree::IntFeatureOp featureOp = nodeOp.getFeature().getDefiningOp<tree::IntFeatureOp>()) {
                        const size_t featureIndex = featureOp.getFeatureIndex()->getZExtValue();
                        featureDistributions[treeIndex][featureIndex] += nodeOp.getTrueProbability().convertToDouble() + nodeOp.getFalseProbability().convertToDouble();
                    }
            });
            // L2 Normalization
            double squareSum = 0;
            if (useThreshold)
                for (auto [_, probability] : thresholdDistributions[treeIndex])
                    squareSum += probability * probability;
            if (useFeature)
                for (auto [_, probability] : featureDistributions[treeIndex])
                    squareSum += probability * probability;
            const double length = std::sqrt(squareSum);
            if (length) {
                if (useThreshold)
                    for (auto& [_, probability] : thresholdDistributions[treeIndex])
                        probability /= length;
                if (useFeature)
                    for (auto& [_, probability] : featureDistributions[treeIndex])
                        probability /= length;
            }
        });
        if ((useThreshold && thresholdDistributions.empty()) || (useFeature && featureDistributions.empty())) {
            return;
        }
        // Find multiple call-fadd-store chains for xgboost.
        // Find call chain for sklearn.
        getOperation().walk([&](func::FuncOp funcOp) {
            if (funcOp.getSymName().str().substr(0, 7) != "predict") {
                return;
            }
            DenseSet<size_t> visitedTreeIndices;
            funcOp.walk([&](func::CallOp callOp) {
                if (callOp.getCallee().str().substr(0, 5) != "tree_") {
                    return;
                }
                const int treeIndex = stoi(callOp.getCallee().str().substr(5));
                if (treeIndex < 0 || visitedTreeIndices.count(treeIndex)) {
                    return;
                }
                SetVector<Operation*> slice;
                getForwardSlice(callOp, &slice);
                std::vector<size_t> treeIndices;
                std::vector<func::CallOp> callOps;
                for (Operation* op : slice) { // XGBoost
                    if (arith::AddFOp addFOp = dyn_cast<arith::AddFOp>(*op)) {
                        if (func::CallOp chainedCallOp = addFOp.getLhs().getDefiningOp<func::CallOp>()) {
                            if (chainedCallOp.getCallee().str().substr(0, 5) != "tree_") {
                                break;
                            }
                            const int chainedTreeIndex = stoi(chainedCallOp.getCallee().str().substr(5));
                            if (chainedTreeIndex < 0) {
                                break;
                            }
                            treeIndices.push_back(chainedTreeIndex);
                            visitedTreeIndices.insert(chainedTreeIndex);
                            callOps.push_back(chainedCallOp);
                        }
                        if (func::CallOp chainedCallOp = addFOp.getRhs().getDefiningOp<func::CallOp>()) {
                            if (chainedCallOp.getCallee().str().substr(0, 5) != "tree_") {
                                break;
                            }
                            const int chainedTreeIndex = stoi(chainedCallOp.getCallee().str().substr(5));
                            if (chainedTreeIndex < 0) {
                                break;
                            }
                            treeIndices.push_back(chainedTreeIndex);
                            visitedTreeIndices.insert(chainedTreeIndex);
                            callOps.push_back(chainedCallOp);
                        }
                    }
                }
                if (treeIndices.empty()) { // sklearn
                    funcOp.walk([&](func::CallOp chainedCallOp) {
                        if (chainedCallOp.getNumResults()) {
                            return;
                        }
                        if (chainedCallOp.getCallee().str().substr(0, 5) != "tree_") {
                            return;
                        }
                        const int chainedTreeIndex = stoi(chainedCallOp.getCallee().str().substr(5));
                        if (chainedTreeIndex < 0 || visitedTreeIndices.count(chainedTreeIndex)) {
                            return;
                        }
                        treeIndices.push_back(chainedTreeIndex);
                        visitedTreeIndices.insert(chainedTreeIndex);
                        callOps.push_back(chainedCallOp);
                    });
                }
                std::vector<size_t> tspIndexToTreeIndex;
                std::ofstream fout;
                // Write .par
                fout.open("reorder.par", std::fstream::out);
                fout << "PROBLEM_FILE = reorder.tsp\n";
                fout << "TOUR_FILE = reorder.tour\n";
                //fout << "OPTIMUM = 1\n";
                //fout << "INITIAL_PERIOD = 1000\n";
                fout << "MAX_TRIALS = 1000\n";
                fout << "POPULATION_SIZE = 3\n";
                fout << "RECOMBINATION = CLARIST\n";
                fout << "RUNS = 1\n";
                fout << "TIME_LIMIT = " << 10 + treeIndices.size() / 10 << '\n';
                fout.close();
                // Write .tsp
                fout.open("reorder.tsp", std::fstream::out);
                fout << "NAME: reorder.tsp\n";
                //fout << "TYPE: HPP\n";
                fout << "TYPE: TSP\n";
                fout << "DIMENSION: " << treeIndices.size() + 1 << '\n';
                fout << "EDGE_WEIGHT_TYPE: SPECIAL\n";
                fout << "NODE_COORD_TYPE: TWOD_COORDS\n";
                fout << "NODE_COORD_SECTION\n";
                for (size_t index = 0; index < treeIndices.size(); ++index) {
                    fout << index + 1 << " 0 0\n";
                    tspIndexToTreeIndex.push_back(treeIndices[index]);
                }
                // Dummy node
                fout << treeIndices.size() + 1 << " 0 0\n";
                fout << "EOF\n";
                fout.close();
                // Write custom nodes for LKH-2.x.x/SRC/Distance_SPECIAL.c
                fout.open("distribution.txt", std::fstream::out);
                fout << treeIndices.size() << '\n';
                for (int index : treeIndices) {
                    //std::map<size_t, double>& distribution = distributions[index];
                    int size = 0;
                    if (useThreshold)
                        size += thresholdDistributions[index].size();
                    if (useFeature)
                        size += featureDistributions[index].size();
                    fout << size;
                    //fout << distribution.size();
                    if (useThreshold)
                        for (auto [index, component] : thresholdDistributions[index])
                            fout << ' ' << index << ' ' << component;
                    if (useFeature)
                        for (auto [index, component] : featureDistributions[index])
                            fout << ' ' << 0x40000000 + index << ' ' << component;
                    fout << " -1 -1 \n";
                }
                fout.close();
                // Run LKH
                //std::system("./LKH reorder.par");
                std::system("./LKH reorder.par > /dev/null 2>&1");
                // Read result
                std::ifstream fin;
                fin.open("reorder.tour", std::fstream::in);
                for (std::string st; !fin.eof() && st != "TOUR_SECTION";) {
                    fin >> st;
                }
                std::vector<size_t> indices;
                size_t dummyIndex = 0;
                for (int index; !fin.eof();) {
                    fin >> index;
                    if (index < 0) {
                        break;
                    }
                    else if ((size_t)index == treeIndices.size() + 1) {
                        dummyIndex = indices.size();
                    }
                    indices.push_back(index - 1);
                }
                // Hamiltonian path = Hamiltonian cycle - dummy node
                std::rotate(indices.begin(), indices.begin() + dummyIndex + 1, indices.end());
                indices.pop_back();
                // Give reversed indices to moveBefore to get forward callOp order
                // Reversed callOp order should also be fine since the path length is the same
                std::reverse(indices.begin(), indices.end());
                for (size_t index = 0; index < indices.size(); ++index) {
                    MLIRContext *context = callOps[index]->getContext();
                    //llvm::outs() << indices.size() << ' ' << tspIndexToTreeIndex.size() << ' ' << callOps.size() << ' ' << index << ' ' << indices[index] << ' ' << tspIndexToTreeIndex[indices[index]] << '\n';
                    callOps[index].setCalleeAttr(SymbolRefAttr::get(context, std::string("tree_") + std::to_string(tspIndexToTreeIndex[indices[index]])));
                }
            });
        });
        // Reorder func::FuncOp
        std::vector<func::FuncOp> funcOps;
        std::vector<size_t> treeIndices;
        getOperation().walk([&](func::FuncOp funcOp) {
            if (funcOp.getSymName().str().substr(0, 7) != "predict") { // Record funcOp
                if (funcOp.getSymName().str().substr(0, 5) != "tree_") {
                    return;
                }
                const int chainedTreeIndex = stoi(funcOp.getSymName().str().substr(5));
                if (chainedTreeIndex < 0) {
                    return;
                }
                funcOps.push_back(funcOp);
            }
            funcOp.walk([&](func::CallOp callOp) { // Record callOp order
                if (callOp.getCallee().str().substr(0, 5) != "tree_") {
                    return;
                }
                const int chainedTreeIndex = stoi(callOp.getCallee().str().substr(5));
                if (chainedTreeIndex < 0) {
                    return;
                }
                treeIndices.push_back(chainedTreeIndex);
            });
        });
        for (size_t index = 1; index < treeIndices.size(); ++index) { // Reorder funcOp
            func::FuncOp previousFuncOp = funcOps[treeIndices[index - 1]];
            func::FuncOp currentFuncOp = funcOps[treeIndices[index]];
            currentFuncOp->moveAfter(previousFuncOp);
        }
        /*std::ofstream fout;
        // Write .par
        fout.open("reorder.par", std::fstream::out);
        fout << "PROBLEM_FILE = reorder.tsp\n";
        fout << "TOUR_FILE = reorder.tour\n";
        //fout << "OPTIMUM = 1\n";
        //fout << "INITIAL_PERIOD = 1000\n";
        fout << "MAX_TRIALS = 1000\n";
        fout << "POPULATION_SIZE = 3\n";
        fout << "RECOMBINATION = CLARIST\n";
        fout << "RUNS = 1\n";
        fout << "TIME_LIMIT = " << 10 + distributions.size() / 10 << '\n';
        fout.close();
        // Write .tsp
        fout.open("reorder.tsp", std::fstream::out);
        fout << "NAME: reorder.tsp\n";
        //fout << "TYPE: HPP\n";
        fout << "TYPE: TSP\n";
        fout << "DIMENSION: " << distributions.size() + 1 << '\n';
        fout << "EDGE_WEIGHT_TYPE: SPECIAL\n";
        fout << "NODE_COORD_TYPE: TWOD_COORDS\n";
        fout << "NODE_COORD_SECTION\n";
        for (size_t index = 0; index < distributions.size(); ++index) {
            fout << index + 1 << " 0 0\n";
        }
        fout << distributions.size() + 1 << " 0 0\n";
        fout << "EOF\n";
        fout.close();
        // Write custom nodes for LKH-2.x.x/SRC/Distance_SPECIAL.c
        fout.open("distribution.txt", std::fstream::out);
        fout << distributions.size() << ' ' << distributions[0].size() << '\n';
        for (std::vector<double>& distribution : distributions) {
            for (double probability : distribution) {
                fout << probability << ' ';
            }
            for (int padding = distributions[0].size() - distribution.size(); padding > 0; --padding) {
                fout << "0 ";
            }
            fout << '\n';
        }
        fout.close();
        // Run LKH
        std::system("./LKH reorder.par");
        // Read result
        std::ifstream fin;
        fin.open("reorder.tour", std::fstream::in);
        for (std::string st; !fin.eof() && st != "TOUR_SECTION";) {
            fin >> st;
        }
        std::vector<size_t> indices;
        size_t dummyIndex = 0;
        for (int index; !fin.eof();) {
            fin >> index;
            if (index < 0) {
                break;
            }
            else if ((size_t)index == distributions.size() + 1) {
                dummyIndex = indices.size();
            }
            indices.push_back(index - 1);
        }
        // Hamiltonian path = Hamiltonian cycle - dummy node
        std::rotate(indices.begin(), indices.begin() + dummyIndex + 1, indices.end());
        indices.resize(distributions.size());
        // Give reversed indices to moveBefore to get forward callOp order
        // Reversed callOp order should also be fine since the path length is the same
        std::reverse(indices.begin(), indices.end());
        // Record callOps
        std::vector<func::CallOp> callOps;
        callOps.resize(distributions.size());
        getOperation().walk([&](func::FuncOp funcOp) {
            if (funcOp.getSymName().str().substr(0, 7) != "predict") {
                return;
            }
            funcOp.walk([&](func::CallOp callOp) {
                if (callOp.getCallee().str().substr(0, 5) != "tree_") {
                    return;
                }
                const int treeIndex = stoi(callOp.getCallee().str().substr(5));
                if (treeIndex < 0) {
                    return;
                }
                callOps[treeIndex] = callOp;
            });
        });
        if (callOps.empty()) {
            return;
        }
        // Reorder callOps
        if (Value acc = dyn_cast<arith::ConstantOp>(callOps[0]->getBlock()->front()); acc && reorderAccumulation) {
            // Accumulate in predict function
            arith::AddFOp addFOp = dyn_cast<arith::AddFOp>(callOps.back()->getNextNode());
            if (!addFOp) {
                return;
            }
            OpBuilder builder(addFOp);
            for (size_t index : indices) {
                func::CallOp callOp = callOps[index];
                func::CallOp newCallOp = builder.create<func::CallOp>(
                    callOp.getLoc(),
                    callOp.getCallee(),
                    callOp.getResultTypes(),
                    callOp.getOperands()
                );
                acc = builder.create<arith::AddFOp>(callOp.getLoc(), acc, newCallOp.getResult(0));
            }
            addFOp.getResult().replaceAllUsesWith(acc);
            for (func::CallOp callOp : llvm::reverse(callOps)) {
                for (auto &use : callOp.getResult(0).getUses()) {
                    if (arith::AddFOp addFOp = dyn_cast<arith::AddFOp>(use.getOwner())) {
                        addFOp->erase();
                    }
                }
                callOp.erase();
            }
        }
        else {
            // Accumulate in tree functions
            for (size_t index : indices) {
                Block *block = callOps[index]->getBlock();
                if (!block || callOps[index] == &block->front())
                    continue;
                callOps[index]->moveBefore(&block->front());
            }
        }*/
    }
};
// TreeInline
void RegisterAllocation(size_t pinningAmount, ModuleOp moduleOp, MLIRContext* context) {
    assert(1 <= pinningAmount && pinningAmount <= 32 && "pinningAmount must be between 1~32");
    func::FuncOp funcOp = moduleOp.lookupSymbol<func::FuncOp>("predict");
    bool isFloat = true;
    Value featureAddr = funcOp.getArgument(0);
    SmallVector<double> probabilities;
    funcOp.walk([&](Operation *op) {
        if (tree::FloatNodeOp nodeOp = dyn_cast<tree::FloatNodeOp>(op)) {
            if (tree::FloatFeatureOp featureOp = nodeOp.getFeature().getDefiningOp<tree::FloatFeatureOp>()) {
                //llvm::outs() << featureAddr << '\n';
                //llvm::outs() << featureOp.getFeatureAddr() << '\n';
                //assert(!featureAddr || featureAddr == featureOp.getFeatureAddr() && "feature address mismatch");
                //featureAddr = featureOp.getFeatureAddr();
                const size_t featureIndex = featureOp.getFeatureIndex()->getZExtValue();
                const double probability = nodeOp.getTrueProbability().convertToDouble() + nodeOp.getFalseProbability().convertToDouble();
                if (probabilities.size() <= featureIndex) {
                    probabilities.resize(featureIndex + 1);
                }
                probabilities[featureIndex] += probability;
            }
        }
        else if (tree::IntNodeOp nodeOp = dyn_cast<tree::IntNodeOp>(op)) {
            if (tree::IntFeatureOp featureOp = nodeOp.getFeature().getDefiningOp<tree::IntFeatureOp>()) {
                //assert(!featureAddr || featureAddr == featureOp.getFeatureAddr() && "feature address mismatch");
                isFloat = false;
                //featureAddr = featureOp.getFeatureAddr();
                const size_t featureIndex = featureOp.getFeatureIndex()->getZExtValue();
                const double probability = nodeOp.getTrueProbability().convertToDouble() + nodeOp.getFalseProbability().convertToDouble();
                if (probabilities.size() <= featureIndex) {
                    probabilities.resize(featureIndex + 1);
                }
                probabilities[featureIndex] += probability;
            }
        }
    });
    typedef std::pair<int, double> ipPair;
    SmallVector<ipPair> indexProbabilities;
    indexProbabilities.reserve(probabilities.size());
    for (size_t index = 0; index < probabilities.size(); ++index) {
        indexProbabilities.push_back({index, probabilities[index]});
    }
    std::sort(indexProbabilities.begin(), indexProbabilities.end(), [](ipPair& lhs, ipPair& rhs) {
        return lhs.second > rhs.second;
    });
    //SmallVector<LLVM::LoadOp> loadOps(probabilities.size());
    SmallVector<tree::FloatFeatureOp> floatFeatureOps;
    SmallVector<tree::IntFeatureOp> intFeatureOps;
    if (isFloat) {
        floatFeatureOps.resize(probabilities.size());
    }
    else {
        intFeatureOps.resize(probabilities.size());
    }
    BitVector pinnedIndices(probabilities.size());
    indexProbabilities.resize(std::min(pinningAmount, indexProbabilities.size()));
    OpBuilder builder(context);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    auto loc = builder.getUnknownLoc();
    for (auto [index, _] : indexProbabilities) {
        pinnedIndices[index] = true;
        //Value featureIndex = builder.create<arith::ConstantIntOp>(loc, index, 32);
        if (isFloat) {
            //Value featurePtr = builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(context), builder.getF32Type(), featureAddr, featureIdx);
            //loadOps[index] = builder.create<LLVM::LoadOp>(loc, builder.getF32Type(), featurePtr);
            floatFeatureOps[index] = builder.create<tree::FloatFeatureOp>(loc, featureAddr, index);
        }
        else {
            //Value featurePtr = builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(context), builder.getI32Type(), featureAddr, featureIdx);
            //loadOps[index] = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(), featurePtr);
            intFeatureOps[index] = builder.create<tree::IntFeatureOp>(loc, featureAddr, index);
        }
    }
    funcOp.walk([&](Operation *op) {
        if (tree::FloatNodeOp nodeOp = dyn_cast<tree::FloatNodeOp>(op)) {
            if (tree::FloatFeatureOp featureOp = nodeOp.getFeature().getDefiningOp<tree::FloatFeatureOp>()) {
                int featureIndex = featureOp.getFeatureIndex()->getZExtValue();
                if (!pinnedIndices[featureIndex]) {
                    return;
                }
                featureOp.replaceAllUsesWith(floatFeatureOps[featureIndex].getOperation());
                featureOp.erase();
            }
        }
        else if (tree::IntNodeOp nodeOp = dyn_cast<tree::IntNodeOp>(op)) {
            if (tree::IntFeatureOp featureOp = nodeOp.getFeature().getDefiningOp<tree::IntFeatureOp>()) {
                int featureIndex = featureOp.getFeatureIndex()->getZExtValue();
                if (!pinnedIndices[featureIndex]) {
                    return;
                }
                featureOp.replaceAllUsesWith(intFeatureOps[featureIndex].getOperation());
                featureOp.erase();
            }
        }
    });
}

class TreeInline : public impl::TreeInlineBase<TreeInline> {
public:
    using impl::TreeInlineBase<TreeInline>::TreeInlineBase;
    void runOnOperation() final {
        ModuleOp moduleOp = getOperation();
        /*OpBuilder builder(moduleOp.getContext());
        SymbolTable symbolTable(moduleOp);
        std::vector<func::CallOp> callOps;
        std::vector<Block*> callOpBlocks;
        llvm::outs() << callOps.size() << ' '<< callOpBlocks.size() << '\n';
        moduleOp.walk([&](func::CallOp callOp) {
            callOps.push_back(callOp);
        });
        for (func::CallOp callOp : callOps) {
            callOpBlocks.push_back(callOp.getOperation()->getBlock()->splitBlock(callOp));
        }
        llvm::outs() << callOps.size() << ' '<< callOpBlocks.size() << '\n';
        if (Operation *next = callOps.back()->getNextNode()) {
            callOpBlocks.push_back(callOps.back()->getBlock()->splitBlock(next));
        }
        else {
            Block *block = callOps.back()->getBlock();
            Block *newBlock = new Block();
            block->getParent()->getBlocks().insertAfter(block->getIterator(), newBlock);
            callOpBlocks.push_back(newBlock);
        }
        llvm::outs() << callOps.size() << ' '<< callOpBlocks.size() << '\n';
        for (size_t callIndex = 0; callIndex < callOps.size(); ++callIndex) {
            func::CallOp callOp = callOps[callIndex];
            auto funcOp = symbolTable.lookup<func::FuncOp>(callOp.getCallee());
            // No such funcOp or only a declaration without implementation
            if (!funcOp || funcOp.isDeclaration()) {
                return;
            }
            if (!callIndex) {
                builder.setInsertionPointToEnd(callOpBlocks[0]->getPrevNode());
                builder.create<cf::BranchOp>(callOps[0].getLoc(), callOpBlocks[0], mappedOperands);
            }
            IRMapping mapper;
            for (auto [argument, operand] : zip(funcOp.getArguments(), callOp.getOperands())) {
                mapper.map(argument, operand);
            }
            //Block *callBlock = callOp.getOperation()->getBlock();
            //Block *afterCallBlock = callBlock->splitBlock(callOp);
            Block *callBlock = callOpBlocks[callIndex];
            Block *afterCallBlock = callOpBlocks[callIndex + 1];
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
                        //builder.setInsertionPointToEnd(newBlock);
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
            //builder.create<cf::BranchOp>(callOp.getLoc(), afterCallBlock);
            callOp.erase();
            funcOp.erase();
            //break;
        }*/
        if (pinningAmount) {
            RegisterAllocation(pinningAmount, moduleOp, &getContext());
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
        Block* formerTrueBlock = op.getSuccessor(0);
        Block* formerFalseBlock = op.getSuccessor(1);
        op.setSuccessor(formerFalseBlock, 0);
        op.setSuccessor(formerTrueBlock, 1);
        //op->getBlock()->moveBefore(formerFalseBlock);
        /*Region *region = op->getBlock()->getParent();
        region->getBlocks().splice(std::next(Region::iterator(op->getBlock())),
                                   region->getBlocks(),
                                   Region::iterator(formerFalseBlock));*/
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
        Block* formerTrueBlock = op.getSuccessor(0);
        Block* formerFalseBlock = op.getSuccessor(1);
        op.setSuccessor(formerFalseBlock, 0);
        op.setSuccessor(formerTrueBlock, 1);
    }
}

void dfsBasicBlocks(Block *block, SmallPtrSet<Block*, 32>& visited, SmallVector<Block*>& order) {
    if (!visited.insert(block).second) {
        return;
    }
    order.push_back(block);
    if (tree::FloatNodeOp floatNodeOp = dyn_cast<tree::FloatNodeOp>(block->getTerminator())) {
        dfsBasicBlocks(floatNodeOp.getSuccessor(0), visited, order);
        dfsBasicBlocks(floatNodeOp.getSuccessor(1), visited, order);
    }
    else if (tree::IntNodeOp intNodeOp = dyn_cast<tree::IntNodeOp>(block->getTerminator())) {
        dfsBasicBlocks(intNodeOp.getSuccessor(0), visited, order);
        dfsBasicBlocks(intNodeOp.getSuccessor(1), visited, order);
    }
}

void reorderBlocks(Region &region, SmallVector<Block*>& order) {
    llvm::iplist<Block> &blocks = region.getBlocks();
    for (Block *block : llvm::reverse(order)) {
        blocks.splice(blocks.begin(),
                      blocks,
                      Region::iterator(block));
    }
}

class TreeSwap : public impl::TreeSwapBase<TreeSwap> {
public:
    using impl::TreeSwapBase<TreeSwap>::TreeSwapBase;
    void runOnOperation() final {
        // Modify predicates and successors
        getOperation().walk([](Operation *op) {
            if (tree::FloatNodeOp nodeOp = dyn_cast<tree::FloatNodeOp>(op)) {
                treeSwapFloat(nodeOp);
            }
            else if (tree::IntNodeOp nodeOp = dyn_cast<tree::IntNodeOp>(op)) {
                treeSwapInt(nodeOp);
            }
        });
        // Reorder basic blocks
        getOperation().walk([](func::FuncOp funcOp) {
            Region &region = funcOp.getBody();
            SmallPtrSet<Block*, 32> visited;
            SmallVector<Block*> order;
            for (Block &block : region) {
                dfsBasicBlocks(&block, visited, order);
            }
            reorderBlocks(region, order);
        });
    }
};
// TreeFLInt
class TreeNodeFLIntRewriter : public OpRewritePattern<tree::FloatNodeOp> {
public:
    TreeNodeFLIntRewriter(MLIRContext *context)
        : OpRewritePattern<tree::FloatNodeOp>(context) {};
    using OpRewritePattern<tree::FloatNodeOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tree::FloatNodeOp op, PatternRewriter &rewriter) const final {
        //llvm::errs() << "Float feature: " << floatFeature << "\n";
        Value intFeature;
        if (tree::FloatFeatureOp floatFeature = op.getFeature().getDefiningOp<tree::FloatFeatureOp>()) {
            IRRewriter::InsertPoint insertPoint = rewriter.saveInsertionPoint();
            rewriter.setInsertionPointAfter(floatFeature);
            if (auto globalName = floatFeature->getAttrOfType<FlatSymbolRefAttr>("globalName")) {
                //LLVM::GlobalOp globalOp = dyn_cast_or_null<LLVM::GlobalOp>(SymbolTable(moduleOp).lookup(globalName.getValue()));
                LLVM::GlobalOp globalOp = dyn_cast_or_null<LLVM::GlobalOp>(SymbolTable::lookupNearestSymbolFrom(op, globalName));
                tree::IntFeatureOp intFeatureOp = rewriter.create<tree::IntFeatureOp>(floatFeature.getLoc(), floatFeature.getFeatureAddr(), floatFeature.getFeatureIndex()->getSExtValue(), globalOp);
                rewriter.replaceOp(floatFeature, intFeatureOp.getResult());
                intFeature = intFeatureOp;
                //intFeature = rewriter.replaceOpWithNewOp<tree::IntFeatureOp>(floatFeature, floatFeature.getFeatureAddr(), rewriter.getI32IntegerAttr(floatFeature.getFeatureIndex()->getSExtValue()), globalName);
            }
            else {
                auto intFeatureOp = rewriter.create<tree::IntFeatureOp>(floatFeature.getLoc(), floatFeature.getFeatureAddr(), floatFeature.getFeatureIndex()->getSExtValue());
                rewriter.replaceOp(floatFeature, intFeatureOp.getResult());
                intFeature = intFeatureOp;
                //intFeature = rewriter.replaceOpWithNewOp<tree::IntFeatureOp>(floatFeature, floatFeature.getFeatureAddr(), floatFeature.getFeatureIndex()->getSExtValue());
            }
            rewriter.restoreInsertionPoint(insertPoint);
        }
        else if (tree::IntFeatureOp intFeatureOp = op.getFeature().getDefiningOp<tree::IntFeatureOp>()) {
            llvm::outs() << "int\n";
            intFeature = intFeatureOp;
        }
        else if (LLVM::LoadOp loadOp = op.getFeature().getDefiningOp<LLVM::LoadOp>()) { // Deal with inline + RA
            llvm::outs() << "else if\n";
            //intFeature = rewriter.create<LLVM::LoadOp>(op.getFeature().getLoc(), rewriter.getI32Type(), where is the pointer);
            if (LLVM::GEPOp gepOp = loadOp.getAddr().getDefiningOp<LLVM::GEPOp>()) {
                SmallVector<LLVM::GEPArg> indices;
                for (auto index : gepOp.getIndices()) {
                    if (auto intAttr = index.dyn_cast<IntegerAttr>()) {
                        indices.push_back(intAttr.getInt());
                    }
                    else {
                        //
                    }
                }
                IRRewriter::InsertPoint insertPoint = rewriter.saveInsertionPoint();
                rewriter.setInsertionPointAfter(loadOp);
                Value featurePtr = rewriter.create<LLVM::GEPOp>(gepOp.getLoc(), LLVM::LLVMPointerType::get(gepOp->getContext()), rewriter.getI32Type(), gepOp.getBase(), indices);
                intFeature = rewriter.create<LLVM::LoadOp>(loadOp.getLoc(), rewriter.getI32Type(), featurePtr);
                //rewriter.replaceAllOpUsesWith(loadOp, intFeature);
                rewriter.restoreInsertionPoint(insertPoint);
            }
        }
        else {
            llvm::outs() << "else\n";
            //intFeature = op.getFeature();
            intFeature = rewriter.create<arith::BitcastOp>(op.getFeature().getLoc(), rewriter.getI32Type(), op.getFeature());
        }
        float threshold = op.getThreshold().convertToFloat();
        arith::CmpIPredicate compare = arith::CmpIPredicate::slt;
        if (op.getPredicate() == arith::CmpFPredicate::OLT) {
            compare = threshold < 0 ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::slt;
        }
        else if (op.getPredicate() == arith::CmpFPredicate::OLE) {
            compare = threshold < 0 ? arith::CmpIPredicate::sge : arith::CmpIPredicate::sle;
        }
        else if (op.getPredicate() == arith::CmpFPredicate::OGT) {
            compare = threshold < 0 ? arith::CmpIPredicate::slt : arith::CmpIPredicate::sgt;
        }
        else if (op.getPredicate() == arith::CmpFPredicate::OGE) {
            compare = threshold < 0 ? arith::CmpIPredicate::sle : arith::CmpIPredicate::sge;
        }
        if (threshold < 0) {
            //Value mask = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0x80000000, rewriter.getI32Type());
            Value mask = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0x80000000, 32);
            intFeature = rewriter.create<arith::XOrIOp>(op.getLoc(), intFeature, mask);
            threshold = -threshold;
        }
        rewriter.create<tree::IntNodeOp>(
            op.getLoc(),
            intFeature,
            rewriter.getI32IntegerAttr(*reinterpret_cast<int32_t*>(&threshold)),
            rewriter.getIndexAttr(-1),
            arith::CmpIPredicateAttr::get(rewriter.getContext(), compare),
            op.getTrueProbabilityAttr(),
            op.getFalseProbabilityAttr(),
            op.getSuccessor(0),
            op.getSuccessor(1)
        );
        rewriter.eraseOp(op);
        //rewriter.replaceOp(op, rewriter.create<tree::IntNodeOp>(op.getLoc(), intFeature, threshold.i32, compare, op.getTrueProbability().convertToDouble(), op.getFalseProbability().convertToDouble(), op.getSuccessor(0), op.getSuccessor(1)));
        /*rewriter.replaceOpWithNewOp<tree::IntNodeOp>(
            op,
            intFeature,
            rewriter.getI32IntegerAttr(threshold.i32),
            rewriter.getIndexAttr(-1),
            arith::CmpIPredicateAttr::get(rewriter.getContext(), compare),
            op.getTrueProbabilityAttr(),
            op.getFalseProbabilityAttr(),
            op.getSuccessor(0),
            op.getSuccessor(1)
        );*/
        //llvm::errs() << "Successfully created: " << *intFeature.getDefiningOp() << "\n";
        //llvm::errs() << "Successfully created: " << *intNode << "\n";
        return success();
    }
};

DenseMap<tree::FloatFeatureOp, tree::IntFeatureOp> createIntFeature(ModuleOp moduleOp) {
    IRRewriter rewriter(moduleOp->getContext());
    IRRewriter::InsertPoint insertPoint = rewriter.saveInsertionPoint();
    DenseMap<tree::FloatFeatureOp, tree::IntFeatureOp> featureMap;
    moduleOp.walk([&](tree::FloatFeatureOp floatFeature) {
        rewriter.setInsertionPoint(floatFeature);
        if (auto globalName = floatFeature->getAttrOfType<FlatSymbolRefAttr>("globalName")) { // Should never be true
            LLVM::GlobalOp globalOp = dyn_cast_or_null<LLVM::GlobalOp>(SymbolTable::lookupNearestSymbolFrom(moduleOp, globalName));
            featureMap[floatFeature] = rewriter.create<tree::IntFeatureOp>(floatFeature.getLoc(), floatFeature.getFeatureAddr(), floatFeature.getFeatureIndex()->getSExtValue(), globalOp);
        }
        else {
            featureMap[floatFeature] = rewriter.create<tree::IntFeatureOp>(floatFeature.getLoc(), floatFeature.getFeatureAddr(), floatFeature.getFeatureIndex()->getSExtValue());
        }
    });
    rewriter.restoreInsertionPoint(insertPoint);
    return featureMap;
}

void replaceNode(ModuleOp moduleOp, DenseMap<tree::FloatFeatureOp, tree::IntFeatureOp>& featureMap) {
    IRRewriter rewriter(moduleOp->getContext());
    IRRewriter::InsertPoint insertPoint = rewriter.saveInsertionPoint();
    moduleOp.walk([&](tree::FloatNodeOp floatNode) {
        rewriter.setInsertionPoint(floatNode);
        if (tree::FloatFeatureOp floatFeature = floatNode.getFeature().getDefiningOp<tree::FloatFeatureOp>()) {
            Value intFeature = featureMap[floatFeature];
            float threshold = floatNode.getThreshold().convertToFloat();
            arith::CmpIPredicate compare = arith::CmpIPredicate::slt;
            if (floatNode.getPredicate() == arith::CmpFPredicate::OLT) {
                compare = threshold < 0 ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::slt;
            }
            else if (floatNode.getPredicate() == arith::CmpFPredicate::OLE) {
                compare = threshold < 0 ? arith::CmpIPredicate::sge : arith::CmpIPredicate::sle;
            }
            else if (floatNode.getPredicate() == arith::CmpFPredicate::OGT) {
                compare = threshold < 0 ? arith::CmpIPredicate::slt : arith::CmpIPredicate::sgt;
            }
            else if (floatNode.getPredicate() == arith::CmpFPredicate::OGE) {
                compare = threshold < 0 ? arith::CmpIPredicate::sle : arith::CmpIPredicate::sge;
            }
            if (threshold < 0) {
                Value mask = rewriter.create<arith::ConstantIntOp>(floatFeature.getLoc(), 0x80000000, 32);
                intFeature = rewriter.create<arith::XOrIOp>(floatFeature.getLoc(), intFeature, mask);
                threshold = -threshold;
            }
            rewriter.create<tree::IntNodeOp>(
                floatNode.getLoc(),
                intFeature,
                rewriter.getI32IntegerAttr(*reinterpret_cast<int32_t*>(&threshold)),
                rewriter.getIndexAttr(-1),
                arith::CmpIPredicateAttr::get(rewriter.getContext(), compare),
                floatNode.getTrueProbabilityAttr(),
                floatNode.getFalseProbabilityAttr(),
                floatNode.getSuccessor(0),
                floatNode.getSuccessor(1)
            );
            floatNode->erase();
        }
    });
    rewriter.restoreInsertionPoint(insertPoint);
}

void eraseFloatFeature(ModuleOp moduleOp) {
    moduleOp.walk([](tree::FloatFeatureOp floatFeature) {
        floatFeature->erase();
    });
}

class TreeFLInt : public impl::TreeFLIntBase<TreeFLInt> {
public:
    using impl::TreeFLIntBase<TreeFLInt>::TreeFLIntBase;
    void runOnOperation() final {
        DenseMap<tree::FloatFeatureOp, tree::IntFeatureOp> featureMap = createIntFeature(getOperation());
        replaceNode(getOperation(), featureMap);
        eraseFloatFeature(getOperation());
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
//typedef std::tuple<double, long long, float> probabilityIndexFloatThreshold;
//typedef std::tuple<double, long long, int> probabilityIndexIntThreshold;
typedef std::pair<long long, float> indexFloatThreshold;
typedef std::pair<long long, int> indexIntThreshold;
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
void TreeBuildGlobal(ModuleOp moduleOp, std::vector<indexFloatThreshold> &floatThresholdVector, std::vector<indexIntThreshold> &intThresholdVector) {
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToStart(moduleOp.getBody());
    if (!floatThresholdVector.empty()) {
        std::vector<float> thresholdVector(floatThresholdVector.size());
        std::transform(floatThresholdVector.cbegin(), floatThresholdVector.cend(), thresholdVector.begin(), [](const indexFloatThreshold pairElement) {
            return pairElement.second;
        });
        /*std::unordered_map<float, int> thresholdMap;
        for (int index = 0; index < (int)floatThresholdVector.size(); ++index) {
            ++thresholdMap[thresholdVector[index]];
        }
        llvm::outs() << "         Thresholds count: " << floatThresholdVector.size() << '\n';
        llvm::outs() << "Distinct thresholds count: " << thresholdMap.size() << '\n';*/
        MemRefType type = MemRefType::get({static_cast<long>(floatThresholdVector.size())}, builder.getF32Type());
        DenseFPElementsAttr value = DenseFPElementsAttr::get(RankedTensorType::get({static_cast<long>(floatThresholdVector.size())}, builder.getF32Type()), llvm::ArrayRef(thresholdVector));
        builder.create<memref::GlobalOp>(moduleOp.getLoc(), builder.getStringAttr("global_float_thresholds"), builder.getStringAttr("private"), type, value, true, builder.getI64IntegerAttr(64));
    }
    else if (!intThresholdVector.empty()) {
        std::vector<int> thresholdVector(intThresholdVector.size());
        std::transform(intThresholdVector.cbegin(), intThresholdVector.cend(), thresholdVector.begin(), [](const indexIntThreshold pairElement) {
            return pairElement.second;
        });
        MemRefType type = MemRefType::get({static_cast<long>(intThresholdVector.size())}, builder.getI32Type());
        DenseIntElementsAttr value = DenseIntElementsAttr::get(RankedTensorType::get({static_cast<long>(intThresholdVector.size())}, builder.getI32Type()), llvm::ArrayRef(thresholdVector));
        builder.create<memref::GlobalOp>(moduleOp.getLoc(), builder.getStringAttr("global_int_thresholds"), builder.getStringAttr("private"), type, value, true, builder.getI64IntegerAttr(64));
    }
}

void TreeBuildGetGlobal(func::FuncOp &funcOp, std::vector<indexFloatThreshold> &floatThresholdVector, std::vector<indexIntThreshold> &intThresholdVector) {
    Block &entryBlock = funcOp.front();
    OpBuilder builder(&entryBlock, entryBlock.begin());
    if (!floatThresholdVector.empty()) {
        builder.create<memref::GetGlobalOp>(funcOp.getLoc(), MemRefType::get({static_cast<long>(floatThresholdVector.size())}, builder.getF32Type()), builder.getStringAttr("global_float_thresholds"));
    }
    else if (!intThresholdVector.empty()) {
        builder.create<memref::GetGlobalOp>(funcOp.getLoc(), MemRefType::get({static_cast<long>(intThresholdVector.size())}, builder.getI32Type()), builder.getStringAttr("global_int_thresholds"));
    }
}
/*
void TreeFloatThresholdCompression(tree::FloatNodeOp &op, std::unordered_map<float, double> &floatThresholdMap) {
    op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, floatThresholdMap.size()));
    floatThresholdMap[op.getThreshold().convertToFloat()] += op.getTrueProbability().convertToDouble() + op.getFalseProbability().convertToDouble();
}*/
// Order-3rd
void TreeFloatThresholdCompression(tree::FloatNodeOp &op, std::vector<indexFloatThreshold> &floatThresholdVector, std::unordered_map<float, int> &floatThresholdMap) {
    if (!floatThresholdMap.count(op.getThreshold().convertToFloat())) {
        floatThresholdVector.emplace_back(floatThresholdMap.size(), op.getThreshold().convertToFloat());
        op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, floatThresholdMap.size()));
        floatThresholdMap[op.getThreshold().convertToFloat()] = floatThresholdMap.size();
    }
    else {
        op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, floatThresholdMap[op.getThreshold().convertToFloat()]));
    }
}

void TreeIntThresholdCompression(tree::IntNodeOp &op, std::vector<indexIntThreshold> &intThresholdVector, std::unordered_map<int, int> &intThresholdMap) {
    if (!intThresholdMap.count(op.getThreshold())) {
        intThresholdVector.emplace_back(intThresholdMap.size(), op.getThreshold());
        op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, intThresholdMap.size()));
        intThresholdMap[op.getThreshold()] = intThresholdMap.size();
    }
    else {
        op.setThresholdIndex(llvm::APInt(sizeof(size_t) * 8, intThresholdMap[op.getThreshold()]));
    }
}

// Order-4th
/*
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
}*/

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
            //LLVMTypeConverter converter(getContext());
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
    TreeFloatNodeLoweringRewriter(MLIRContext *context, std::vector<indexFloatThreshold>& floatThresholdVector)
        : OpRewritePattern<tree::FloatNodeOp>(context), floatThresholdVector(floatThresholdVector) {}
    using OpRewritePattern<tree::FloatNodeOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tree::FloatNodeOp op, PatternRewriter &rewriter) const final {
        ValueRange nullList = {};
        Value threshold;
        if (0 <= op.getThresholdIndex().getSExtValue() && op.getThresholdIndex().getSExtValue() < static_cast<int64_t>(floatThresholdVector.size())) {
            Value index = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(floatThresholdVector[op.getThresholdIndex().getSExtValue()].first));
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
    std::vector<indexFloatThreshold>& floatThresholdVector;
};

class TreeIntNodeLoweringRewriter : public OpRewritePattern<tree::IntNodeOp> {
public:
    TreeIntNodeLoweringRewriter(MLIRContext *context, std::vector<indexIntThreshold>& intThresholdVector)
        : OpRewritePattern<tree::IntNodeOp>(context), intThresholdVector(intThresholdVector) {}
    using OpRewritePattern<tree::IntNodeOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tree::IntNodeOp op, PatternRewriter &rewriter) const final {
        ValueRange nullList = {};
        Value threshold;
        if (0 <= op.getThresholdIndex().getSExtValue() && op.getThresholdIndex().getSExtValue() < static_cast<int64_t>(intThresholdVector.size())) {
            Value index = rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexType(), rewriter.getIndexAttr(intThresholdVector[op.getThresholdIndex().getSExtValue()].first));
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
    std::vector<indexIntThreshold>& intThresholdVector;
};

class TreeLowering : public impl::TreeLoweringBase<TreeLowering> {
public:
    using impl::TreeLoweringBase<TreeLowering>::TreeLoweringBase;
    void runOnOperation() final {
        //std::vector<probabilityIndexFloatThreshold> floatThresholdTupleVector;
        //std::vector<probabilityIndexIntThreshold> intThresholdTupleVector;
        std::vector<indexFloatThreshold> floatThresholdVector;
        std::vector<indexIntThreshold> intThresholdVector;
        std::unordered_map<float, int> floatThresholdMap;
        std::unordered_map<int, int> intThresholdMap;
        //std::unordered_map<int64_t, int> treeIndexFloatThresholdMap;
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
            // Order-4th
            /*int treeIndex = 0;
            getOperation().walk([&](func::FuncOp funcOp) {
                funcOp->walk([&](tree::FloatNodeOp nodeOp) {
                    TreeFloatThresholdCompression(nodeOp, floatThresholdTupleVector, treeIndexFloatThresholdMap, treeIndex);
                });
                ++treeIndex;
            });*/
            // Order-3rd
            int counter = 0;
            getOperation().walk([&](Operation *op) {
                if (tree::FloatNodeOp nodeOp = dyn_cast<tree::FloatNodeOp>(op)) {
                    TreeFloatThresholdCompression(nodeOp, floatThresholdVector, floatThresholdMap);
                }
                else if (tree::IntNodeOp nodeOp = dyn_cast<tree::IntNodeOp>(op)) {
                    TreeIntThresholdCompression(nodeOp, intThresholdVector, intThresholdMap);
                }
                ++counter;
            });
            llvm::outs() << (floatThresholdMap.size() ? floatThresholdMap.size() : intThresholdMap.size()) << '/' << counter << '\n';
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
            TreeBuildGlobal(getOperation(), floatThresholdVector, intThresholdVector);
            //treeIndex = 0;
            getOperation().walk([&](func::FuncOp funcOp) {
                TreeBuildGetGlobal(funcOp, floatThresholdVector, intThresholdVector);
            });
        }
        mlir::ConversionTarget target(getContext());
        target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
        //target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect>();
        target.addIllegalDialect<TreeDialect>();
        RewritePatternSet patterns(&getContext());
        patterns.add<TreeFloatFeatureLoweringRewriter>(&getContext(), getOperation());
        patterns.add<TreeIntFeatureLoweringRewriter>(&getContext(), getOperation());
        patterns.add<TreeFloatNodeLoweringRewriter>(&getContext(), floatThresholdVector);
        patterns.add<TreeIntNodeLoweringRewriter>(&getContext(), intThresholdVector);
        FrozenRewritePatternSet patternSet(std::move(patterns));
        if (failed(applyFullConversion(getOperation(), target, patternSet))) {
            signalPassFailure();
        }
    }
};
} // namespace
} // namespace mlir::tree
