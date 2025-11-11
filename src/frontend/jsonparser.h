#ifndef JSON_PARSER_H
#define JSON_PARSER_H

#include <fstream>
#include "json.hpp"

#include "Tree/TreeDialect.h"
#include "Tree/TreeOps.h"
#include "Tree/TreePasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
/*
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
*/
#include "decisiontree.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
//#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
//#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"

using json = nlohmann::json;
using namespace mlir;

namespace Treehierarchy
{
    class BuildOptions
    {
    public:
        bool enable_swap = false;
        bool enable_flint = false;
        bool enable_ra = false;
        size_t regNum = 32;
        bool enable_hoisting = false;
        bool enable_ordering = false;
        bool enable_inline = false;
    };

    class JsonParser
    {
    public:
        JsonParser(const std::string &forestJSONPath, BuildOptions option) : m_option(option), m_forest(new DecisionForest()),
                                                                             m_context(), m_builder(&m_context),
                                                                             m_module(ModuleOp::create(m_builder.getUnknownLoc(), llvm::StringRef("ForestModule")))
        {
            std::ifstream fin(forestJSONPath);
            assert(fin);
            fin >> m_json;
            initMLIRContext(m_context);
        }

        // Provide a virtual destructor definition
        virtual ~JsonParser()
        {
            delete m_forest;
        }

        virtual void ConstructForest() = 0;

        LLVM::GlobalOp pin_reg[32];
        LLVM::AddressOfOp pin_addr[32];

        ModuleOp buildHIRModule()
        {
            if (m_option.enable_ra) {
                m_forest->SetRegNum(m_option.regNum);
                // Initialize RA variables
                auto loc = m_builder.getUnknownLoc();
                size_t pinRegNum = m_forest->GetRegNum();
                pinRegNum = (pinRegNum < m_forest->GetFeatureSize()) ? pinRegNum : m_forest->GetFeatureSize();
                m_forest->SetRegNum(pinRegNum);

                for (size_t i = 0; i < pinRegNum; i++)
                {
                    // Set global vars for pin registers
                    //pin_reg[i] = m_builder.create<LLVM::GlobalOp>(loc, getFeatureType(), /*isConstant=*/false,
                    pin_reg[i] = m_builder.create<LLVM::GlobalOp>(loc, getF32(), /*isConstant=*/false,
                                    LLVM::Linkage::Internal, "pin_reg_" + std::to_string(i), Attribute());
                    m_module.push_back(pin_reg[i]);
                }
            }
          
            for (size_t i = 0; i < m_forest->GetTreeSize(); i++)
            {
                OpBuilder::InsertPoint insertPoint = m_builder.saveInsertionPoint();

                func::FuncOp function(getFunctionPrototype("tree_" + std::to_string(i)));

                Block *entryBlock = function.addEntryBlock();
                DecisionTree *tree = m_forest->GetTree(i);
                m_builder.setInsertionPointToStart(entryBlock);
                buildNodeOp(entryBlock, tree, 0);
                m_module.push_back(function);

                m_builder.restoreInsertionPoint(insertPoint);
            }

            CreatePredictFunction();

            return m_module;
        }

        ModuleOp lowerToLLVMModule()
        {
            /*llvm::cl::opt<bool> disableMultithreading(
                "disable-multithreading",
                llvm::cl::desc("Disable multi-threading in MLIR"),
                llvm::cl::init(true)
            );
            m_context.disableMultithreading();*/
            PassManager pm(&m_context);
            //m_context.getOrLoadDialect<vector::VectorDialect>();
            m_context.getOrLoadDialect<memref::MemRefDialect>();
            DialectRegistry registry;
            registry.insert<LLVM::LLVMDialect>();
            LLVM::registerInlinerInterface(registry);
            func::registerInlinerExtension(registry);
            m_context.appendDialectRegistry(registry);

            //pm.enableIRPrinting();
            //llvm::DebugFlag = true;
            //llvm::setCurrentDebugType("inliner-pass");
            if (m_option.enable_inline)
            {
                //OpPassManager &funcPM = pm.nest<func::FuncOp>();
                //funcPM.addPass(mlir::createInlinerPass());
                //pm.addPass(createInlinerPass(InlinerOptions({.maxInliningIterations = 9487})));
                pm.addPass(tree::createTreeInline());
            }
            if (m_option.enable_swap)
            {
                pm.addPass(tree::createTreeSwap());
            }
            if (m_option.enable_flint)
            {
                pm.addPass(tree::createTreeFLInt());
            }
            if (m_option.enable_hoisting) {
                pm.addPass(tree::createTreeLoadHoisting(tree::TreeLoadHoistingOptions({.probabilityThreshold = 0.8})));
            }
            pm.addPass(createSymbolDCEPass());
            // Eliminate every downstream feature node with the same feature index
            // Equivalent to LLVM default register allocation
            //pm.addPass(createCSEPass());
            pm.addPass(tree::createTreeLowering(tree::TreeLoweringOptions({.thresholdOrdering = m_option.enable_ordering})));
            
            //llvm::errs() << "Swap: " << (m_option.enable_swap ? "true" : "false") << '\n';
            //llvm::errs() << "FLInt: " << (m_option.enable_flint ? "true" : "false") << '\n';
            //llvm::errs() << "Hoisting: " << (m_option.enable_hoisting ? "true" : "false") << '\n';
            if (failed(pm.run(m_module)))
            {
                llvm::errs() << "Failed to execute pass manager\n";
            }//return m_module; // Temp: for testing

            LLVMTypeConverter converter(&m_context);

            ConversionTarget target(m_context);
            RewritePatternSet patterns(&m_context);

            target.addLegalDialect<LLVM::LLVMDialect>();
            target.addIllegalDialect<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect>();

            //populateVectorToLLVMConversionPatterns(converter, patterns);
            cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
            populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
            arith::populateArithToLLVMConversionPatterns(converter, patterns);
            populateFuncToLLVMConversionPatterns(converter, patterns);
            populateMathToLLVMConversionPatterns(converter, patterns);

            if (failed(applyPartialConversion(m_module, target, std::move(patterns))))
            {
                llvm::errs() << "Decision forest lowering pass failed\n";
            }

            return m_module;
        }

        size_t getForestClassNum() { return m_forest->GetClassNum(); }

    protected:
        BuildOptions m_option;
        json m_json;
        DecisionForest *m_forest;
        DecisionTree *m_decisionTree;
        MLIRContext m_context;
        OpBuilder m_builder;
        ModuleOp m_module;

        virtual void CreatePredictFunction() = 0;
        virtual void CreateLeafNode(Value result, DecisionTree::Node node) = 0;
        virtual arith::CmpFPredicate getComparePredicate() = 0;
        virtual arith::CmpFPredicate getReverseComparePredicate() = 0;
        virtual LLVM::ICmpPredicate getCompareIntPredicate() = 0;
        virtual LLVM::ICmpPredicate getReverseCompareIntPredicate() = 0;
        virtual FunctionType getTreeFunctionType() = 0;

        void initMLIRContext(MLIRContext &context)
        {
            context.getOrLoadDialect<arith::ArithDialect>();
            context.getOrLoadDialect<func::FuncDialect>();
            context.getOrLoadDialect<LLVM::LLVMDialect>();
            context.getOrLoadDialect<math::MathDialect>();
            context.getOrLoadDialect<tree::TreeDialect>();
        }

        /*Type getFeaturePointerType()
        {
            LLVMTypeConverter converter(&m_context);
            if (m_option.enable_flint)
            {
                return converter.getPointerType(getI32());
            }
            else
            {
                return converter.getPointerType(getF32());
            }
        }

        Type getResultPointerType()
        {
            LLVMTypeConverter converter(&m_context);
            return converter.getPointerType(getF32());
        }*/

        Type getPointerType()
        {
            return LLVM::LLVMPointerType::get(m_builder.getContext());
        }

        Type getFeatureType()
        {
            return m_option.enable_flint ? getI32() : getF32();
        }

        Type getI32()
        {
            return m_builder.getI32Type();
        }

        Type getF32()
        {
            return m_builder.getF32Type();
        }

        func::FuncOp getFunctionPrototype(std::string funName)
        {
            auto loc = m_builder.getUnknownLoc();
            auto functionType = getTreeFunctionType();
            auto function = m_builder.create<func::FuncOp>(loc, funName, functionType);
            function.setPrivate();

            return function;
        }

        Value createThreshold(float thresholdVal)
        {
            auto loc = m_builder.getUnknownLoc();
            if (m_option.enable_flint)
            {
                if (thresholdVal < 0)
                {
                    thresholdVal *= -1;
                }
                int intValue = *(int *)&thresholdVal;
                //return m_builder.create<arith::ConstantIntOp>(loc, intValue, getI32());
                return m_builder.create<arith::ConstantIntOp>(loc, intValue, 32);
            }
            else
            {
                return m_builder.create<arith::ConstantOp>(loc, getF32(), m_builder.getF32FloatAttr(thresholdVal));
            }
        }

        void buildNodeOp(Block *entryBlock, DecisionTree *tree, int idx)
        {
            DecisionTree::Node node = tree->GetNode(idx);
            auto loc = m_builder.getUnknownLoc();

            if (!node.IsLeaf())
            {
                //Value threshold = createThreshold(node.threshold);
                Value feature;

                int nodeGlobalIdx = m_forest->GetGlobalIdxFromFeature(node.featureIndex);
                //printf("Feature: %d, Get Idx: %d\n", node.featureIndex, nodeGlobalIdx);
                if(m_option.enable_ra && nodeGlobalIdx >= 0)
                {
                    /*Value global_addr = m_builder.create<LLVM::AddressOfOp>(loc, pin_reg[nodeGlobalIdx]);
                    feature = m_builder.create<LLVM::LoadOp>(loc, getF32(), global_addr);*/
                    //llvm::outs() << "If before\n";
                    feature = m_builder.create<tree::FloatFeatureOp>(loc, entryBlock->getArgument(0), node.featureIndex, pin_reg[nodeGlobalIdx]);
                    //llvm::outs() << "If after\n";
                }
                else
                {
                    /*Value featureIdx = m_builder.create<arith::ConstantIntOp>(loc, node.featureIndex, getI32());
                    Value input = entryBlock->getArgument(0);
                    Value featurePtr = m_builder.create<LLVM::GEPOp>(loc, getFeaturePointerType(), getF32(), input, featureIdx);
                    feature = m_builder.create<LLVM::LoadOp>(loc, getF32(), featurePtr);*/
                    //llvm::outs() << "Else before\n";
                    feature = m_builder.create<tree::FloatFeatureOp>(loc, entryBlock->getArgument(0), node.featureIndex);
                    //llvm::outs() << "Else after\n";
                }
              
                /*if (m_option.enable_flint && node.threshold < 0)
                {
                    Value mask = m_builder.create<arith::ConstantIntOp>(loc, 0x1 << 31, getI32());
                    feature = m_builder.create<mlir::arith::XOrIOp>(loc, feature, mask);
                }*/

                OpBuilder::InsertPoint insertPoint = m_builder.saveInsertionPoint();

                /*auto leftNode = tree->GetNode(node.leftChild);
                auto rightNode = tree->GetNode(node.rightChild);
                auto predicate = getComparePredicate();
                auto predicate2 = getCompareIntPredicate();*/
                int64_t leftIdx = node.leftChild;
                int64_t rightIdx = node.rightChild;

                /*if (m_option.enable_swap && leftNode.probability < rightNode.probability)
                {
                    predicate = getReverseComparePredicate();
                    predicate2 = getReverseCompareIntPredicate();
                    leftIdx = node.rightChild;
                    rightIdx = node.leftChild;
                }*/

                /*Value condition;
                if (m_option.enable_flint && node.threshold < 0)
                {
                    condition = m_builder.create<LLVM::ICmpOp>(loc, predicate2, threshold, feature);
                }
                else if (m_option.enable_flint)
                {
                    condition = m_builder.create<LLVM::ICmpOp>(loc, predicate2, feature, threshold);
                }
                else
                {
                    condition = m_builder.create<arith::CmpFOp>(loc, predicate, feature, threshold);
                }*/

                Region *funcBody = entryBlock->getParent();
                Block *tBlock = m_builder.createBlock(funcBody);
                m_builder.setInsertionPointToStart(tBlock);
                buildNodeOp(entryBlock, tree, leftIdx);

                Block *fBlock = m_builder.createBlock(funcBody);
                m_builder.setInsertionPointToStart(fBlock);
                buildNodeOp(entryBlock, tree, rightIdx);

                m_builder.restoreInsertionPoint(insertPoint);
                ValueRange nullList = {};
                //m_builder.create<LLVM::CondBrOp>(loc, condition, tBlock, nullList, fBlock, nullList);
                m_builder.create<tree::FloatNodeOp>(loc, feature, node.threshold,
                                                    getComparePredicate(),
                                                    tree->GetNode(leftIdx).probability, tree->GetNode(rightIdx).probability,
                                                    tBlock, fBlock);
            }
            else
            {
                Value result = entryBlock->getArgument(1);
                CreateLeafNode(result, node);
            }
        }
    };
}
#endif