#pragma once

// Pass manager
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"

// Translation
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/raw_os_ostream.h"

// MLIR IR
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

// Dialects 
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"

// AST
#include "ASTBuilder.h"


class BackEnd {
 public:
    BackEnd();

    void generateStat(std::shared_ptr<StatAST> node);
    ExprResult generateExpr(std::shared_ptr<ExprAST> node);
    void generateDeclStat(std::shared_ptr<VarStatAST> node);
    void generateAssignStat(std::shared_ptr<VarStatAST> node);
    void generateCondStat(std::shared_ptr<BlockStatAST> node);
    void generateLoopStat(std::shared_ptr<BlockStatAST> node);
    void generatePrintStat(std::shared_ptr<StatAST> node);
    void generatePrint(std::string format);
    void generateIntPrint(std::string format, mlir::Value value);
    ExprResult generateBinExpr(std::shared_ptr<BinExprAST> node);
    ExprResult generateIntBinExpr(std::shared_ptr<BinExprAST> node);
    ExprResult generateVecBinExpr(std::shared_ptr<BinExprAST> node);
    ExprResult generateVecOpExpr(std::shared_ptr<BinExprAST> node);
    ExprResult generateVecIntBinExpr(std::shared_ptr<BinExprAST> node);
    ExprResult generateVecIntOpExpr(std::shared_ptr<BinExprAST> node);
    mlir::Value generateBinOp(std::string op, mlir::Value lhs, mlir::Value rhs);
    ExprResult generateGenExpr(std::shared_ptr<ScopedBinExprAST> node);
    ExprResult generateFilterExpr(std::shared_ptr<ScopedBinExprAST> node);
    ExprResult generateVecIndexExpr(std::shared_ptr<BinExprAST> node);
    ExprResult generateIndexExpr(std::shared_ptr<BinExprAST> node);
    ExprResult generateRangeExpr(mlir::Value lowerBound, mlir::Value upperBound);
    ExprResult generateNumExpr(std::shared_ptr<NumAST> node);
    ExprResult generateVarExpr(std::shared_ptr<VarAST> node);

    int emitModule(std::shared_ptr<BlockStatAST> root);
    int lowerDialects();
    void dumpLLVM(std::ostream &os);
 
 protected:
    void setupPrintf();
    void createGlobalString(const char *str, const char *string_name);
      
 private:
    // MLIR
    mlir::MLIRContext context;
    mlir::ModuleOp module;
    std::shared_ptr<mlir::OpBuilder> builder;
    mlir::Location loc;

    // LLVM 
    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> llvm_module;
    mlir::LLVM::LLVMFuncOp mainFunc;

    // Types
    mlir::Type intType;
    mlir::Type ptrType;

    // Constants
    mlir::Value zero;
    mlir::Value one;
};
