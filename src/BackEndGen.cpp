#include "BackEnd.h"


void BackEnd::generateStat(std::shared_ptr<StatAST> node) {
    if (node->op == "decl") {
        generateDeclStat(std::dynamic_pointer_cast<VarStatAST>(node));
    } else if (node->op == "assign") {
        generateAssignStat(std::dynamic_pointer_cast<VarStatAST>(node));
    } else if (node->op == "cond") {
        generateCondStat(std::dynamic_pointer_cast<BlockStatAST>(node));
    } else if (node->op == "loop") {
        generateLoopStat(std::dynamic_pointer_cast<BlockStatAST>(node));
    } else if (node->op == "print") {
        generatePrintStat(node);
    } else {
        std::cout << "Error: Invalid statement " << node->op << "\n";
    }
}

void BackEnd::generateDeclStat(std::shared_ptr<VarStatAST> node) {
    std::cout << "AHA\n";
    mlir::Value value = generateExpr(node->expr);

    mlir::Value varSize = builder->create<mlir::LLVM::ConstantOp>(loc, intType, 1);
    mlir::Value var = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, varSize);
    builder->create<mlir::LLVM::StoreOp>(loc, value, var);

    node->var->value = var;
    node->var->block = builder->getInsertionBlock();
    std::cout << node->var << "HERE1\n";
}

void BackEnd::generateAssignStat(std::shared_ptr<VarStatAST> node) {
    std::cout << node->expr << "\n";
    mlir::Value value = generateExpr(node->expr);
    std::cout << "HERE10\n";
    std::cout << node->var << "\n";

    builder->create<mlir::LLVM::StoreOp>(loc, value, node->var->value);
    std::cout << "HERE3\n";
}

void BackEnd::generateCondStat(std::shared_ptr<BlockStatAST> node) {
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    mlir::Value cond = generateExpr(node->expr);
    std::cout << "HERE7\n";

    cond = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, cond, zero);
    builder->create<mlir::LLVM::CondBrOp>(loc, cond, merge, body);

    builder->setInsertionPointToStart(body);
    for (auto stat : node->stats) {
        std::cout << "HERE8\n";
        generateStat(stat);
    }
    builder->create<mlir::LLVM::BrOp>(loc, merge);

    builder->setInsertionPointToStart(merge);
    std::cout << "HERE2\n";
}

void BackEnd::generateLoopStat(std::shared_ptr<BlockStatAST> node) {
    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);
    mlir::Value cond = generateExpr(node->expr);

    cond = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, cond, zero);
    builder->create<mlir::LLVM::CondBrOp>(loc, cond, merge, body);

    builder->setInsertionPointToStart(body);
    for (auto stat : node->stats) {
        std::cout << "HERE11\n";
        generateStat(stat);
    }
    builder->create<mlir::LLVM::BrOp>(loc, header);

    builder->setInsertionPointToStart(merge);
    std::cout << "HERE12\n";
}

void BackEnd::generatePrintStat(std::shared_ptr<StatAST> node) {
    std::cout << "PRINTING\n";
    mlir::Value value = generateExpr(node->expr);
    std::cout << "MADE\n";

    mlir::Value formatStringPtr = builder->create<mlir::LLVM::AddressOfOp>(loc, formatString); 
    mlir::ValueRange args = {formatStringPtr, value}; 
    mlir::LLVM::LLVMFuncOp printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"); 
    builder->create<mlir::LLVM::CallOp>(loc, printfFunc, args);
}

mlir::Value BackEnd::generateExpr(std::shared_ptr<ExprAST> node) {
    if (auto bin = std::dynamic_pointer_cast<BinExprAST>(node)) {
        std::cout << "HERE4\n";
        return generateBinExpr(bin);
    } else if (auto num = std::dynamic_pointer_cast<NumAST>(node)) {
        std::cout << "HERE5\n";
        return generateNumExpr(num);
    } else if (auto var = std::dynamic_pointer_cast<VarAST>(node)) {
        std::cout << "HERE\n";
        return generateVarExpr(var);
    } else {
        std::cout << "Error: Expr type not found\n";
        return nullptr;
    }
}

mlir::Value BackEnd::generateBinExpr(std::shared_ptr<BinExprAST> node) {
    mlir::Value lhs = generateExpr(node->lhs);
    mlir::Value rhs = generateExpr(node->rhs);
    std::cout << "HERE6\n";

    if (node->op == "*") {
        return builder->create<mlir::LLVM::MulOp>(loc, lhs, rhs);
    } else if (node->op == "/") {
        return builder->create<mlir::LLVM::SDivOp>(loc, lhs, rhs);
    } else if (node->op == "+") {
        return builder->create<mlir::LLVM::AddOp>(loc, lhs, rhs);
    } else if (node->op == "-") {
        return builder->create<mlir::LLVM::SubOp>(loc, lhs, rhs);
    } else if (node->op == "==") {
        mlir::Value result = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, lhs, rhs);
        return builder->create<mlir::LLVM::ZExtOp>(loc, intType, result);
    } else if (node->op == "!=") {
        mlir::Value result = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, lhs, rhs);
        return builder->create<mlir::LLVM::ZExtOp>(loc, intType, result);
    } else if (node->op == "<") {
        mlir::Value result = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, lhs, rhs);
        return builder->create<mlir::LLVM::ZExtOp>(loc, intType, result);
    } else if (node->op == ">") {
        mlir::Value result = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, lhs, rhs);
        return builder->create<mlir::LLVM::ZExtOp>(loc, intType, result);
    } else {
        std::cout << "Error: Operation not found\n";
        return nullptr;
    }
}

mlir::Value BackEnd::generateNumExpr(std::shared_ptr<NumAST> node) {
    return builder->create<mlir::LLVM::ConstantOp>(loc, intType, node->value);
}

mlir::Value BackEnd::generateVarExpr(std::shared_ptr<VarAST> node) {
    std::cout << node->var << "\n";
    return builder->create<mlir::LLVM::LoadOp>(loc, intType, node->var->value);
}
