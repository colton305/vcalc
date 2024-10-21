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
    ExprResult value = generateExpr(node->expr);

    if (node->expr->type == "int") {
        mlir::Value var = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
        builder->create<mlir::LLVM::StoreOp>(loc, value.value, var);
        node->var->value.value = var;
    } else {
        node->var->value.value = value.value;
        mlir::Value size = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
        builder->create<mlir::LLVM::StoreOp>(loc, value.size, size);
        node->var->value.size = size;
    }
    std::cout << node->var << "HERE1\n";
}

void BackEnd::generateAssignStat(std::shared_ptr<VarStatAST> node) {
    std::cout << node->expr << "\n";
    ExprResult value = generateExpr(node->expr);
    std::cout << "HERE10\n";
    std::cout << node->var << "\n";

    if (node->expr->type == "int") {
        builder->create<mlir::LLVM::StoreOp>(loc, value.value, node->var->value.value);
    } else {
        node->var->value.value = value.value;
        builder->create<mlir::LLVM::StoreOp>(loc, value.size, node->var->value.size);
    }
    std::cout << "HERE3\n";
}

void BackEnd::generateCondStat(std::shared_ptr<BlockStatAST> node) {
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    ExprResult cond = generateExpr(node->expr);
    std::cout << "HERE7\n";

    cond.value = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, cond.value, zero);
    builder->create<mlir::LLVM::CondBrOp>(loc, cond.value, merge, body);

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
    ExprResult cond = generateExpr(node->expr);

    cond.value = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, cond.value, zero);
    builder->create<mlir::LLVM::CondBrOp>(loc, cond.value, merge, body);

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
    ExprResult value = generateExpr(node->expr);
    std::cout << "MADE\n";
    if (node->expr->type == "int") {
        generateIntPrint("intFormat", value.value);
    } else if (node->expr->type == "vector") {
        generatePrint("vectorStartFormat");

        mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
        builder->create<mlir::LLVM::StoreOp>(loc, zero, index);
        mlir::Block* header = mainFunc.addBlock();
        mlir::Block* body = mainFunc.addBlock();
        mlir::Block* postBody = mainFunc.addBlock();
        mlir::Block* merge = mainFunc.addBlock();

        builder->create<mlir::LLVM::BrOp>(loc, header);
        builder->setInsertionPointToStart(header);

        mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
        mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, value.size);
        builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
        builder->setInsertionPointToStart(body);

        mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, value.value, indexValue);
        mlir::Value printValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
        generateIntPrint("vectorIntFormat", printValue);

        indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
        builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);
        comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, value.size);
        builder->create<mlir::LLVM::CondBrOp>(loc, comp, postBody, header);
        builder->setInsertionPointToStart(postBody);

        generatePrint("vectorSpaceFormat");
        builder->create<mlir::LLVM::BrOp>(loc, header);

        builder->setInsertionPointToStart(merge);
        generatePrint("vectorEndFormat");

    } else {
        std::cout << "Error: Undefined type\n";
    }
}

void BackEnd::generatePrint(std::string format) {
    mlir::LLVM::GlobalOp formatString = module.lookupSymbol<mlir::LLVM::GlobalOp>(format);
    mlir::Value formatStringPtr = builder->create<mlir::LLVM::AddressOfOp>(loc, formatString); 
    mlir::ValueRange args = {formatStringPtr}; 
    mlir::LLVM::LLVMFuncOp printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"); 
    builder->create<mlir::LLVM::CallOp>(loc, printfFunc, args);
}

void BackEnd::generateIntPrint(std::string format, mlir::Value value) {
    mlir::LLVM::GlobalOp formatString = module.lookupSymbol<mlir::LLVM::GlobalOp>(format);
    mlir::Value formatStringPtr = builder->create<mlir::LLVM::AddressOfOp>(loc, formatString); 
    mlir::ValueRange args = {formatStringPtr, value}; 
    mlir::LLVM::LLVMFuncOp printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"); 
    builder->create<mlir::LLVM::CallOp>(loc, printfFunc, args);
}
