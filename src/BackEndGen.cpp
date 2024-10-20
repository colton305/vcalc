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

    mlir::Value var = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, value.value, var);

    node->var->value.value = var;
    std::cout << node->var << "HERE1\n";
}

void BackEnd::generateAssignStat(std::shared_ptr<VarStatAST> node) {
    std::cout << node->expr << "\n";
    ExprResult value = generateExpr(node->expr);
    std::cout << "HERE10\n";
    std::cout << node->var << "\n";

    builder->create<mlir::LLVM::StoreOp>(loc, value.value, node->var->value.value);
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

ExprResult BackEnd::generateExpr(std::shared_ptr<ExprAST> node) {
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
    }
}

ExprResult BackEnd::generateBinExpr(std::shared_ptr<BinExprAST> node) {
    std::cout << "HERE6\n";

    if (node->lhs->type == "int" && node->rhs->type == "int") {
        return generateIntBinExpr(node);
    } else if (node->lhs->type == "vector" && node->rhs->type == "vector") {
        return generateVecBinExpr(node);
    } else {
        return generateVecIntBinExpr(node);
    }
    std::cout << "TYPE FALLTHROUGH\n";
}

ExprResult BackEnd::generateIntBinExpr(std::shared_ptr<BinExprAST> node) {
    ExprResult lhs = generateExpr(node->lhs);
    ExprResult rhs = generateExpr(node->rhs);
    ExprResult result;

    if (node->op == "..") {
        return generateRangeExpr(lhs.value, rhs.value);
    } else if (node->op == "*") {
        result.value =  builder->create<mlir::LLVM::MulOp>(loc, lhs.value, rhs.value);
    } else if (node->op == "/") {
        result.value =  builder->create<mlir::LLVM::SDivOp>(loc, lhs.value, rhs.value);
    } else if (node->op == "+") {
        result.value =  builder->create<mlir::LLVM::AddOp>(loc, lhs.value, rhs.value);
    } else if (node->op == "-") {
        result.value =  builder->create<mlir::LLVM::SubOp>(loc, lhs.value, rhs.value);
    } else if (node->op == "==") {
        result.value = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, lhs.value, rhs.value);
        result.value =  builder->create<mlir::LLVM::ZExtOp>(loc, intType, result.value);
    } else if (node->op == "!=") {
        result.value = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, lhs.value, rhs.value);
        result.value =  builder->create<mlir::LLVM::ZExtOp>(loc, intType, result.value);
    } else if (node->op == "<") {
        result.value = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, lhs.value, rhs.value);
        result.value =  builder->create<mlir::LLVM::ZExtOp>(loc, intType, result.value);
    } else if (node->op == ">") {
        result.value = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, lhs.value, rhs.value);
        result.value =  builder->create<mlir::LLVM::ZExtOp>(loc, intType, result.value);
    } else {
        std::cout << "Error: Operation not found\n";
    }

    return result;
}

ExprResult BackEnd::generateVecBinExpr(std::shared_ptr<BinExprAST> node) {
    if (node->op == "[") {
        return generateVecIndexExpr(node);
    } else {
        std::cout << "Error: Operation not found\n";
    }
}

ExprResult BackEnd::generateVecIntBinExpr(std::shared_ptr<BinExprAST> node) {
    if (node->op == "|") {
        return generateGenExpr(std::dynamic_pointer_cast<ScopedBinExprAST>(node));
    } else if (node->op == "&") {
        return generateFilterExpr(std::dynamic_pointer_cast<ScopedBinExprAST>(node));
    } else if (node->op == "[") {
        return generateIndexExpr(node);
    } else {
        std::cout << "Error: Operation not found\n";
    }
}

ExprResult BackEnd::generateGenExpr(std::shared_ptr<ScopedBinExprAST> node) {
    ExprResult vector = generateExpr(node->lhs);
    std::cout << node->iterator << "\n"; 
    node->iterator->value.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    ExprResult result;
    result.size = vector.size;
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, result.size, one);
    mlir::Value arraySize = builder->create<mlir::LLVM::SelectOp>(loc, comp, result.size, one);
    result.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, arraySize);

    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, index);

    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, vector.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
    builder->setInsertionPointToStart(body);

    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, vector.value, indexValue);
    mlir::Value vectorValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, vectorValue, node->iterator->value.value);

    ExprResult rhs = generateExpr(node->rhs);
    ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, result.value, indexValue);
    builder->create<mlir::LLVM::StoreOp>(loc, rhs.value, ptr);

    indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);
    builder->create<mlir::LLVM::BrOp>(loc, header);

    builder->setInsertionPointToStart(merge);
    return result;
}

ExprResult BackEnd::generateFilterExpr(std::shared_ptr<ScopedBinExprAST> node) {
    ExprResult vector = generateExpr(node->lhs);
    node->iterator->value.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    ExprResult result;
    result.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, vector.size);

    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* insert = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    mlir::Value arraySize = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, index);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, arraySize);

    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, vector.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
    builder->setInsertionPointToStart(body);

    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, vector.value, indexValue);
    mlir::Value vectorValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, vectorValue, node->iterator->value.value);

    indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);

    ExprResult rhs = generateExpr(node->rhs);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, rhs.value, zero);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, insert, header);
    builder->setInsertionPointToStart(insert);

    mlir::Value sizeValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, arraySize);
    ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, result.value, sizeValue);
    builder->create<mlir::LLVM::StoreOp>(loc, vectorValue, ptr);
    sizeValue = builder->create<mlir::LLVM::AddOp>(loc, sizeValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, sizeValue, arraySize);
    builder->create<mlir::LLVM::BrOp>(loc, header);
    
    builder->setInsertionPointToStart(merge);
    result.size = builder->create<mlir::LLVM::LoadOp>(loc, intType, arraySize);

    return result;
}

ExprResult BackEnd::generateVecIndexExpr(std::shared_ptr<BinExprAST> node) {
    ExprResult vector = generateExpr(node->lhs);
    ExprResult indexVector = generateExpr(node->rhs);

    ExprResult result;
    result.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, indexVector.size);
    result.size = indexVector.size;

    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* ib = mainFunc.addBlock();
    mlir::Block* postBody = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, index);

    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, result.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
    builder->setInsertionPointToStart(body);

    mlir::Value resultPtr = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, resultPtr);

    mlir::Value vectorIndex = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, indexVector.value, indexValue);
    mlir::Value vectorIndexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, vectorIndex);

    mlir::Value positive = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sge, vectorIndexValue, zero);
    mlir::Value inBounds = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, vectorIndexValue, vector.size);
    comp = builder->create<mlir::LLVM::AndOp>(loc, positive, inBounds);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, ib, postBody);
    builder->setInsertionPointToStart(ib);

    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, vector.value, vectorIndexValue);
    mlir::Value value = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, value, resultPtr);
    builder->create<mlir::LLVM::BrOp>(loc, postBody);
    builder->setInsertionPointToStart(postBody);

    ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, result.value, indexValue);
    mlir::Value resultValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, resultPtr);
    builder->create<mlir::LLVM::StoreOp>(loc, resultValue, ptr);
    indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);
    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(merge);

    return result;
}

ExprResult BackEnd::generateIndexExpr(std::shared_ptr<BinExprAST> node) {
    ExprResult vector = generateExpr(node->lhs);
    ExprResult index = generateExpr(node->rhs);

    mlir::Block* ib = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    mlir::Value resultPtr = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, resultPtr);

    mlir::Value positive = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sge, index.value, zero);
    mlir::Value inBounds = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, index.value, vector.size);
    mlir::Value comp = builder->create<mlir::LLVM::AndOp>(loc, positive, inBounds);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, ib, merge);
    builder->setInsertionPointToStart(ib);

    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, vector.value, index.value);
    mlir::Value value = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, value, resultPtr);
    builder->create<mlir::LLVM::BrOp>(loc, merge);
    builder->setInsertionPointToStart(merge);

    ExprResult result;
    result.value = builder->create<mlir::LLVM::LoadOp>(loc, intType, resultPtr);
    return result;
}

ExprResult BackEnd::generateRangeExpr(mlir::Value lowerBound, mlir::Value upperBound) {
    ExprResult result;

    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sle, lowerBound, upperBound);
    result.size = builder->create<mlir::LLVM::SubOp>(loc, upperBound, lowerBound);
    result.size = builder->create<mlir::LLVM::AddOp>(loc, result.size, one);
    result.size = builder->create<mlir::LLVM::SelectOp>(loc, comp, result.size, zero);
    
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, result.size, one);
    mlir::Value arraySize = builder->create<mlir::LLVM::SelectOp>(loc, comp, result.size, one);

    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    result.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, arraySize);

    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, lowerBound, index);

    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sle, indexValue, upperBound);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
    builder->setInsertionPointToStart(body);

    mlir::Value relativeIndex = builder->create<mlir::LLVM::SubOp>(loc, indexValue, lowerBound);
    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, result.value, relativeIndex);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, ptr);

    indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);
    builder->create<mlir::LLVM::BrOp>(loc, header);

    builder->setInsertionPointToStart(merge);
    return result;
}

ExprResult BackEnd::generateNumExpr(std::shared_ptr<NumAST> node) {
    ExprResult result;
    result.value = builder->create<mlir::LLVM::ConstantOp>(loc, intType, node->value);
    return result;
}

ExprResult BackEnd::generateVarExpr(std::shared_ptr<VarAST> node) {
    std::cout << node->var << "\n";
    ExprResult result;
    result.value = builder->create<mlir::LLVM::LoadOp>(loc, intType, node->var->value.value);
    return result;
}
