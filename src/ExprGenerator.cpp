#include "BackEnd.h"


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
    } else {
        result.value = generateBinOp(node->op, lhs.value, rhs.value);
    }

    return result;
}

ExprResult BackEnd::generateVecIntBinExpr(std::shared_ptr<BinExprAST> node) {
    if (node->op == "|") {
        return generateGenExpr(std::dynamic_pointer_cast<ScopedBinExprAST>(node));
    } else if (node->op == "&") {
        return generateFilterExpr(std::dynamic_pointer_cast<ScopedBinExprAST>(node));
    } else if (node->op == "[") {
        return generateIndexExpr(node);
    } else {
        return generateVecIntOpExpr(node);
    }
}

ExprResult BackEnd::generateVecIntOpExpr(std::shared_ptr<BinExprAST> node) {
    ExprResult lhs = generateExpr(node->lhs);
    ExprResult rhs = generateExpr(node->rhs);

    ExprResult result;
    if (node->lhs->type == "vector") {
        result.size = lhs.size;
    } else {
        result.size = rhs.size;
    }

    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    result.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, result.size);
    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, index);

    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, result.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
    builder->setInsertionPointToStart(body);

    mlir::Value lhsValue;
    mlir::Value rhsValue;
    if (node->lhs->type == "vector") {
        mlir::Value lhsPtr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, lhs.value, indexValue);
        lhsValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, lhsPtr);
        rhsValue = rhs.value;
    } else {
        lhsValue = lhs.value;
        mlir::Value rhsPtr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, rhs.value, indexValue);
        rhsValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, rhsPtr);
    }
    mlir::Value value = generateBinOp(node->op, lhsValue, rhsValue);
    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, result.value, indexValue);
    builder->create<mlir::LLVM::StoreOp>(loc, value, ptr);
    indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);
    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(merge);

    return result;
}

mlir::Value BackEnd::generateBinOp(std::string op, mlir::Value lhs, mlir::Value rhs) {
    mlir::Value returnValue;

    if (op == "*") {
        returnValue =  builder->create<mlir::LLVM::MulOp>(loc, lhs, rhs);
    } else if (op == "/") {
        mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, rhs, zero);
        rhs = builder->create<mlir::LLVM::SelectOp>(loc, comp, rhs, one);
        returnValue =  builder->create<mlir::LLVM::SDivOp>(loc, lhs, rhs);
    } else if (op == "+") {
        returnValue =  builder->create<mlir::LLVM::AddOp>(loc, lhs, rhs);
    } else if (op == "-") {
        returnValue =  builder->create<mlir::LLVM::SubOp>(loc, lhs, rhs);
    } else if (op == "==") {
        returnValue = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, lhs, rhs);
        returnValue =  builder->create<mlir::LLVM::ZExtOp>(loc, intType, returnValue);
    } else if (op == "!=") {
        returnValue = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, lhs, rhs);
        returnValue =  builder->create<mlir::LLVM::ZExtOp>(loc, intType, returnValue);
    } else if (op == "<") {
        returnValue = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, lhs, rhs);
        returnValue =  builder->create<mlir::LLVM::ZExtOp>(loc, intType, returnValue);
    } else if (op == ">") {
        returnValue = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, lhs, rhs);
        returnValue =  builder->create<mlir::LLVM::ZExtOp>(loc, intType, returnValue);
    } else {
        std::cout << "Error: Operation not found\n";
    }

    return returnValue;
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
    if (node->type == "int") {
        result.value = builder->create<mlir::LLVM::LoadOp>(loc, intType, node->var->value.value);
    } else {
        result.value = node->var->value.value;
        result.size = builder->create<mlir::LLVM::LoadOp>(loc, intType, node->var->value.size);
    }
    return result;
}
