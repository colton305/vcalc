#include "BackEnd.h"


ExprResult BackEnd::generateVecBinExpr(std::shared_ptr<BinExprAST> node) {
    if (node->op == "[") {
        return generateVecIndexExpr(node);
    } else {
        return generateVecOpExpr(node);
    }
}

ExprResult BackEnd::generateVecOpExpr(std::shared_ptr<BinExprAST> node) {
    ExprResult lhs = generateExpr(node->lhs);
    ExprResult rhs = generateExpr(node->rhs);
    
    ExprResult result;
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, lhs.size, rhs.size);
    result.size = builder->create<mlir::LLVM::SelectOp>(loc, comp, lhs.size, rhs.size);
    result.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, result.size);

    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* lhsBody = mainFunc.addBlock();
    mlir::Block* lhsInBounds = mainFunc.addBlock();
    mlir::Block* rhsBody = mainFunc.addBlock();
    mlir::Block* rhsInBounds = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    mlir::Value lhsPtr = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    mlir::Value rhsPtr = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);

    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, index);

    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, result.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, lhsBody, merge);
    builder->setInsertionPointToStart(lhsBody);

    builder->create<mlir::LLVM::StoreOp>(loc, zero, lhsPtr);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, lhs.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, lhsInBounds, rhsBody);
    builder->setInsertionPointToStart(lhsInBounds);

    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, lhs.value, indexValue);
    mlir::Value value = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, value, lhsPtr);
    builder->create<mlir::LLVM::BrOp>(loc, rhsBody);
    builder->setInsertionPointToStart(rhsBody);

    builder->create<mlir::LLVM::StoreOp>(loc, zero, rhsPtr);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, rhs.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, rhsInBounds, body);
    builder->setInsertionPointToStart(rhsInBounds);

    ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, rhs.value, indexValue);
    value = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, value, rhsPtr);
    builder->create<mlir::LLVM::BrOp>(loc, body);
    builder->setInsertionPointToStart(body);

    mlir::Value lhsValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, lhsPtr);
    mlir::Value rhsValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, rhsPtr);
    value = generateBinOp(node->op, lhsValue, rhsValue);
    ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, result.value, indexValue);
    builder->create<mlir::LLVM::StoreOp>(loc, value, ptr);
    indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);
    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(merge);

    return result;
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
