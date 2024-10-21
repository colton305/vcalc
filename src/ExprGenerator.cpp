#include "BackEnd.h"


/**
 * Generates code for a given expression node.
 * Dispatches to the appropriate method based on the type of the expression node.
 * 
 * @param node The expression node to generate code for.
 * @return The result of the expression generation.
 */
ExprResult BackEnd::generateExpr(std::shared_ptr<ExprAST> node) {
    if (auto bin = std::dynamic_pointer_cast<BinExprAST>(node)) {
        return generateBinExpr(bin);
    } else if (auto num = std::dynamic_pointer_cast<NumAST>(node)) {
        return generateNumExpr(num);
    } else if (auto var = std::dynamic_pointer_cast<VarAST>(node)) {
        return generateVarExpr(var);
    } else {
        std::cout << "Error: Expr type not found\n";
        ExprResult result;
        return result;
    }
}

/**
 * Generates code for a binary expression.
 * Determines the type of the binary operands (int or vector) and delegates to the appropriate generation function.
 * 
 * @param node The binary expression node.
 * @return The result of the binary expression generation.
 */
ExprResult BackEnd::generateBinExpr(std::shared_ptr<BinExprAST> node) {

    if (node->lhs->type == "int" && node->rhs->type == "int") {
        return generateIntBinExpr(node);
    } else if (node->lhs->type == "vector" && node->rhs->type == "vector") {
        return generateVecBinExpr(node);
    } else {
        return generateVecIntBinExpr(node);
    }
}

/**
 * Generates code for a binary expression with integer operands.
 * Handles range expressions (e.g. "..") and simple arithmetic binary operations.
 * 
 * @param node The binary expression node.
 * @return The result of the integer binary expression generation.
 */
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

/**
 * Generates code for a mixed vector and integer binary expression.
 * Handles operations such as filtering, generating vectors, and indexing.
 * 
 * @param node The binary expression node.
 * @return The result of the vector-integer binary expression generation.
 */
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

/**
 * Generates code for vector-integer operations, such as arithmetic and element-wise operations.
 * Handles dynamic allocation and iterates through the vector to apply the operation.
 * 
 * @param node The binary expression node.
 * @return The result of the vector-integer operation.
 */
ExprResult BackEnd::generateVecIntOpExpr(std::shared_ptr<BinExprAST> node) {
    // Generate the code for the left-hand side (lhs) and right-hand side (rhs) expressions
    ExprResult lhs = generateExpr(node->lhs);
    ExprResult rhs = generateExpr(node->rhs);

    ExprResult result;
    // Determine the size of the resulting vector, which will be the same as the vector operand
    if (node->lhs->type == "vector") {
        result.size = lhs.size;
    } else {
        result.size = rhs.size;
    }

    // Create new basic blocks for the loop: one for the loop header, one for the loop body, and one for merging after the loop
    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    result.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, result.size);
    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, index);

    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    // Load the current value of the index and compare it with the size of the vector
    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, result.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
    builder->setInsertionPointToStart(body);

    mlir::Value lhsValue;
    mlir::Value rhsValue;
    // If lhs is a vector, load the value from the vector at the current index
    // Otherwise, lhs is a scalar, so use the scalar value directly
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

/**
 * Generates the appropriate LLVM operation based on the binary operator.
 * Supports arithmetic and comparison operations.
 * 
 * @param op The binary operator as a string (e.g., "+", "*").
 * @param lhs The left-hand side operand.
 * @param rhs The right-hand side operand.
 * @return The result of the binary operation.
 */
mlir::Value BackEnd::generateBinOp(std::string op, mlir::Value lhs, mlir::Value rhs) {
    mlir::Value returnValue;

    if (op == "*") {
        returnValue = builder->create<mlir::LLVM::MulOp>(loc, lhs, rhs);
    } else if (op == "/") {
        mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, rhs, zero);
        rhs = builder->create<mlir::LLVM::SelectOp>(loc, comp, rhs, one);
        returnValue = builder->create<mlir::LLVM::SDivOp>(loc, lhs, rhs);
    } else if (op == "+") {
        returnValue = builder->create<mlir::LLVM::AddOp>(loc, lhs, rhs);
    } else if (op == "-") {
        returnValue = builder->create<mlir::LLVM::SubOp>(loc, lhs, rhs);
    } else if (op == "==") {
        returnValue = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, lhs, rhs);
        returnValue = builder->create<mlir::LLVM::ZExtOp>(loc, intType, returnValue);
    } else if (op == "!=") {
        returnValue = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, lhs, rhs);
        returnValue = builder->create<mlir::LLVM::ZExtOp>(loc, intType, returnValue);
    } else if (op == "<") {
        returnValue = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, lhs, rhs);
        returnValue = builder->create<mlir::LLVM::ZExtOp>(loc, intType, returnValue);
    } else if (op == ">") {
        returnValue = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, lhs, rhs);
        returnValue = builder->create<mlir::LLVM::ZExtOp>(loc, intType, returnValue);
    } else {
        std::cout << "Error: Operation not found\n";
    }

    return returnValue;
}

/**
 * Generates code for indexing into a vector using an index expression.
 * The index is checked for validity, ensuring it is within the bounds of the vector.
 * If the index is valid, the value at the index is loaded from the vector.
 * 
 * - If the index is out of bounds or negative, the result will not change (it remains zero).
 * - Otherwise, the value at the given index is returned.
 * 
 * @param node The binary expression node containing a vector and an index expression.
 * @return The result of accessing the vector at the specified index.
 */
ExprResult BackEnd::generateIndexExpr(std::shared_ptr<BinExprAST> node) {
    // Generate code for both the vector and index expressions
    ExprResult vector = generateExpr(node->lhs);
    ExprResult index = generateExpr(node->rhs);

    // Create new basic blocks for the index check and merge
    mlir::Block* ib = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    // Allocate space for the result value, initializing it to zero
    mlir::Value resultPtr = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, resultPtr);

    // Check if the index is valid (non-negative and within vector bounds)
    mlir::Value positive = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sge, index.value, zero);
    mlir::Value inBounds = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, index.value, vector.size);
    mlir::Value comp = builder->create<mlir::LLVM::AndOp>(loc, positive, inBounds);

    // Conditionally branch based on index validity: if valid, go to the index block; otherwise, go to merge
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, ib, merge);

    // Begin generating code for the index block
    builder->setInsertionPointToStart(ib);

    // Load the value from the vector at the specified index and store it in resultPtr
    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, vector.value, index.value);
    mlir::Value value = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, value, resultPtr);

    // Branch to the merge block
    builder->create<mlir::LLVM::BrOp>(loc, merge);

    // Continue code generation in the merge block
    builder->setInsertionPointToStart(merge);

    // Load the final result and return it
    ExprResult result;
    result.value = builder->create<mlir::LLVM::LoadOp>(loc, intType, resultPtr);
    return result;
}

/**
 * Generates code for creating a range expression (i.e., an array of integers between two bounds).
 * If the lower bound is less than or equal to the upper bound, the result will be a vector of integers.
 * Otherwise, an empty array (size 0) is returned.
 * 
 * @param lowerBound The lower bound of the range.
 * @param upperBound The upper bound of the range.
 * @return The result containing the array of integers representing the range.
 */
ExprResult BackEnd::generateRangeExpr(mlir::Value lowerBound, mlir::Value upperBound) {
    ExprResult result;

    // Check if the lower bound is less than or equal to the upper bound
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sle, lowerBound, upperBound);

    // Compute the size of the range (upperBound - lowerBound + 1), ensuring a non-negative size
    result.size = builder->create<mlir::LLVM::SubOp>(loc, upperBound, lowerBound);
    result.size = builder->create<mlir::LLVM::AddOp>(loc, result.size, one);
    result.size = builder->create<mlir::LLVM::SelectOp>(loc, comp, result.size, zero);

    // Ensure that the array size is at least 1
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, result.size, one);
    mlir::Value arraySize = builder->create<mlir::LLVM::SelectOp>(loc, comp, result.size, one);

    // Create new basic blocks for the loop header, body, and merge
    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    // Allocate memory for the result array
    result.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, arraySize);

    // Allocate memory for the index counter, initialized to the lower bound
    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, lowerBound, index);

    // Begin the loop by branching to the loop header
    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    // Check if the current index is less than or equal to the upper bound
    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sle, indexValue, upperBound);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);

    // Generate code for the loop body
    builder->setInsertionPointToStart(body);

    // Store the current index in the result array
    mlir::Value relativeIndex = builder->create<mlir::LLVM::SubOp>(loc, indexValue, lowerBound);
    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, result.value, relativeIndex);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, ptr);

    // Increment the index and store it back
    indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);

    // Jump back to the loop header to check the condition again
    builder->create<mlir::LLVM::BrOp>(loc, header);

    // Continue code generation in the merge block
    builder->setInsertionPointToStart(merge);
    return result;
}

/**
 * Generates code for a numeric expression, returning a constant integer value.
 * 
 * @param node The numeric AST node representing the constant integer value.
 * @return The result containing the constant integer value.
 */
ExprResult BackEnd::generateNumExpr(std::shared_ptr<NumAST> node) {
    ExprResult result;

    // Generate an LLVM constant operation for the integer value
    result.value = builder->create<mlir::LLVM::ConstantOp>(loc, intType, node->value);

    return result;
}

/**
 * Generates code for a variable expression.
 * The variable can either represent an integer or a vector, and the corresponding value is loaded.
 * 
 * @param node The variable AST node containing information about the variable (name, type).
 * @return The result containing the variable's value.
 */
ExprResult BackEnd::generateVarExpr(std::shared_ptr<VarAST> node) {
    ExprResult result;

    // If the variable is an integer, load its value
    if (node->type == "int") {
        result.value = builder->create<mlir::LLVM::LoadOp>(loc, intType, node->var->value.value);
    } 
    // Otherwise, assume the variable is a vector and load its value and size
    else {
        result.value = node->var->value.value;
        result.size = builder->create<mlir::LLVM::LoadOp>(loc, intType, node->var->value.size);
    }

    return result;
}
