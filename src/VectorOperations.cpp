#include "BackEnd.h"


/**
 * Generates code for a binary expression involving vectors. If the operator is an index (`[ ]`), 
 * it generates a vector index expression. Otherwise, it generates a vector operation expression.
 *
 * @param node The binary expression AST node representing the vector operation or index.
 * @return ExprResult The result of the expression, including value and size.
 */
ExprResult BackEnd::generateVecBinExpr(std::shared_ptr<BinExprAST> node) {
    if (node->op == "[") {
        return generateVecIndexExpr(node);
    } else {
        return generateVecOpExpr(node);
    }
}

/**
 * Generates code for a binary expression involving vector operations (like element-wise addition, etc.).
 * This creates an LLVM-based conditional block structure for processing the vectors.
 *
 * @param node The binary expression AST node representing the vector operation.
 * @return ExprResult The result of the vector operation, including the calculated value and size.
 */
ExprResult BackEnd::generateVecOpExpr(std::shared_ptr<BinExprAST> node) {
    // Generate the left-hand side (lhs) and right-hand side (rhs) expressions
    ExprResult lhs = generateExpr(node->lhs);
    ExprResult rhs = generateExpr(node->rhs);
    
    ExprResult result;
    
    // Create a comparison operation for vector sizes
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, lhs.size, rhs.size);
    
    // Select the larger of the two sizes
    result.size = builder->create<mlir::LLVM::SelectOp>(loc, comp, lhs.size, rhs.size);
    
    // Allocate memory for the result
    mlir::Value arraySize = builder->create<mlir::LLVM::MulOp>(loc, builder->create<mlir::LLVM::SExtOp>(loc, arraySizeType, result.size), intSize);
    result.value = builder->create<mlir::LLVM::CallOp>(loc, mallocFn, mlir::ValueRange{arraySize}).getResult();

    // Create basic blocks for vector processing
    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* lhsBody = mainFunc.addBlock();
    mlir::Block* lhsInBounds = mainFunc.addBlock();
    mlir::Block* rhsBody = mainFunc.addBlock();
    mlir::Block* rhsInBounds = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    // Allocate pointers for lhs and rhs
    mlir::Value lhsPtr = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    mlir::Value rhsPtr = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);

    // Allocate and initialize an index
    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, index);

    // Create loops for processing elements in the vectors
    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    // Compare the current index with the vector size
    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, result.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, lhsBody, merge);
    builder->setInsertionPointToStart(lhsBody);

    // Process the lhs (left-hand side) vector
    builder->create<mlir::LLVM::StoreOp>(loc, zero, lhsPtr);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, lhs.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, lhsInBounds, rhsBody);
    builder->setInsertionPointToStart(lhsInBounds);

    // Load the lhs value at the current index
    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, lhs.value, indexValue);
    mlir::Value value = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, value, lhsPtr);
    builder->create<mlir::LLVM::BrOp>(loc, rhsBody);
    builder->setInsertionPointToStart(rhsBody);

    // Process the rhs (right-hand side) vector
    builder->create<mlir::LLVM::StoreOp>(loc, zero, rhsPtr);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, rhs.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, rhsInBounds, body);
    builder->setInsertionPointToStart(rhsInBounds);

    // Load the rhs value at the current index
    ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, rhs.value, indexValue);
    value = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, value, rhsPtr);
    builder->create<mlir::LLVM::BrOp>(loc, body);
    builder->setInsertionPointToStart(body);

    // Perform the binary operation between lhs and rhs values
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

/**
 * Generates code for a scoped binary expression, where the lhs (left-hand side) is a vector 
 * and the rhs (right-hand side) is an expression. This function processes the lhs and rhs 
 * iteratively, updating the vector elements.
 *
 * @param node The scoped binary expression AST node, including the lhs, rhs, and the iterator.
 * @return ExprResult The result of the expression, with the updated vector and size.
 */
ExprResult BackEnd::generateGenExpr(std::shared_ptr<ScopedBinExprAST> node) {
    // Generate the left-hand side (lhs) expression (vector)
    ExprResult vector = generateExpr(node->lhs);
    node->iterator->value.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);

    ExprResult result;
    result.size = vector.size;

    // Generate conditional blocks for the iteration
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sgt, result.size, one);
    mlir::Value arraySize = builder->create<mlir::LLVM::SelectOp>(loc, comp, result.size, one);
    arraySize = builder->create<mlir::LLVM::MulOp>(loc, builder->create<mlir::LLVM::SExtOp>(loc, arraySizeType, arraySize), intSize);
    result.value = builder->create<mlir::LLVM::CallOp>(loc, mallocFn, mlir::ValueRange{arraySize}).getResult();

    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, index);

    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    // Loop until all elements are processed
    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, vector.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
    builder->setInsertionPointToStart(body);

    // Process each element in the lhs (vector)
    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, vector.value, indexValue);
    mlir::Value vectorValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, vectorValue, node->iterator->value.value);

    // Generate the right-hand side expression
    ExprResult rhs = generateExpr(node->rhs);
    ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, result.value, indexValue);
    builder->create<mlir::LLVM::StoreOp>(loc, rhs.value, ptr);
    
    indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);
    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(merge);

    return result;
}

/**
 * Generates code for filtering elements from a vector based on a condition. The function iterates through the 
 * vector, applies the condition (from the right-hand side of the expression), and stores the elements that satisfy 
 * the condition in the result.
 *
 * @param node The ScopedBinExprAST node representing the filtering expression, where lhs is the vector to be filtered 
 *             and rhs is the condition expression.
 * @return ExprResult The result of the filtering operation, which contains the filtered vector and its size.
 */
ExprResult BackEnd::generateFilterExpr(std::shared_ptr<ScopedBinExprAST> node) {
    // Generate the left-hand side expression (vector)
    ExprResult vector = generateExpr(node->lhs);

    // Allocate space for the iterator and result vector
    node->iterator->value.value = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    ExprResult result;
    mlir::Value arraySize = builder->create<mlir::LLVM::MulOp>(loc, builder->create<mlir::LLVM::SExtOp>(loc, arraySizeType, vector.size), intSize);
    result.value = builder->create<mlir::LLVM::CallOp>(loc, mallocFn, mlir::ValueRange{arraySize}).getResult();

    // Create basic blocks for loop structure
    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* insert = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    // Initialize loop variables (index and array size)
    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    arraySize = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, index);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, arraySize);

    // Start the loop at the header block
    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    // Check if index is less than the vector size
    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, vector.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
    builder->setInsertionPointToStart(body);

    // Store the vector element at the current index in the iterator
    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, vector.value, indexValue);
    mlir::Value vectorValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, vectorValue, node->iterator->value.value);

    // Increment the index
    indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);

    // Generate the right-hand side condition and check if it is non-zero (true)
    ExprResult rhs = generateExpr(node->rhs);
    comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::ne, rhs.value, zero);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, insert, header);
    builder->setInsertionPointToStart(insert);

    // If condition is true, store the element in the result
    mlir::Value sizeValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, arraySize);
    ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, result.value, sizeValue);
    builder->create<mlir::LLVM::StoreOp>(loc, vectorValue, ptr);
    sizeValue = builder->create<mlir::LLVM::AddOp>(loc, sizeValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, sizeValue, arraySize);

    // Continue the loop
    builder->create<mlir::LLVM::BrOp>(loc, header);

    builder->setInsertionPointToStart(merge);
    result.size = builder->create<mlir::LLVM::LoadOp>(loc, intType, arraySize);

    return result;
}

/**
 * Generates code for indexing into a vector using an index vector (right-hand side of the expression). 
 * The function iterates over the index vector, checks for valid index values, and stores the corresponding 
 * elements from the vector into the result.
 *
 * @param node The BinExprAST node representing the indexing operation, where lhs is the vector to index into, 
 *             and rhs is the index vector.
 * @return ExprResult The result of the indexing operation, which contains the indexed elements and their size.
 */
ExprResult BackEnd::generateVecIndexExpr(std::shared_ptr<BinExprAST> node) {
    // Generate the left-hand side (vector) and right-hand side (index vector) expressions
    ExprResult vector = generateExpr(node->lhs);
    ExprResult indexVector = generateExpr(node->rhs);

    // Allocate space for the result and index vector size
    ExprResult result;
    mlir::Value arraySize = builder->create<mlir::LLVM::MulOp>(loc, builder->create<mlir::LLVM::SExtOp>(loc, arraySizeType, indexVector.size), intSize);
    result.value = builder->create<mlir::LLVM::CallOp>(loc, mallocFn, mlir::ValueRange{arraySize}).getResult();
    result.size = indexVector.size;

    // Create basic blocks for loop structure
    mlir::Block* header = mainFunc.addBlock();
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* ib = mainFunc.addBlock();
    mlir::Block* postBody = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    // Initialize loop variables (index)
    mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, index);

    // Start the loop at the header block
    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(header);

    // Check if index is less than the result size (index vector size)
    mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
    mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, result.size);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
    builder->setInsertionPointToStart(body);

    // Store result pointer
    mlir::Value resultPtr = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
    builder->create<mlir::LLVM::StoreOp>(loc, zero, resultPtr);

    // Load index vector element and check for valid index
    mlir::Value vectorIndex = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, indexVector.value, indexValue);
    mlir::Value vectorIndexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, vectorIndex);
    mlir::Value positive = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::sge, vectorIndexValue, zero);
    mlir::Value inBounds = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, vectorIndexValue, vector.size);
    comp = builder->create<mlir::LLVM::AndOp>(loc, positive, inBounds);
    builder->create<mlir::LLVM::CondBrOp>(loc, comp, ib, postBody);
    builder->setInsertionPointToStart(ib);

    // If index is valid, load the vector element and store in result
    mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, vector.value, vectorIndexValue);
    mlir::Value value = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
    builder->create<mlir::LLVM::StoreOp>(loc, value, resultPtr);
    builder->create<mlir::LLVM::BrOp>(loc, postBody);
    builder->setInsertionPointToStart(postBody);

    // Store result in the final result array
    ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, result.value, indexValue);
    mlir::Value resultValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, resultPtr);
    builder->create<mlir::LLVM::StoreOp>(loc, resultValue, ptr);

    // Increment index and continue loop
    indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
    builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);
    builder->create<mlir::LLVM::BrOp>(loc, header);
    builder->setInsertionPointToStart(merge);

    return result;
}
