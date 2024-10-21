#include "BackEnd.h"


/**
 * @brief Generates code for a given statement node.
 * 
 * This function identifies the type of the statement and calls the corresponding 
 * code generation function for handling declaration, assignment, condition, loop, or print statements.
 * 
 * @param node A shared pointer to the AST node representing the statement.
 */
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

/**
 * @brief Generates code for a variable declaration statement.
 * 
 * Allocates memory for the declared variable, generates the code for its initialization 
 * expression, and stores the result in the allocated memory.
 * 
 * @param node A shared pointer to the AST node representing the variable declaration.
 */
void BackEnd::generateDeclStat(std::shared_ptr<VarStatAST> node) {
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
}

/**
 * @brief Generates code for a variable assignment statement.
 * 
 * Computes the value to be assigned and stores it in the appropriate variable memory location.
 * 
 * @param node A shared pointer to the AST node representing the variable assignment.
 */
void BackEnd::generateAssignStat(std::shared_ptr<VarStatAST> node) {
    ExprResult value = generateExpr(node->expr);

    if (node->expr->type == "int") {
        builder->create<mlir::LLVM::StoreOp>(loc, value.value, node->var->value.value);
    } else {
        node->var->value.value = value.value;
        builder->create<mlir::LLVM::StoreOp>(loc, value.size, node->var->value.size);
    }
}

/**
 * @brief Generates code for a conditional (if-else) statement.
 * 
 * Generates LLVM code to evaluate the condition and branch to the appropriate block of code 
 * depending on the condition's result.
 * 
 * @param node A shared pointer to the AST node representing the conditional statement.
 */
void BackEnd::generateCondStat(std::shared_ptr<BlockStatAST> node) {
    mlir::Block* body = mainFunc.addBlock();
    mlir::Block* merge = mainFunc.addBlock();

    ExprResult cond = generateExpr(node->expr);

    cond.value = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::eq, cond.value, zero);
    builder->create<mlir::LLVM::CondBrOp>(loc, cond.value, merge, body);

    builder->setInsertionPointToStart(body);
    for (auto stat : node->stats) {
        generateStat(stat);
    }
    builder->create<mlir::LLVM::BrOp>(loc, merge);

    builder->setInsertionPointToStart(merge);
}

/**
 * @brief Generates code for a loop (while) statement.
 * 
 * Creates the necessary blocks for the loop's header, body, and merge point, and 
 * generates code to evaluate the loop condition and execute the loop body.
 * 
 * @param node A shared pointer to the AST node representing the loop statement.
 */
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
        generateStat(stat);
    }
    builder->create<mlir::LLVM::BrOp>(loc, header);

    builder->setInsertionPointToStart(merge);
}

/**
 * @brief Generates code for a print statement.
 * 
 * Evaluates the expression to be printed and prints the result. Handles both integer 
 * and vector types, printing them in the appropriate format.
 * 
 * @param node A shared pointer to the AST node representing the print statement.
 */
void BackEnd::generatePrintStat(std::shared_ptr<StatAST> node) {
    ExprResult value = generateExpr(node->expr);
    if (node->expr->type == "int") {
        generateIntPrint("intFormat", value.value);
    } else if (node->expr->type == "vector") {
        generatePrint("vectorStartFormat");

        // Allocate memory for the index of the vector and initialize to 0
        mlir::Value index = builder->create<mlir::LLVM::AllocaOp>(loc, ptrType, intType, one);
        builder->create<mlir::LLVM::StoreOp>(loc, zero, index);

        // Define the blocks for the loop that will iterate over the vector elements
        mlir::Block* header = mainFunc.addBlock();
        mlir::Block* body = mainFunc.addBlock();
        mlir::Block* postBody = mainFunc.addBlock();
        mlir::Block* merge = mainFunc.addBlock();

        // Jump to the header block to start the loop
        builder->create<mlir::LLVM::BrOp>(loc, header);
        builder->setInsertionPointToStart(header);

        // Load the current index value and compare it to the size of the vector
        mlir::Value indexValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, index);
        mlir::Value comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, value.size);
        builder->create<mlir::LLVM::CondBrOp>(loc, comp, body, merge);
        builder->setInsertionPointToStart(body);

        // Generate the pointer to the current vector element based on the index
        mlir::Value ptr = builder->create<mlir::LLVM::GEPOp>(loc, ptrType, intType, value.value, indexValue);
        mlir::Value printValue = builder->create<mlir::LLVM::LoadOp>(loc, intType, ptr);
        generateIntPrint("vectorIntFormat", printValue);

        // Increment the index and store it back into the memory location
        indexValue = builder->create<mlir::LLVM::AddOp>(loc, indexValue, one);
        builder->create<mlir::LLVM::StoreOp>(loc, indexValue, index);
        comp = builder->create<mlir::LLVM::ICmpOp>(loc, mlir::LLVM::ICmpPredicate::slt, indexValue, value.size);
        builder->create<mlir::LLVM::CondBrOp>(loc, comp, postBody, header);
        builder->setInsertionPointToStart(postBody);

        // Print space between vector elements
        generatePrint("vectorSpaceFormat");
        builder->create<mlir::LLVM::BrOp>(loc, header);

        // Set insertion point to the merge block, which is executed after the loop completes
        builder->setInsertionPointToStart(merge);
        generatePrint("vectorEndFormat");

    } else {
        std::cout << "Error: Undefined type\n";
    }
}

/**
 * @brief Generates LLVM code to print a format string.
 * 
 * @param format The format string identifier to print.
 */
void BackEnd::generatePrint(std::string format) {
    mlir::LLVM::GlobalOp formatString = module.lookupSymbol<mlir::LLVM::GlobalOp>(format);
    mlir::Value formatStringPtr = builder->create<mlir::LLVM::AddressOfOp>(loc, formatString); 
    mlir::ValueRange args = {formatStringPtr}; 
    mlir::LLVM::LLVMFuncOp printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"); 
    builder->create<mlir::LLVM::CallOp>(loc, printfFunc, args);
}

/**
 * @brief Generates LLVM code to print an integer value.
 * 
 * @param format The format string identifier for printing the integer.
 * @param value The integer value to print.
 */
void BackEnd::generateIntPrint(std::string format, mlir::Value value) {
    mlir::LLVM::GlobalOp formatString = module.lookupSymbol<mlir::LLVM::GlobalOp>(format);
    mlir::Value formatStringPtr = builder->create<mlir::LLVM::AddressOfOp>(loc, formatString); 
    mlir::ValueRange args = {formatStringPtr, value}; 
    mlir::LLVM::LLVMFuncOp printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf"); 
    builder->create<mlir::LLVM::CallOp>(loc, printfFunc, args);
}
