#include "ASTBuilder.h"

/// Visits the File node in the parse tree.
/// Initializes a new root block and scope for the AST. Iterates over each statement in the file context and visits it,
/// adding the resulting AST nodes to the block's statement list.
/// @param ctx The file context from the parser.
/// @return A shared pointer to the root BlockStatAST.
std::any ASTBuilder::visitFile(VCalcParser::FileContext *ctx) {
    current_block = std::make_shared<BlockStatAST>("", nullptr);
    current_block->scope = std::make_shared<Scope>(nullptr);
    current_scope = current_block->scope;
    for (auto stat : ctx->stat()) {
        current_block->stats.push_back(std::any_cast<std::shared_ptr<StatAST>>(visit(stat)));
    }
    
    return std::static_pointer_cast<BlockStatAST>(current_block);
}

/// Visits a variable declaration node in the parse tree.
/// Defines a new variable in the current scope, visits the expression associated with the declaration, 
/// and constructs a VarStatAST to represent the declaration.
/// @param ctx The declaration context from the parser.
/// @return A shared pointer to the resulting VarStatAST.
std::any ASTBuilder::visitDecl(VCalcParser::DeclContext *ctx) {
    auto var = current_scope->define(ctx->type->getText(), ctx->ID()->getText());
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    auto decl = std::make_shared<VarStatAST>("decl", expr, var);
    
    return std::static_pointer_cast<StatAST>(decl);
}

/// Visits an assignment node in the parse tree.
/// Resolves the variable to be assigned in the current scope, visits the right-hand side expression,
/// and constructs a VarStatAST to represent the assignment.
/// @param ctx The assignment context from the parser.
/// @return A shared pointer to the resulting VarStatAST.
std::any ASTBuilder::visitAssign(VCalcParser::AssignContext *ctx) {
    auto var = current_scope->resolve(ctx->ID()->getText());
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    auto assign = std::make_shared<VarStatAST>("assign", expr, var);
    
    return std::static_pointer_cast<StatAST>(assign);
}

/// Visits a conditional statement node in the parse tree.
/// Creates a new block with its own scope to represent the condition, visits the associated expression, 
/// and adds statements within the conditional block.
/// @param ctx The conditional context from the parser.
/// @return A shared pointer to the resulting BlockStatAST representing the conditional block.
std::any ASTBuilder::visitCond(VCalcParser::CondContext *ctx) {
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    std::shared_ptr<BlockStatAST> cond_block = std::make_shared<BlockStatAST>("cond", expr);
    cond_block->parent_block = current_block;
    cond_block->scope = std::make_shared<Scope>(current_scope);
    current_block = cond_block;
    current_scope = current_block->scope;
    for (auto stat : ctx->stat()) {
        current_block->stats.push_back(std::any_cast<std::shared_ptr<StatAST>>(visit(stat)));
    }
    current_block = current_block->parent_block;
    current_scope = current_block->scope;
    
    return std::static_pointer_cast<StatAST>(cond_block);
}

/// Visits a loop statement node in the parse tree.
/// Similar to the conditional block, this creates a new block with its own scope for the loop
/// and processes the statements within the loop.
/// @param ctx The loop context from the parser.
/// @return A shared pointer to the resulting BlockStatAST representing the loop block.
std::any ASTBuilder::visitLoop(VCalcParser::LoopContext *ctx) {
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    std::shared_ptr<BlockStatAST> loop_block = std::make_shared<BlockStatAST>("loop", expr);
    loop_block->parent_block = current_block;
    loop_block->scope = std::make_shared<Scope>(current_scope);
    current_block = loop_block;
    current_scope = current_block->scope;
    for (auto stat : ctx->stat()) {
        current_block->stats.push_back(std::any_cast<std::shared_ptr<StatAST>>(visit(stat)));
    }
    current_block = current_block->parent_block;
    current_scope = current_block->scope;
    
    return std::static_pointer_cast<StatAST>(loop_block);
}

/// Visits a print statement node in the parse tree.
/// Visits the expression to be printed and returns a new StatAST representing the print operation.
/// @param ctx The print context from the parser.
/// @return A shared pointer to the resulting StatAST for the print statement.
std::any ASTBuilder::visitPrint(VCalcParser::PrintContext *ctx) {
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    
    return std::make_shared<StatAST>("print", expr);
}

/// Visits a parenthesis expression node in the parse tree.
/// Simply visits the enclosed expression and returns the resulting AST node.
/// @param ctx The parenthesis context from the parser.
/// @return A shared pointer to the resulting ExprAST.
std::any ASTBuilder::visitParen(VCalcParser::ParenContext *ctx) {
    auto ast_node = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    return std::static_pointer_cast<ExprAST>(ast_node);
}

/// Visits a generator expression node in the parse tree.
/// Creates a new scope for the iterator, visits the expressions for the generator,
/// and constructs a ScopedBinExprAST to represent the generator operation.
/// @param ctx The generator context from the parser.
/// @return A shared pointer to the resulting ScopedBinExprAST.
std::any ASTBuilder::visitGen(VCalcParser::GenContext *ctx) {
    auto ast_node = std::make_shared<ScopedBinExprAST>("vector", ctx->op->getText(), nullptr, nullptr);
    ast_node->scope = std::make_shared<Scope>(current_scope);
    current_scope = ast_node->scope;
    current_scope->define("int", ctx->ID()->getText());
    ast_node->iterator = current_scope->resolve(ctx->ID()->getText());
    std::cout << ast_node->iterator << "\n";

    std::cout << "Started gen\n";
    ast_node->lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    ast_node->rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    current_scope = current_scope->parent_scope;
    std::cout << "Completed gen\n";
    return std::static_pointer_cast<ExprAST>(ast_node);
}

/// Visits a filter expression node in the parse tree.
/// Similar to the generator node, this creates a scope for the iterator and constructs
/// a ScopedBinExprAST for the filter operation.
/// @param ctx The filter context from the parser.
/// @return A shared pointer to the resulting ScopedBinExprAST.
std::any ASTBuilder::visitFilter(VCalcParser::FilterContext *ctx) {
    auto ast_node = std::make_shared<ScopedBinExprAST>("vector", ctx->op->getText(), nullptr, nullptr);
    ast_node->scope = std::make_shared<Scope>(current_scope);
    current_scope = ast_node->scope;
    current_scope->define("int", ctx->ID()->getText());
    ast_node->iterator = current_scope->resolve(ctx->ID()->getText());
    std::cout << ast_node->iterator << "\n";

    std::cout << "Started filter\n";
    ast_node->lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    ast_node->rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    current_scope = current_scope->parent_scope;
    std::cout << "Completed filter\n";
    return std::static_pointer_cast<ExprAST>(ast_node);
}

/// Visits an index access expression node in the parse tree.
/// Visits the left-hand and right-hand side expressions and creates a BinExprAST to represent the index operation.
/// @param ctx The index context from the parser.
/// @return A shared pointer to the resulting BinExprAST.
std::any ASTBuilder::visitIndex(VCalcParser::IndexContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    auto ast_node = std::make_shared<BinExprAST>(rhs->type, ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

/// Visits a range expression node in the parse tree.
/// Visits the left-hand and right-hand side expressions and creates a BinExprAST to represent the range operation.
/// @param ctx The range context from the parser.
/// @return A shared pointer to the resulting BinExprAST.
std::any ASTBuilder::visitRange(VCalcParser::RangeContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    auto ast_node = std::make_shared<BinExprAST>("vector", ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

/// Visits a multiplication or division expression node in the parse tree.
/// Visits the left-hand and right-hand side expressions and creates a BinExprAST to represent the operation.
/// The resulting type depends on whether any operand is a vector.
/// @param ctx The multiplication/division context from the parser.
/// @return A shared pointer to the resulting BinExprAST.
std::any ASTBuilder::visitMulDiv(VCalcParser::MulDivContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));

    std::string type = "int";
    if (lhs->type == "vector" || rhs->type == "vector") {
        type = "vector";
    }
    
    auto ast_node = std::make_shared<BinExprAST>(type, ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

/// Visits an addition or subtraction expression node in the parse tree.
/// Similar to multiplication/division, it visits both operands and determines the resulting type.
/// @param ctx The addition/subtraction context from the parser.
/// @return A shared pointer to the resulting BinExprAST.
std::any ASTBuilder::visitAddSub(VCalcParser::AddSubContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    std::string type = "int";
    if (lhs->type == "vector" || rhs->type == "vector") {
        type = "vector";
    }
    
    auto ast_node = std::make_shared<BinExprAST>(type, ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

/// Visits a strict comparison expression node in the parse tree.
/// Visits both operands and creates a BinExprAST representing the comparison.
/// @param ctx The strict comparison context from the parser.
/// @return A shared pointer to the resulting BinExprAST.
std::any ASTBuilder::visitStrictComp(VCalcParser::StrictCompContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    std::string type = "int";
    if (lhs->type == "vector" || rhs->type == "vector") {
        type = "vector";
    }
    
    auto ast_node = std::make_shared<BinExprAST>(type, ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

/// Visits an equality comparison expression node in the parse tree.
/// Visits both operands and creates a BinExprAST representing the comparison.
/// @param ctx The equality comparison context from the parser.
/// @return A shared pointer to the resulting BinExprAST.
std::any ASTBuilder::visitEqComp(VCalcParser::EqCompContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    std::string type = "int";
    if (lhs->type == "vector" || rhs->type == "vector") {
        type = "vector";
    }
    
    auto ast_node = std::make_shared<BinExprAST>(type, ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

/// Visits a numeric atom node in the parse tree.
/// Creates a NumAST representing the numeric constant.
/// @param ctx The numeric atom context from the parser.
/// @return A shared pointer to the resulting NumAST.
std::any ASTBuilder::visitNumAtom(VCalcParser::NumAtomContext *ctx) {
    auto ast_node = std::make_shared<NumAST>(stoi(ctx->NUM()->getText()));
    return std::static_pointer_cast<ExprAST>(ast_node);
}

/// Visits an identifier atom node in the parse tree.
/// Resolves the variable by name in the current scope and creates a VarAST representing it.
/// @param ctx The identifier atom context from the parser.
/// @return A shared pointer to the resulting VarAST.
std::any ASTBuilder::visitIdAtom(VCalcParser::IdAtomContext *ctx) {
    auto ast_node = std::make_shared<VarAST>(current_scope->resolve(ctx->ID()->getText()));
    std::cout << current_scope->resolve(ctx->ID()->getText()) << "\n";
    return std::static_pointer_cast<ExprAST>(ast_node);
}
