#include "ASTBuilder.h"


std::any ASTBuilder::visitFile(VCalcParser::FileContext *ctx) {
    current_block = std::make_shared<BlockStatAST>("", nullptr);
    current_block->scope = std::make_shared<Scope>(nullptr);
    current_scope = current_block->scope;
    for (auto stat : ctx->stat()) {
        current_block->stats.push_back(std::any_cast<std::shared_ptr<StatAST>>(visit(stat)));
    }
    
    return std::static_pointer_cast<BlockStatAST>(current_block);
}

std::any ASTBuilder::visitDecl(VCalcParser::DeclContext *ctx) {
    auto var = current_scope->define(ctx->type->getText(), ctx->ID()->getText());
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    auto decl = std::make_shared<VarStatAST>("decl", expr, var);
    
    return std::static_pointer_cast<StatAST>(decl);
}

std::any ASTBuilder::visitAssign(VCalcParser::AssignContext *ctx) {
    auto var = current_scope->resolve(ctx->ID()->getText());
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    auto assign = std::make_shared<VarStatAST>("assign", expr, var);
    
    return std::static_pointer_cast<StatAST>(assign);
}

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

std::any ASTBuilder::visitPrint(VCalcParser::PrintContext *ctx) {
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    
    return std::make_shared<StatAST>("print", expr);
}

std::any ASTBuilder::visitParen(VCalcParser::ParenContext *ctx) {
    auto ast_node = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    return std::static_pointer_cast<ExprAST>(ast_node);
}

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

std::any ASTBuilder::visitIndex(VCalcParser::IndexContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    auto ast_node = std::make_shared<BinExprAST>(rhs->type, ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

std::any ASTBuilder::visitRange(VCalcParser::RangeContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    auto ast_node = std::make_shared<BinExprAST>("vector", ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

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

std::any ASTBuilder::visitNumAtom(VCalcParser::NumAtomContext *ctx) {
    auto ast_node = std::make_shared<NumAST>(stoi(ctx->NUM()->getText()));
    return std::static_pointer_cast<ExprAST>(ast_node);
}

std::any ASTBuilder::visitIdAtom(VCalcParser::IdAtomContext *ctx) {
    auto ast_node = std::make_shared<VarAST>(current_scope->resolve(ctx->ID()->getText()));
    std::cout << current_scope->resolve(ctx->ID()->getText()) << "\n";
    return std::static_pointer_cast<ExprAST>(ast_node);
}
