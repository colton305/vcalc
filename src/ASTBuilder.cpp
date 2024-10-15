#include "ASTBuilder.h"


std::any ASTBuilder::visitFile(VCalcParser::FileContext *ctx) {
    current_block = std::make_shared<BlockStatAST>("", nullptr);
    for (auto stat : ctx->stat()) {
        current_block->stats.push_back(std::any_cast<std::shared_ptr<StatAST>>(visit(stat)));
    }
    
    return std::static_pointer_cast<StatAST>(current_block);
}

std::any ASTBuilder::visitDecl(VCalcParser::DeclContext *ctx) {
    current_block->define(ctx->ID()->getText());
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    
    return std::make_shared<StatAST>("decl", expr);
}

std::any ASTBuilder::visitAssign(VCalcParser::AssignContext *ctx) {
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    
    return std::make_shared<StatAST>("assign", expr);
}

std::any ASTBuilder::visitCond(VCalcParser::CondContext *ctx) {
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    std::shared_ptr<BlockStatAST> cond_block = std::make_shared<BlockStatAST>("cond", expr);
    cond_block->parent_block = current_block;
    current_block = cond_block;
    for (auto stat : ctx->stat()) {
        current_block->stats.push_back(std::any_cast<std::shared_ptr<StatAST>>(visit(stat)));
    }
    
    return std::static_pointer_cast<StatAST>(current_block);
}

std::any ASTBuilder::visitLoop(VCalcParser::LoopContext *ctx) {
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    std::shared_ptr<BlockStatAST> loop_block = std::make_shared<BlockStatAST>("loop", expr);
    loop_block->parent_block = current_block;
    current_block = loop_block;
    for (auto stat : ctx->stat()) {
        current_block->stats.push_back(std::any_cast<std::shared_ptr<StatAST>>(visit(stat)));
    }
    
    return std::static_pointer_cast<StatAST>(current_block);
}

std::any ASTBuilder::visitPrint(VCalcParser::PrintContext *ctx) {
    auto expr = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr()));
    
    return std::make_shared<StatAST>("print", expr);
}

std::any ASTBuilder::visitMulDiv(VCalcParser::MulDivContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    auto ast_node = std::make_shared<BinExprAST>(ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

std::any ASTBuilder::visitAddSub(VCalcParser::AddSubContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    auto ast_node = std::make_shared<BinExprAST>(ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

std::any ASTBuilder::visitStrictComp(VCalcParser::StrictCompContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    auto ast_node = std::make_shared<BinExprAST>(ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

std::any ASTBuilder::visitEqComp(VCalcParser::EqCompContext *ctx) {
    auto lhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(0)));
    auto rhs = std::any_cast<std::shared_ptr<ExprAST>>(visit(ctx->expr(1)));
    
    auto ast_node = std::make_shared<BinExprAST>(ctx->op->getText(), lhs, rhs);
    return std::static_pointer_cast<ExprAST>(ast_node);
}

std::any ASTBuilder::visitNumAtom(VCalcParser::NumAtomContext *ctx) {
    auto ast_node = std::make_shared<NumAST>(stoi(ctx->NUM()->getText()));
    return std::static_pointer_cast<ExprAST>(ast_node);
}

std::any ASTBuilder::visitIdAtom(VCalcParser::IdAtomContext *ctx) {
    auto ast_node = std::make_shared<VarAST>(current_block->resolve(ctx->ID()->getText()));
    return std::static_pointer_cast<ExprAST>(ast_node);
}
