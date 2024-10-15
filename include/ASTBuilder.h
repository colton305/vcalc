#ifndef ANTLRINTRO_INCLUDE_ASTBUILDER_H_
#define ANTLRINTRO_INCLUDE_ASTBUILDER_H_

#include "AST.h"
#include "VCalcBaseVisitor.h"

using namespace vcalc;

class ASTBuilder : public VCalcBaseVisitor {
std::shared_ptr<BlockStatAST> current_block;

public:
    std::any visitFile(VCalcParser::FileContext *) override;
    std::any visitDecl(VCalcParser::DeclContext *) override;
    std::any visitAssign(VCalcParser::AssignContext *) override;
    std::any visitCond(VCalcParser::CondContext *) override;
    std::any visitLoop(VCalcParser::LoopContext *) override;
    std::any visitPrint(VCalcParser::PrintContext *) override;
    std::any visitMulDiv(VCalcParser::MulDivContext *) override;
    std::any visitAddSub(VCalcParser::AddSubContext *) override;
    std::any visitStrictComp(VCalcParser::StrictCompContext *) override;
    std::any visitEqComp(VCalcParser::EqCompContext *) override;
    std::any visitNumAtom(VCalcParser::NumAtomContext *) override;
    std::any visitIdAtom(VCalcParser::IdAtomContext *) override;
};

#endif
