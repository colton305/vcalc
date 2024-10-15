#include "AST.h"


void BlockStatAST::define(std::string var) {
    scope[var] = std::make_shared<Variable>();
};

std::shared_ptr<Variable> BlockStatAST::resolve(std::string var) {
    return scope[var];
}
