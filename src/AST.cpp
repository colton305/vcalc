#include "AST.h"
#include <iostream> 

std::shared_ptr<Variable> BlockStatAST::define(std::string var) {
    scope[var] = std::make_shared<Variable>(var);
    return scope[var];
};

std::shared_ptr<Variable> BlockStatAST::resolve(std::string var) {
    std::cout << "Here\n";
    auto exists = scope.find(var);
    if (exists != scope.end()) {
        return exists->second;
    } else {
        return parent_block->resolve(var);
    }
}
