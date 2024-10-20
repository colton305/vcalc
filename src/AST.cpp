#include "AST.h"
#include <iostream> 

std::shared_ptr<Variable> Scope::define(std::string type, std::string var) {
    scope[var] = std::make_shared<Variable>(type, var);
    return scope[var];
};

std::shared_ptr<Variable> Scope::resolve(std::string var) {
    auto exists = scope.find(var);
    if (exists != scope.end()) {
        return exists->second;
    } else {
        return parent_scope->resolve(var);
    }
}
