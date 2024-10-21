#include "AST.h"
#include <iostream>

/// Defines a new variable in the current scope with a given type and name.
/// The variable is stored in the `scope` map, which tracks variables within this scope.
/// @param type The type of the variable (e.g., "int", "float").
/// @param var The name of the variable.
/// @return A shared pointer to the newly created Variable.
std::shared_ptr<Variable> Scope::define(std::string type, std::string var) {
    scope[var] = std::make_shared<Variable>(type, var);
    return scope[var];
};

/// Resolves a variable by its name, searching first in the current scope. 
/// If not found, it recurses into the parent scope until the variable is resolved or no more parent scopes exist.
/// @param var The name of the variable to resolve.
/// @return A shared pointer to the resolved Variable, or nullptr if the variable is not found.
std::shared_ptr<Variable> Scope::resolve(std::string var) {
    auto exists = scope.find(var);
    if (exists != scope.end()) {
        return exists->second;
    } else {
        return parent_scope->resolve(var);  ///< Recursively search in the parent scope.
    }
}