#include "mlir/IR/Value.h"

#include <string>
#include <memory>
#include <vector>
#include <map>

/// Struct to hold the result of an expression.
/// Contains an MLIR value and its corresponding size.
struct ExprResult {
    mlir::Value value;  ///< The MLIR value representing the expression result.
    mlir::Value size;   ///< The size of the result, if applicable.
};

/// Class representing a variable with a type, name, and value.
class Variable {
public:
    std::string type;   ///< The type of the variable (e.g., "int", "float").
    std::string name;   ///< The name of the variable.
    ExprResult value;   ///< The value of the variable, stored as an ExprResult.

    /// Constructor to initialize a variable with a type and name.
    /// @param type The type of the variable.
    /// @param name The name of the variable.
    Variable(std::string type, std::string name) : type(type), name(name) {}
};

/// Class representing a scope in which variables are defined and resolved.
/// A scope may have a parent scope.
class Scope {
public:
    /// Define a new variable in the current scope.
    /// @param type The type of the variable.
    /// @param var The name of the variable.
    /// @return A shared pointer to the defined Variable.
    std::shared_ptr<Variable> define(std::string type, std::string var);

    /// Resolve a variable by its name within the current scope or its parent scope.
    /// @param var The name of the variable.
    /// @return A shared pointer to the resolved Variable, or nullptr if not found.
    std::shared_ptr<Variable> resolve(std::string var);

    std::shared_ptr<Scope> parent_scope;  ///< The parent scope, or nullptr if no parent.
    std::map<std::string, std::shared_ptr<Variable>> scope;  ///< Map of variable names to Variables in this scope.

    /// Constructor to initialize a scope with a given parent scope.
    /// @param parent_scope The parent scope of the current scope, or nullptr for global scope.
    Scope(std::shared_ptr<Scope> parent_scope) : parent_scope(parent_scope) {}
};

/// Abstract base class for expression nodes in the AST.
class ExprAST {
public:
    std::string type;   ///< The type of the expression.

    /// Constructor to initialize an expression with a type.
    /// @param type The type of the expression.
    ExprAST(std::string type) : type(type) {}

    /// Virtual destructor to ensure proper cleanup of derived classes.
    virtual ~ExprAST() = default;
};

/// Class representing a statement in the AST.
class StatAST {
public:
    std::string op;   ///< The operation of the statement (e.g., assignment, return).
    std::shared_ptr<ExprAST> expr;   ///< The expression associated with the statement.

    /// Constructor to initialize a statement with an operation and expression.
    /// @param op The operation of the statement.
    /// @param expr A shared pointer to the expression.
    StatAST(std::string op, std::shared_ptr<ExprAST> expr) : op(op), expr(expr) {}

    /// Virtual destructor to ensure proper cleanup of derived classes.
    virtual ~StatAST() = default;
};

/// Class representing a variable statement in the AST.
class VarStatAST : public StatAST {
public:
    std::shared_ptr<Variable> var;  ///< The variable associated with the statement.

    /// Constructor to initialize a variable statement with an operation, expression, and variable.
    /// @param op The operation of the statement.
    /// @param expr A shared pointer to the expression.
    /// @param var A shared pointer to the variable.
    VarStatAST(std::string op, std::shared_ptr<ExprAST> expr, std::shared_ptr<Variable> var) 
        : StatAST(op, expr), var(var) {}

    /// Virtual destructor to ensure proper cleanup.
    virtual ~VarStatAST() = default;
};

/// Class representing a block statement in the AST, which may contain multiple statements and its own scope.
class BlockStatAST : public StatAST {
public:
    std::shared_ptr<BlockStatAST> parent_block;  ///< The parent block, or nullptr if no parent.
    std::vector<std::shared_ptr<StatAST>> stats; ///< The list of statements in this block.
    std::shared_ptr<Scope> scope;                ///< The scope associated with this block.

    /// Constructor to initialize a block statement with an operation and expression.
    /// @param op The operation of the statement.
    /// @param expr A shared pointer to the expression.
    BlockStatAST(std::string op, std::shared_ptr<ExprAST> expr) : StatAST(op, expr) {}

    /// Virtual destructor to ensure proper cleanup.
    virtual ~BlockStatAST() = default;
};

/// Class representing a binary expression in the AST (e.g., addition, subtraction).
class BinExprAST : public ExprAST {
public:
    std::string op;               ///< The operator of the binary expression (e.g., "+", "-").
    std::shared_ptr<ExprAST> lhs; ///< The left-hand side of the binary expression.
    std::shared_ptr<ExprAST> rhs; ///< The right-hand side of the binary expression.

    /// Constructor to initialize a binary expression with a type, operator, and two sub-expressions.
    /// @param type The type of the binary expression.
    /// @param op The operator of the binary expression.
    /// @param lhs A shared pointer to the left-hand side expression.
    /// @param rhs A shared pointer to the right-hand side expression.
    BinExprAST(std::string type, std::string op, std::shared_ptr<ExprAST> lhs, std::shared_ptr<ExprAST> rhs)
        : ExprAST(type), op(op), lhs(lhs), rhs(rhs) {}

    /// Virtual destructor to ensure proper cleanup.
    virtual ~BinExprAST() = default;
};

/// Class representing a scoped binary expression, which includes a scope and an iterator variable.
class ScopedBinExprAST : public BinExprAST {
public:
    std::shared_ptr<Scope> scope;     ///< The scope associated with the binary expression.
    std::shared_ptr<Variable> iterator; ///< The iterator variable, if applicable.

    /// Constructor to initialize a scoped binary expression with a type, operator, and sub-expressions.
    /// @param type The type of the binary expression.
    /// @param op The operator of the binary expression.
    /// @param lhs A shared pointer to the left-hand side expression.
    /// @param rhs A shared pointer to the right-hand side expression.
    ScopedBinExprAST(std::string type, std::string op, std::shared_ptr<ExprAST> lhs, std::shared_ptr<ExprAST> rhs)
        : BinExprAST(type, op, lhs, rhs) {}

    /// Virtual destructor to ensure proper cleanup.
    virtual ~ScopedBinExprAST() = default;
};

/// Class representing a numeric literal in the AST.
class NumAST : public ExprAST {
public:
    int value;   ///< The value of the numeric literal.

    /// Constructor to initialize a numeric literal with a value.
    /// @param value The integer value of the numeric literal.
    NumAST(int value) : ExprAST("int"), value(value) {}

    /// Virtual destructor to ensure proper cleanup.
    virtual ~NumAST() = default;
};

/// Class representing a variable expression in the AST.
class VarAST : public ExprAST {
public:
    std::shared_ptr<Variable> var;  ///< The variable associated with the expression.

    /// Constructor to initialize a variable expression with a variable.
    /// @param var A shared pointer to the variable.
    VarAST(std::shared_ptr<Variable> var) : ExprAST(var->type), var(var) {}

    /// Virtual destructor to ensure proper cleanup.
    virtual ~VarAST() = default;
};
