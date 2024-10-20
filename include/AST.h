#include "mlir/IR/Value.h"

#include <string>
#include <memory>
#include <vector>
#include <map>

struct ExprResult {
    mlir::Value value;
    mlir::Value size;
};

class Variable {
public:
    std::string type;
    std::string name;
    ExprResult value;

    Variable(std::string type, std::string name) : type(type), name(name) {}
};

class Scope {
public:
    std::shared_ptr<Variable> define(std::string type, std::string var);
    std::shared_ptr<Variable> resolve(std::string var);

    std::shared_ptr<Scope> parent_scope;
    std::map<std::string, std::shared_ptr<Variable>> scope;

    Scope(std::shared_ptr<Scope> parent_scope) : parent_scope(parent_scope) {}
};

class ExprAST {
public:
    std::string type;

    ExprAST(std::string type) : type(type) {}

    virtual ~ExprAST() = default;
};

class StatAST {
public:
    std::string op;
    std::shared_ptr<ExprAST> expr;

    StatAST(std::string op, std::shared_ptr<ExprAST> expr) : op(op), expr(expr) {}

    virtual ~StatAST() = default;
};

class VarStatAST : public StatAST {
public:
    std::shared_ptr<Variable> var;

    VarStatAST(std::string op, std::shared_ptr<ExprAST> expr, std::shared_ptr<Variable> var) : StatAST(op, expr), var(var) {}

    virtual ~VarStatAST() = default;
};

class BlockStatAST : public StatAST {
public:
    std::shared_ptr<BlockStatAST> parent_block;
    std::vector<std::shared_ptr<StatAST>> stats;
    std::shared_ptr<Scope> scope;

    BlockStatAST(std::string op, std::shared_ptr<ExprAST> expr) : StatAST(op, expr) {}

    virtual ~BlockStatAST() = default;
};

class BinExprAST : public ExprAST {
public:
    std::string op;
    std::shared_ptr<ExprAST> lhs, rhs;

    BinExprAST(std::string type, std::string op, std::shared_ptr<ExprAST> lhs, std::shared_ptr<ExprAST> rhs)
        : ExprAST(type), op(op), lhs(lhs), rhs(rhs) {}

    virtual ~BinExprAST() = default;
};

class ScopedBinExprAST : public BinExprAST {
public:
    std::shared_ptr<Scope> scope;
    std::shared_ptr<Variable> iterator;

    ScopedBinExprAST(std::string type, std::string op, std::shared_ptr<ExprAST> lhs, std::shared_ptr<ExprAST> rhs) : BinExprAST(type, op, lhs, rhs) {}

    virtual ~ScopedBinExprAST() = default;
};

class NumAST : public ExprAST {
public:
    int value;

    NumAST(int value) : ExprAST("int"), value(value) {}

    virtual ~NumAST() = default;
};

class VarAST : public ExprAST {
public:
    std::shared_ptr<Variable> var;

    VarAST(std::shared_ptr<Variable> var) : ExprAST(var->type), var(var) {}

    virtual ~VarAST() = default;
};
