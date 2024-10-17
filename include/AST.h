#include "mlir/IR/Value.h"

#include <string>
#include <memory>
#include <vector>
#include <map>


class Variable {
public:
    std::string name;
    mlir::Value value;
    mlir::Block* block;

    Variable(std::string name) : name(name) {}
};

class ExprAST {
public:
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
    std::shared_ptr<Variable> define(std::string var);
    std::shared_ptr<Variable> resolve(std::string var);

    std::shared_ptr<BlockStatAST> parent_block;
    std::vector<std::shared_ptr<StatAST>> stats;
    std::map<std::string, std::shared_ptr<Variable>> scope;

    BlockStatAST(std::string op, std::shared_ptr<ExprAST> expr) : StatAST(op, expr) {}

    virtual ~BlockStatAST() = default;
};

class BinExprAST : public ExprAST {
public:
    std::string op;
    std::shared_ptr<ExprAST> lhs, rhs;

    BinExprAST(std::string op, std::shared_ptr<ExprAST> lhs, std::shared_ptr<ExprAST> rhs)
        : op(op), lhs(lhs), rhs(rhs) {}

    virtual ~BinExprAST() = default;
};

class NumAST : public ExprAST {
public:
    int value;

    NumAST(int value) : value(value) {}

    virtual ~NumAST() = default;
};

class VarAST : public ExprAST {
public:
    std::shared_ptr<Variable> var;

    VarAST(std::shared_ptr<Variable> var) : var(var) {}

    virtual ~VarAST() = default;
};
