#include <string>
#include <memory>
#include <vector>
#include <map>


class Variable {
public:
    std::string name;
};

class ExprAST {};

class StatAST {
public:
    std::string op;
    std::shared_ptr<ExprAST> expr;

    StatAST(std::string op, std::shared_ptr<ExprAST> expr) : op(op), expr(expr) {}
};

class BlockStatAST : public StatAST {
public:
    void define(std::string var);
    std::shared_ptr<Variable> resolve(std::string var);

    std::shared_ptr<BlockStatAST> parent_block;
    std::vector<std::shared_ptr<StatAST>> stats;
    std::map<std::string, std::shared_ptr<Variable>> scope;

    BlockStatAST(std::string op, std::shared_ptr<ExprAST> expr) : StatAST(op, expr) {}
};

class BinExprAST : public ExprAST {
public:
    std::string op;
    std::shared_ptr<ExprAST> lhs, rhs;

    BinExprAST(std::string op, std::shared_ptr<ExprAST> lhs, std::shared_ptr<ExprAST> rhs)
        : op(op), lhs(lhs), rhs(rhs) {}
};

class NumAST : public ExprAST {
public:
    int value;

    NumAST(int value) : value(value) {}
};

class VarAST : public ExprAST {
public:
    std::shared_ptr<Variable> var;

    VarAST(std::shared_ptr<Variable> var) : var(var) {}
};
