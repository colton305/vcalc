grammar VCalc;



file
    : stat*? EOF ;

stat
    : assign
    | cond
    | decl
    | loop 
    | print ;

cond
    : IF '(' expr ')' stat* FI ';' ;

loop
    : LOOP '(' expr ')' stat* POOL ';' ;

decl
    : type=(INT | VECTOR) ID '=' expr ';' ;

assign
    : ID '=' expr ';' ;

print
    : PRINT '(' expr ')' ';' ;

expr
    : '[' ID IN expr '|' expr ']'  #gen
    | '[' ID IN expr '|' expr ']'  #filter
    | '(' expr ')'  #paren
    | expr '[' expr ']'  #index
    | expr '..' expr  #range
    | expr op=('*' | '/') expr  #mulDiv
    | expr op=('+' | '-') expr  #addSub
    | expr op=('<' | '>') expr  #strictComp
    | expr op=('==' | '!=') expr  #eqComp
    | NUM  #numAtom
    | ID  #idAtom ;


RANGE: '..' ;
IN: 'in' ;
IF: 'if' ;
FI: 'fi' ;
LOOP: 'loop' ;
POOL: 'pool';
INT: 'int' ;
VECTOR: 'vector';
PRINT: 'print' ;
NUM: [0-9]+ ;
ID: [a-zA-Z][a-zA-Z0-9]* ;
COMMENT: '//' ~[\r\n]* -> skip ;
// Skip whitespace
WS : [ \t\r\n]+ -> skip ;