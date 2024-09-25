grammar VCalc;

file: .*? EOF;

// Skip whitespace
WS : [ \t\r\n]+ -> skip ;
