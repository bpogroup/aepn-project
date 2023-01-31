import ply.lex as lex
import ply.yacc as yacc
from cpn_ast import PetriNetNode, PlaceNode, TransitionNode, ArcNode, MarkingNode, TokenNode


class PNLexer:
    reserved = {
        'Places': 'PLACES',
        'Transitions': 'TRANSITIONS',
        'Arcs': 'ARCS',
        'Markings': 'MARKINGS'
    }

    tokens = [
        'EXPRESSION',
        'LABEL',
        'NUMBER',
        'COMMA',
        'EQUALS',
        'LPAREN',
        'RPAREN',
        'OPENSET',
        'CLOSESET',
        'PLUSPLUS'
    ] + list(reserved.values())
    
    t_COMMA = r','
    t_EQUALS = r'='
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_OPENSET = r'\{'
    t_CLOSESET = r'\}'
    t_PLUSPLUS = r'\+\+'

    t_ignore  = ' \t'

    def t_EXPRESSION(self, t):
        r'`[^`]+`'
        t.value = t.value[1:len(t.value)-1]
        return t

    def t_TEXT(self, t):
        r'[;a-zA-Z][a-zA-Z0-9;]*' #DIRTY! the ; character is treated as text to allow markings of arbitrary size
        if t.value in self.reserved:  # Check for reserved words
            t.type = self.reserved[t.value]
        else:
            t.type = "LABEL"
        return t
    
    def t_NUMBER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t
     
    def t_newline(self, t):
         r'\n+'
         t.lexer.lineno += len(t.value)
             
    def t_error(self, t):
        raise Exception("Line %s, illegal character '%s'" % (t.lineno, t.value[0]))
        t.lexer.skip(1)

    def __init__(self):    
        self.lexer = lex.lex(module=self)


class PNParser:
    
    tokens = PNLexer.tokens
    
    def p_petrinet(self, p):
        '''petrinet : PLACES EQUALS OPENSET places CLOSESET\
                      TRANSITIONS EQUALS OPENSET transitions CLOSESET\
                      ARCS EQUALS OPENSET arcs CLOSESET\
                      MARKINGS EQUALS OPENSET markings CLOSESET'''
        p[0] = PetriNetNode(p[4], p[9], p[14], p[19])

    def p_places(self, p):
        '''places : place COMMA places
                  | place'''
        p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

    def p_transitions(self, p):
        '''transitions : transition COMMA transitions
                       | transition'''
        p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3] 

    def p_arcs(self, p):
        '''arcs : arc COMMA arcs
                | arc'''
        p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

    def p_markings(self, p):
        '''markings : marking COMMA markings
                    | marking'''
        p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

    def p_place(self, p):
        '''place : LABEL LPAREN RPAREN'''
        p[0] = PlaceNode(p[1], "")

    def p_transition(self, p):
        '''transition : LABEL LPAREN RPAREN
                      | LABEL LPAREN LABEL COMMA NUMBER RPAREN'''
        if len(p) == 4:
            p[0] = TransitionNode(p[1])
        elif len(p) == 7:
            p[0] = TransitionNode(p[1], p[3], p[5])
        else:
            raise Exception(f"Invalid transition token")

    def p_arc(self, p):
        '''arc : LPAREN LABEL COMMA LABEL RPAREN
               | LPAREN LABEL COMMA LABEL COMMA LABEL RPAREN
               | LPAREN LABEL COMMA LABEL COMMA LABEL COMMA PLUSPLUS NUMBER RPAREN
               | LPAREN LABEL COMMA LABEL COMMA LABEL COMMA NUMBER COMMA NUMBER RPAREN
               | LPAREN LABEL COMMA LABEL COMMA LABEL COMMA PLUSPLUS LABEL LPAREN NUMBER RPAREN RPAREN'''
        if len(p) == 6: #unlabeled arc
            p[0] = ArcNode(p[2], p[4])
        elif len(p) == 8: #labeled arc
            p[0] = ArcNode(p[2], p[4], p[6])
        elif len(p) == 11: #fixed delay labelled time increasing arc (outgoing from transition)
            p[0] = ArcNode(p[2], p[4], p[6], increment=p[9])
        elif len(p) == 12: #labelled time windowed arc (incoming in transition) NOT REALLY NEEDED
            p[0] = ArcNode(p[2], p[4], p[6], tw_low=p[8], tw_high=p[10])
        elif len(p) == 14: #variable delay labelled time increasing arc (outgoing from transition)
            p[0] = ArcNode(p[2], p[4], p[6], delay_type=p[9], params=p[11])
        else:
            raise Exception("Invalid arc token")

    def p_marking(self, p):
        '''marking : LPAREN LABEL COMMA tokens RPAREN'''
        p[0] = MarkingNode(p[2], p[4])

    def p_tokens(self, p):
        '''tokens : NUMBER EXPRESSION
                  | NUMBER EXPRESSION PLUSPLUS tokens'''
        token = TokenNode(p[1], p[2])
        p[0] = [token] if len(p) == 3 else [token] + p[4]

    def p_error(self, token):
        if token is not None:
            raise Exception("Line %s, illegal token %s" % (token.lineno, token.value))
        else:
            raise Exception('Unexpected end of input')

    def parse(self, text):
        return self.parser.parse(text)
    
    def __init__(self):
        self.lexer = PNLexer()
        self.parser = yacc.yacc(module=self)
