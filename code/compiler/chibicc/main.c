#include <assert.h>
#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef enum {
  TK_PUNCT, // punctuators
  TK_NUM, 
  TK_EOF,

} TokenKind;



typedef struct Token Token;
struct Token {
  TokenKind kind;
  Token* next;
  int val; // if TokenKind is TK_NUM, its value
  char* loc;
  int len;
};

static char *current_input;
static void error(char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap); 
  fprintf(stderr, "\n");
  exit(1);
};

// reports an error location and exit
static void verror_at(char *loc, char* fmt, va_list ap) {
  int pos = loc - current_input;
  fprintf(stderr, "%s\n",current_input);
  fprintf(stderr, "%*s", pos, ""); //print pos spaces
  fprintf(stderr, "^ ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  exit(1);
}  


static void error_at(char* loc, char* fmt, ...){
  va_list ap;
  va_start(ap, fmt);
  verror_at(loc, fmt, ap);
}

static void error_tok(Token*tok, char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  verror_at(tok->loc, fmt, ap);
}


static bool equal(Token* tok, char* op) {
  return memcmp(tok->loc, op, tok->len) == 0 && op[tok->len] == '\0';
}

static bool startwith(char* p, char* q) {
  return strncmp(p, q, strlen(q)) == 0;
}

static int read_punct(char* p){
  if(startwith(p, "==") || startwith(p, "!=")
    || startwith(p, "<=") || startwith(p, ">=")) {
    return 2;
  }
  return ispunct(*p) ? 1 : 0;
}

static Token* skip(Token* tok, char*s) {
  if(!equal(tok, s)) {
    error_tok(tok, "expected '%s'", s);
  }
  return tok->next;
}

static int get_number(Token* tok) {
  if(tok->kind != TK_NUM) {
    error_tok(tok, "expected a number");
  }

  return tok->val;
}

static Token* new_token(TokenKind kind, char* start, char* end) {
  Token* tok = calloc(1, sizeof(Token));
  tok->kind = kind;
  tok->loc = start;
  tok->len = end - start;
  return tok;
}


static Token* tokenize(void) {
  char* p = current_input;
  Token head = {};
  Token* cur = &head;
  while(*p) {
    if(isspace(*p)) {
      p++;
      continue;
    }

    if(isdigit(*p)) {
      cur = cur->next = new_token(TK_NUM, p, p);
      char* q = p;
      cur->val = strtoul(p, &p, 10);
      cur->len = p - q;
      continue;
    }

    int punct_len = read_punct(p);
    if(punct_len) {
      cur->next = new_token(TK_PUNCT, p, p+punct_len);
      cur = cur->next;
      p+=punct_len;
      continue;;
    }
    error("invalid token");


  }
  cur = cur->next = new_token(TK_EOF, p, p);
  return head.next;
}

typedef enum {
  ND_ADD, // +
  ND_SUB, // - 
  ND_MUL, // *
  ND_DIV, // /
  ND_NEG, //  unary -
  ND_EQ, // ==
  ND_NE, // !=
  ND_LT, // <
  ND_LE, // <=
  ND_NUM, // Integer
  //
} NodeKind;

typedef struct Node Node;
struct Node {
  NodeKind kind;
  Node* lhs; //left hand side
  Node* rhs; // right hand side
  int val;  // used if kind == ND_NUM
};


static Node* new_node(NodeKind kind) {
  Node* node = calloc(1, sizeof(Node));
  node->kind = kind;
  return node;
}

static Node *new_binary(NodeKind kind, Node* lhs, Node* rhs) {
  Node* node = new_node(kind);
  node->lhs = lhs;
  node->rhs = rhs;
  return node;
}

static Node* new_unary(NodeKind kind, Node*expr) {
  Node* node = new_node(kind);
  node->kind = kind;
  node->lhs = expr;
  return node;
}

static Node* new_num(int val) {
  Node* node = new_node(ND_NUM);
  node->val = val;
  return node;
}

static Node* expr(Token** rest, Token* tok) ;
static Node* mul(Token** rest, Token* tok);
static Node* unary(Token** rest, Token* tok);
static Node* equality(Token** rest, Token* tok);
static Node* relational(Token**rest, Token* tok);
static Node* add(Token** rest, Token*tok);
static Node* primary(Token **rest, Token* tok);

// expr = mul ("+" mul | "-" mul)*
static Node* expr(Token **rest, Token* tok){
  return equality(rest, tok);
}

// bottom up
// equality = relational ("==" relational | "!= relational")*
static Node* equality(Token **rest, Token*tok) {
  Node* node = relational(&tok, tok);
  for(;;) {
    if(equal(tok, "==")) {
      node = new_binary(ND_EQ, node, relational(&tok, tok->next));
      continue;
    }

    if(equal(tok, "!=")) {
      node = new_binary(ND_NE, node, relational(&tok, tok->next));
      continue;;
    }
    *rest = tok;
    return node;
  }

}

// relational = add ("<" add | "<=" add | ">" add | ">=" add)*
static Node* relational(Token**rest, Token* tok) {
  Node* node = add(&tok, tok);
  for(;;) {
    if(equal(tok, "<")) {
      node = new_binary(ND_LT, node, add(&tok, tok->next));
      continue;;
    }

    if(equal(tok, "<=")) {
      node = new_binary(ND_LE, node, add(&tok, tok->next));
      continue;;
    }

    if(equal(tok, ">")) {
      node = new_binary(ND_LT, add(&tok, tok->next), node);
      continue;;
    }

    if(equal(tok, ">=")) {
      node = new_binary(ND_LE, add(&tok, tok->next), node);
      continue;;
    }
    *rest = tok;
    return node;

  }
}

// add = mul ("+" mul | "-" mul)*
static Node* add(Token**rest, Token* tok) {
  Node* node= mul(&tok, tok);

  for(;;){
    if(equal(tok, "+") ) {
      node = new_binary(ND_ADD, node, mul(&tok, tok->next));
      continue;
    }

    if(equal(tok, "-")) {
      node = new_binary(ND_SUB, node, mul(&tok, tok->next));
      continue;
    }

    *rest = tok;
    return node;

  }
}


// expr = unary ("*" unary | "/" unary)*
static Node* mul(Token**rest, Token* tok) {
  Node* node = unary(&tok, tok);

  for(;;) {
    if(equal(tok, "*")) {
      node =  new_binary(ND_MUL, node, unary(&tok, tok->next));
      continue;;
    }

    if(equal(tok, "/")) {
      node = new_binary(ND_DIV, node, unary(&tok, tok->next));
      continue;;
    }

    *rest = tok;
    return node;
  }
}


// unary = ("+" | "-") unary | primary
static Node* unary(Token **rest, Token* tok) {
  if(equal(tok, "+"))
    return unary(rest, tok->next);

  if(equal(tok, "-"))
    return new_unary(ND_NEG, unary(rest, tok->next));

  return primary(rest, tok);

}


// primary = "(" expr ")" | num
static Node* primary(Token **rest, Token* tok) {
  if(equal(tok, "(")) {
    Node* node = expr(&tok, tok->next);
    *rest = skip(tok, ")");
    return node;
  }

  if(tok->kind  == TK_NUM) {
    Node* node = new_num(tok->val);
    *rest = tok->next;
    return node;
  }

  error_tok(tok, "expected an expression");

}

//
// Code generator
//
//
static int depth;

static void push(void) {
  printf(" push %%rax\n");
  depth++;
}

static void pop(char* arg) {
  printf(" pop %s\n", arg);
  depth--;
}


static void gen_expr(Node* node) {
  switch(node->kind) {
    case ND_NUM:
    printf(" mov $%d, %%rax\n", node->val);
    return;

    case ND_NEG:
    gen_expr(node->lhs);
    printf(" neg %%rax\n");
    return;
  }

  gen_expr(node->rhs);
  push();
  gen_expr(node->lhs);
  pop("%rdi");

  switch(node->kind) {
    case ND_ADD:
    printf(" add %%rdi, %%rax\n");
    return;
    
    case ND_SUB:
    printf(" sub %%rdi, %%rax\n");
    return;

    case ND_MUL:
    printf(" imul %%rdi, %%rax\n");
    return;

    case ND_DIV:
    printf(" cqo\n");
    printf(" idiv %%rdi\n");
    return;

    case ND_EQ:
    case ND_NE:
    case ND_LT:
    case ND_LE:
    printf(" cmp %%rdi, %%rax\n");

    if(node->kind == ND_EQ) {
      printf(" sete %%al\n");
    }else if(node->kind == ND_NE) {
      printf(" setne %%al\n");
    } else if(node->kind == ND_LT) {
      printf(" setl %%al\n");
    } else if(node->kind == ND_LE) {
      printf(" setle %%al\n");
    }

    printf(" movzb %%al, %%rax\n");
    return;
  }
  error("invalid expression");
}

int main(int argc, char** argv) {
  if (argc !=2) {
    error("%s: invalid number of arguments", argv[0]);
  }
  current_input = argv[1];
  Token* tok = tokenize();
  Node* node = expr(&tok, tok);

  if(tok->kind != TK_EOF) {
    error_tok(tok, "extra token");
  }


  printf("  .globl main\n");
  printf("main: \n");

  // Traverse the AST to emit assembly
  gen_expr(node);

  printf("  ret\n");

  assert(depth == 0);
  return 0;
} 


