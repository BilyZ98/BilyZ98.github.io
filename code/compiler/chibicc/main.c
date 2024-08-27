
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

static void error(char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap); 
  fprintf(stderr, "\n");
  exit(1);
};

static bool equal(Token* tok, char* op) {
  return memcmp(tok->loc, op, tok->len) == 0 && op[tok->len] == '\0';
}


static Token* skip(Token* tok, char*s) {
  if(!equal(tok, s)) {
    error("expected '%s', cur tok is '%s', len %d", s, tok->loc, tok->len);
  }
  return tok->next;
}

static int get_number(Token* tok) {
  if(tok->kind != TK_NUM) {
    error("expected a number");
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


static Token* tokenize(char* p) {
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

    if(*p == '+' || *p == '-') {
      cur = cur->next = new_token(TK_PUNCT, p, p+1);
      p++;
      continue;;
    }

    error("invalid token");


  }
  cur = cur->next = new_token(TK_EOF, p, p);
  return head.next;
}



int main(int argc, char** argv) {
  if (argc !=2) {
    error("%s: invalid number of arguments", argv[0]);
  }
  Token* tok = tokenize(argv[1]);

  printf("  .globl main\n");
  printf("main: \n");
  printf("  mov $%d, %%rax\n", get_number(tok));
  tok = tok->next;

  while(tok->kind != TK_EOF) {
    if(equal(tok, "+")) {
      printf("  add $%d, %%rax\n", get_number(tok->next));
      tok = tok->next->next;
      continue;
    }

    tok = skip(tok, "-");
    printf("  sub $%d, %%rax\n", get_number(tok));
    tok = tok->next;
  }

  printf("  ret\n");
  return 0;
} 

