mod types;

use crate::parse::{Token, lex::Keyword};
use types::Type;

use std::collections::VecDeque;

pub type Span = std::ops::Range<usize>;
pub type Spanned<T> = (T, Span);

#[derive(Debug)]
pub enum ParseErrorKind {
    MalformedVariable,
    MalformedExpression,
    MalformedType,
    MalformedLet,
    ExpectedSemiColon,
    EndOfInput,
}

#[derive(Debug)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Option<Span>,
    pub reason: Option<String>,
}

impl ParseError {
    pub fn with_reason(kind: ParseErrorKind, span: Span, reason: &str) -> Self {
        ParseError {
            kind,
            span: Some(span),
            reason: Some(reason.to_string()),
        }
    }

    pub fn new(kind: ParseErrorKind, span: Span) -> Self {
        ParseError {
            kind,
            span: Some(span),
            reason: None,
        }
    }

    pub fn spanless(kind: ParseErrorKind) -> Self {
        ParseError {
            kind,
            span: None,
            reason: None,
        }
    }
}
pub type ParseResult<T> = Result<T, ParseError>;

pub trait Parseable
where
    Self: Sized,
{
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>>;
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Var {
    pub name: String,
    pub ty: Type,
}

impl Parseable for Var {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        let (name, span) = parser.expect_ident()?;

        let Some((Token::Colon, _)) = parser.pop_next() else {
            return Err(ParseError::with_reason(
                ParseErrorKind::MalformedVariable,
                span,
                "Expected colon",
            ));
        };
        let (ty, ty_span) = Type::parse(parser)?;
        let span = span.start..ty_span.end;
        Ok((Var { name, ty }, span))
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Statement {
    Let {
        var: Var,
        value: Option<Spanned<Expression>>,
    },
    Block(Vec<Spanned<Statement>>),
}

macro_rules! match_optional_token {
    ($parser:expr, $token_pattern:pat, $parse_fn:expr) => {
        match $parser.pop_next() {
            Some(($token_pattern, _)) => Some($parse_fn),
            Some(t) => {
                $parser.push_front(t);
                None
            }
            None => None,
        }
    };
}

impl Parseable for Statement {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        match parser.peek_next() {
            Some((Token::Keyword(Keyword::Let), _)) => {
                parser.pop_next();
                let (var, var_span) = Var::parse(parser)?;
                let value = match_optional_token!(parser, Token::Eq, Expression::parse(parser)?);
                let span = if let Some((_, span)) = &value {
                    var_span.start..span.end
                } else {
                    var_span
                };

                let Some((Token::SemiColon, _)) = parser.pop_next() else {
                    return Err(ParseError::new(ParseErrorKind::ExpectedSemiColon, span));
                };

                Ok((Statement::Let { var, value }, span))
            }
            Some((Token::LBrace, span)) => {
                parser.pop_next();

                let mut stmts = Vec::new();
                let mut last_span = span.clone();
                while parser.has_next() {
                    if let Some((Token::RBrace, _)) = parser.peek_next() {
                        parser.pop_next();
                        break;
                    }
                    let stmt = Statement::parse(parser)?;
                    last_span = stmt.1.clone();
                    stmts.push(stmt);
                }

                let block_span = span.start..last_span.end;
                Ok((Statement::Block(stmts), block_span))
            }
            Some(t) => {
                let span = t.1;
                Err(ParseError::with_reason(
                    ParseErrorKind::MalformedExpression,
                    span,
                    &format!("Expected statement, got {:?}", t.0),
                ))
            }

            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expression {
    Int(i32),
    String(String),
}

impl Parseable for Expression {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        match parser.pop_next() {
            Some((Token::Int(i), span)) => Ok((Expression::Int(i), span)),
            // Some((Token::String(s), span)) => Ok((Expression::String(s), span)),
            Some(t) => {
                let span = t.1;
                Err(ParseError::with_reason(
                    ParseErrorKind::MalformedExpression,
                    span,
                    "Expected expression",
                ))
            }
            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Declaration {
    Function {
        name: String,
        arguments: Vec<Var>,
        body: Statement,
        ret_type: Option<Type>,
    },
}

fn parse_fn_args(parser: &mut Parser) -> ParseResult<Vec<Var>> {
    let mut vars = Vec::new();
    while parser.peek_next().is_some() {
        if let Some((Token::RParen, _)) = parser.peek_next() {
            parser.pop_next();
            break;
        }
        vars.push(Var::parse(parser)?.0);
        if let Some((Token::Comma, _)) = parser.peek_next() {
            parser.pop_next();
        } else if let Some((Token::RParen, _)) = parser.peek_next() {
            parser.pop_next();
            break;
        }
    }

    Ok(vars)
}

impl Parseable for Declaration {
    fn parse(parser: &mut Parser) -> ParseResult<Spanned<Self>> {
        let start_span = parser.expect_kw_kind(Keyword::Fn)?;
        let (name, _) = parser.expect_ident()?;
        let _lparen = parser.expect_next(Token::LParen)?;
        let arguments = parse_fn_args(parser)?;
        let ret_type = if let Some((Token::Arrow, _)) = parser.peek_next() {
            Some(Type::parse(parser)?.0)
        } else {
            None
        };
        let (body, body_span) = Statement::parse(parser)?;
        let span = start_span.start..body_span.end;
        Ok((
            Declaration::Function {
                name,
                arguments,
                body,
                ret_type,
            },
            span,
        ))
    }
}

pub struct Parser {
    tokens: VecDeque<Spanned<Token>>,
}

impl Parser {
    pub fn new(tokens: VecDeque<Spanned<Token>>) -> Self {
        Self { tokens }
    }

    fn has_next(&self) -> bool {
        !self.tokens.is_empty()
    }

    fn expect_next(&mut self, token: Token) -> ParseResult<Spanned<Token>> {
        match self.tokens.pop_front() {
            Some(t) if t.0 == token => Ok(t),
            Some(t) => Err(ParseError::with_reason(
                ParseErrorKind::MalformedExpression,
                t.1,
                &format!("Expected `{:?}`, got `{:?}`", token, t.0),
            )),
            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }

    fn expect_kw(&mut self) -> ParseResult<Spanned<Keyword>> {
        match self.tokens.pop_front() {
            Some((Token::Keyword(kw), span)) => Ok((kw, span)),
            Some(t) => Err(ParseError::with_reason(
                ParseErrorKind::MalformedExpression,
                t.1,
                &format!("Expected keyword, got `{:?}`", t.0),
            )),
            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }

    fn expect_kw_kind(&mut self, kw: Keyword) -> ParseResult<Span> {
        match self.expect_kw() {
            Ok((kw2, span)) if kw == kw2 => Ok(span),
            Ok((kw2, span)) => Err(ParseError::with_reason(
                ParseErrorKind::MalformedExpression,
                span,
                &format!("Expected `{:?}`, got `{:?}`", kw, kw2),
            )),
            Err(err) => Err(err),
        }
    }

    fn expect_ident(&mut self) -> ParseResult<(String, Span)> {
        match self.tokens.pop_front() {
            Some((Token::Identifier(ident), span)) => Ok((ident, span)),
            Some(t) => Err(ParseError::with_reason(
                ParseErrorKind::MalformedExpression,
                t.1,
                &format!("Expected identifier, got `{:?}`", t.0),
            )),
            None => Err(ParseError::spanless(ParseErrorKind::EndOfInput)),
        }
    }

    fn take_until(&mut self, until_tok: Token) -> ParseResult<Vec<Spanned<Token>>> {
        let mut tokens = Vec::new();
        while let Some(t) = self.pop_next() {
            if t.0 == until_tok {
                break;
            }
            tokens.push(t);
        }
        Ok(tokens)
    }

    fn pop_next(&mut self) -> Option<Spanned<Token>> {
        self.tokens.pop_front()
    }

    fn push_front(&mut self, token: Spanned<Token>) {
        self.tokens.push_front(token);
    }

    fn peek_next(&self) -> Option<Spanned<Token>> {
        self.tokens.front().cloned()
    }
}

impl Iterator for Parser {
    type Item = Spanned<Declaration>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.tokens.is_empty() {
            None
        } else {
            match Declaration::parse(self) {
                Ok(statement) => Some(statement),
                Err(err) => panic!("Parser error: {:?}", err),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_parser() {
    //     let src = "let a: int = 1;";
    //     let lexer = crate::parse::lex::tokenize(src);
    //     let toks = lexer.collect();
    //     println!("Toks: {:?}", toks);
    //     let parser = Parser::new(toks);
    //
    //     let statements: Vec<Spanned<Statement>> = parser.into_iter().collect();
    //     assert_eq!(statements.len(), 1);
    // }

    #[test]
    fn test_parse_fn() {
        let src = "fn main() {}";
        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        let parser = Parser::new(toks);

        let decls = parser.collect::<Vec<Spanned<Declaration>>>();
        let expected = vec![(
            Declaration::Function {
                name: "main".to_string(),
                arguments: vec![],
                body: Statement::Block(vec![]),
                ret_type: None,
            },
            0..11,
        )];
        assert_eq!(decls.len(), 1);
        assert_eq!(decls, expected);
    }

    #[test]
    fn test_parse_fn_with_params_and_body() {
        let src = "fn main(a: int, b: int) {
            let c: int = 1;
        }";
        let lexer = crate::parse::lex::tokenize(src);
        let toks = lexer.collect();
        let parser = Parser::new(toks);

        let decls = parser.collect::<Vec<Spanned<Declaration>>>();
        let expected = vec![(
            Declaration::Function {
                name: "main".to_string(),
                arguments: vec![
                    Var {
                        name: "a".to_string(),
                        ty: Type::Int,
                    },
                    Var {
                        name: "b".to_string(),
                        ty: Type::Int,
                    },
                ],
                body: Statement::Block(vec![(
                    Statement::Let {
                        var: Var {
                            name: "c".to_string(),
                            ty: Type::Int,
                        },
                        value: Some((Expression::Int(1), 51..52)),
                    },
                    42..52,
                )]),
                ret_type: None,
            },
            0..52,
        )];
        assert_eq!(decls.len(), 1);
        assert_eq!(decls, expected);
    }
}
