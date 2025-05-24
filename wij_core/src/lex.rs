mod macros;

use std::iter::Peekable;

use crate::Span;
use crate::WijError;
use crate::ast::BinOp;
use crate::ast::Spanned;
use crate::{advance_single_token, handle_operator};

type LexItem = char;

#[derive(Debug)]
pub enum LexErrorKind {
    UnexpectedChar(LexItem),
    UnexpectedKeyword(String),
    UnexpectedEOF,
}

#[derive(Debug)]
pub struct LexError {
    kind: LexErrorKind,
    span: Span,
}

impl LexError {
    pub fn new(kind: LexErrorKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl WijError for LexError {
    fn span(&self) -> Option<Span> {
        Some(self.span.clone())
    }

    fn reason(&self) -> String {
        match &self.kind {
            LexErrorKind::UnexpectedChar(c) => format!("Unexpected character: {}", c),
            LexErrorKind::UnexpectedKeyword(kw) => format!("Unexpected keyword: {}", kw),
            LexErrorKind::UnexpectedEOF => "Unexpected end of input".to_string(),
        }
    }

    fn notes(&self) -> Vec<(String, crate::Span)> {
        Vec::new()
    }
}

pub type LexResult<T> = Result<T, LexError>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Keyword {
    Let,
    Fn,
    Type,
    If,
    Else,
    Match,
    For,
    While,
    In,
    True,
    False,
    Foreign,
    Use,
    Pub,
    Module,

    Return,
    Break,
    Continue,

    Int,
    Str,
    Bool,
    Byte,

    Procs,
    // Self is a kw
    Self_,
}

impl TryFrom<&str> for Keyword {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "let" => Ok(Keyword::Let),
            "int" => Ok(Keyword::Int),
            "fn" => Ok(Keyword::Fn),
            "type" => Ok(Keyword::Type),
            "if" => Ok(Keyword::If),
            "else" => Ok(Keyword::Else),
            "match" => Ok(Keyword::Match),
            "for" => Ok(Keyword::For),
            "in" => Ok(Keyword::In),
            "true" => Ok(Keyword::True),
            "false" => Ok(Keyword::False),
            "bool" => Ok(Keyword::Bool),
            "foreign" => Ok(Keyword::Foreign),
            "use" => Ok(Keyword::Use),
            "pub" => Ok(Keyword::Pub),
            "module" => Ok(Keyword::Module),
            "str" => Ok(Keyword::Str),
            "byte" => Ok(Keyword::Byte),
            "return" => Ok(Keyword::Return),
            "break" => Ok(Keyword::Break),
            "continue" => Ok(Keyword::Continue),
            "procs" => Ok(Keyword::Procs),
            "self" => Ok(Keyword::Self_),
            _ => Err(()),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Token {
    Int(i32),
    Keyword(Keyword),
    Identifier(String),
    String(String),
    BinOp(BinOp),
    Eq,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    SemiColon,
    Colon,
    Arrow,
    Bar,
    Dot,
    Tick,
}

impl TryFrom<&str> for BinOp {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "+" => Ok(BinOp::Add),
            "-" => Ok(BinOp::Sub),
            "*" => Ok(BinOp::Mul),
            "/" => Ok(BinOp::Div),
            "%" => Ok(BinOp::Mod),
            "==" => Ok(BinOp::EqEq),
            "!=" => Ok(BinOp::NEq),
            ">" => Ok(BinOp::Gt),
            ">=" => Ok(BinOp::GtEq),
            "<" => Ok(BinOp::Lt),
            "<=" => Ok(BinOp::LtEq),
            "and" => Ok(BinOp::And),
            "or" => Ok(BinOp::Or),
            _ => Err(()),
        }
    }
}

pub struct Lexer<T: Iterator<Item = LexItem>> {
    chars: Peekable<T>,
    start: usize,
    current: usize,
}

impl<T: Iterator<Item = LexItem>> Lexer<T> {
    pub fn new(chars: T) -> Self {
        Lexer {
            chars: chars.peekable(),
            start: 0,
            current: 0,
        }
    }

    fn next_token(&mut self) -> Result<Spanned<Token>, LexError> {
        while let Some(c) = self.chars.peek() {
            match c {
                '/' => {
                    if self.chars.peek() == Some(&'/') {
                        while let Some(&c) = self.chars.peek() {
                            if c == '\n' {
                                self.chars.next();
                                self.current += 1;
                                self.start = self.current;
                                break;
                            } else {
                                self.chars.next();
                                self.current += 1;
                            }
                        }
                    } else {
                        advance_single_token!(self, Token::BinOp(BinOp::Div))
                    }
                }
                '\'' => {
                    advance_single_token!(self, Token::Tick)
                }
                ';' => {
                    advance_single_token!(self, Token::SemiColon)
                }
                ':' => {
                    advance_single_token!(self, Token::Colon)
                }
                '(' => {
                    advance_single_token!(self, Token::LParen)
                }
                ')' => {
                    advance_single_token!(self, Token::RParen)
                }
                '{' => {
                    advance_single_token!(self, Token::LBrace)
                }
                '}' => {
                    advance_single_token!(self, Token::RBrace)
                }
                '[' => {
                    advance_single_token!(self, Token::LBracket)
                }
                ']' => {
                    advance_single_token!(self, Token::RBracket)
                }
                ',' => {
                    advance_single_token!(self, Token::Comma)
                }
                '|' => {
                    advance_single_token!(self, Token::Bar)
                }
                '=' => {
                    handle_operator!(self, '=', '=', Token::Eq, Token::BinOp(BinOp::EqEq))
                }
                '.' => {
                    advance_single_token!(self, Token::Dot)
                }
                '>' => {
                    handle_operator!(
                        self,
                        '>',
                        '=',
                        Token::BinOp(BinOp::Gt),
                        Token::BinOp(BinOp::GtEq)
                    )
                }
                '<' => {
                    handle_operator!(
                        self,
                        '<',
                        '=',
                        Token::BinOp(BinOp::Lt),
                        Token::BinOp(BinOp::LtEq)
                    )
                }
                '-' => {
                    handle_operator!(self, '-', '>', Token::BinOp(BinOp::Sub), Token::Arrow)
                }
                '"' => {
                    self.chars.next();
                    self.current += 1;
                    let mut string = String::new();
                    loop {
                        match self.chars.peek() {
                            Some('"') => {
                                self.chars.next();
                                self.current += 1;
                                break;
                            }
                            Some(c) => {
                                string.push(*c);
                                self.chars.next();
                                self.current += 1;
                            }
                            None => {
                                return Err(LexError::new(
                                    LexErrorKind::UnexpectedEOF,
                                    self.current..self.current,
                                ));
                            }
                        }
                    }

                    let span = self.start..self.current;
                    self.start = self.current;
                    return Ok((Token::String(string), span));
                }
                ' ' | '\t' | '\n' | '\r' => {
                    self.chars.next();
                    self.start += 1;
                    self.current += 1;
                }
                c => {
                    if c.is_ascii_digit() {
                        let token = self.next_number();
                        self.start = self.current;
                        return token;
                    } else if c.is_alphabetic() || *c == '_' {
                        let token = self.next_kw_var();
                        self.start = self.current;
                        return token;
                    } else {
                        if let Ok(op) = BinOp::try_from(c.to_string().as_str()) {
                            advance_single_token!(self, Token::BinOp(op))
                        }

                        return Err(LexError::new(
                            LexErrorKind::UnexpectedChar(*c),
                            self.start..self.current,
                        ));
                    }
                }
            }
        }

        Err(LexError::new(
            LexErrorKind::UnexpectedEOF,
            self.current..self.current,
        ))
    }

    fn next_number(&mut self) -> Result<Spanned<Token>, LexError> {
        let mut number = String::new();
        while let Some(c) = self.chars.peek() {
            if c.is_ascii_digit() {
                number.push(*c);
                self.chars.next();
                self.current += 1;
            } else {
                break;
            }
        }

        Ok((
            Token::Int(number.parse().unwrap()),
            self.start..self.current,
        ))
    }

    fn next_kw_var(&mut self) -> Result<Spanned<Token>, LexError> {
        let mut kw_var = String::new();
        while let Some(c) = self.chars.peek() {
            if c.is_alphanumeric() || *c == '_' {
                kw_var.push(*c);
                self.chars.next();
                self.current += 1;
            } else {
                break;
            }
        }

        match Keyword::try_from(kw_var.as_str()) {
            Ok(kw) => Ok((Token::Keyword(kw), self.start..self.current)),
            Err(_) => match BinOp::try_from(kw_var.as_str()) {
                Ok(op) => Ok((Token::BinOp(op), self.start..self.current)),
                Err(_) => Ok((Token::Identifier(kw_var), self.start..self.current)),
            },
        }
    }
}

impl<T: Iterator<Item = LexItem>> Iterator for Lexer<T> {
    type Item = LexResult<Spanned<Token>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.chars.peek().is_none() {
            None
        } else {
            let tok = self.next_token();
            match tok {
                Ok(token) => Some(Ok(token)),
                Err(err) => Some(Err(err)),
            }
        }
    }
}

pub fn tokenize(src: &str) -> Lexer<impl Iterator<Item = LexItem>> {
    let chars = src.chars();
    Lexer::new(chars)
}

#[cfg(test)]
mod tests;
