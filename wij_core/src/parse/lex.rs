use std::iter::Peekable;

use crate::AstError;
use crate::ast::BinOp;
use crate::ast::Span;
use crate::ast::Spanned;

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

impl AstError for LexError {
    fn span(&self) -> Option<Span> {
        Some(self.span.clone())
    }
    fn reason(&self) -> &str {
        match &self.kind {
            LexErrorKind::UnexpectedChar(_) => "Unexpected character",
            LexErrorKind::UnexpectedKeyword(_) => "Unexpected keyword",
            LexErrorKind::UnexpectedEOF => "Unexpected end of input",
        }
    }
}

pub type LexResult<T> = Result<T, LexError>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Keyword {
    Let,
    Int,
    Fn,
    Type,
    If,
    Else,
    Match,
    For,
    In,
    Return,
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
            "return" => Ok(Keyword::Return),
            _ => Err(()),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Token {
    Int(i32),
    Keyword(Keyword),
    Identifier(String),
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
}

impl TryFrom<&str> for BinOp {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "+" => Ok(BinOp::Add),
            "-" => Ok(BinOp::Sub),
            "*" => Ok(BinOp::Mul),
            "/" => Ok(BinOp::Div),
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

macro_rules! advance_single_token {
    ($self:expr, $token_type:expr) => {{
        $self.chars.next();
        $self.current += 1;
        let token = ($token_type, $self.start..$self.current);
        $self.start = $self.current;
        return Ok(token);
    }};
}

macro_rules! handle_operator {
    ($self:expr, $first_char:expr, $second_char:expr, 
     $single_token:expr, $double_token:expr) => {{
        $self.chars.next(); 
        $self.current += 1;
        
        if $self.chars.peek() == Some(&$second_char) {
            $self.chars.next(); 
            $self.current += 1;
            
            let span = $self.start..$self.current;
            $self.start = $self.current;
            return Ok(($double_token, span));
        } else {
            let span = $self.start..$self.current;
            $self.start = $self.current;
            return Ok(($single_token, span));
        }
    }};
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
                '>' => {
                    handle_operator!(self, '>', '=', Token::BinOp(BinOp::Gt), Token::BinOp(BinOp::GtEq))
                }
                '<' => {
                    handle_operator!(self, '<', '=', Token::BinOp(BinOp::Lt), Token::BinOp(BinOp::LtEq))
                }
                '-' => {
                    handle_operator!(self, '-', '>', Token::BinOp(BinOp::Sub), Token::Arrow)
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
                    } else if c.is_alphabetic() {
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
            if c.is_alphanumeric() {
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
mod tests {
    use super::*;

    #[test]
    fn test_let() {
        let src = "let a: int = 1;";
        let lexer = tokenize(src);
        let tokens = lexer.map(|t| t.unwrap()).collect::<Vec<Spanned<Token>>>();

        let expected = vec![
            (Token::Keyword(Keyword::Let), 0..3),
            (Token::Identifier("a".to_string()), 4..5),
            (Token::Colon, 5..6),
            (Token::Keyword(Keyword::Int), 7..10),
            (Token::Eq, 11..12),
            (Token::Int(1), 13..14),
            (Token::SemiColon, 14..15),
        ];
        assert_eq!(tokens.len(), expected.len());
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_fn() {
        let src = "fn main() {}";
        let lexer = tokenize(src);
        let tokens = lexer.map(|t| t.unwrap()).collect::<Vec<Spanned<Token>>>();

        let expected = vec![
            (Token::Keyword(Keyword::Fn), 0..2),
            (Token::Identifier("main".to_string()), 3..7),
            (Token::LParen, 7..8),
            (Token::RParen, 8..9),
            (Token::LBrace, 10..11),
            (Token::RBrace, 11..12),
        ];
        assert_eq!(tokens.len(), expected.len());
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_fn_with_params() {
        let src = "fn main(a: int, b: int) {}";
        let lexer = tokenize(src);
        let tokens = lexer.map(|t| t.unwrap()).collect::<Vec<Spanned<Token>>>();

        let expected = vec![
            (Token::Keyword(Keyword::Fn), 0..2),
            (Token::Identifier("main".to_string()), 3..7),
            (Token::LParen, 7..8),
            (Token::Identifier("a".to_string()), 8..9),
            (Token::Colon, 9..10),
            (Token::Keyword(Keyword::Int), 11..14),
            (Token::Comma, 14..15),
            (Token::Identifier("b".to_string()), 16..17),
            (Token::Colon, 17..18),
            (Token::Keyword(Keyword::Int), 19..22),
            (Token::RParen, 22..23),
            (Token::LBrace, 24..25),
            (Token::RBrace, 25..26),
        ];
        assert_eq!(tokens.len(), expected.len());
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_for() {
        let src = "for i in a {}";
        let lexer = tokenize(src);
        let tokens = lexer.map(|t| t.unwrap()).collect::<Vec<Spanned<Token>>>();

        let expected = vec![
            (Token::Keyword(Keyword::For), 0..3),
            (Token::Identifier("i".to_string()), 4..5),
            (Token::Keyword(Keyword::In), 6..8),
            (Token::Identifier("a".to_string()), 9..10),
            (Token::LBrace, 11..12),
            (Token::RBrace, 12..13),
        ];
        assert_eq!(tokens.len(), expected.len());
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_match() {
        let src = "match a { 1 -> 2, 2 -> 3 }";
        let lexer = tokenize(src);
        let tokens = lexer.map(|t| t.unwrap()).collect::<Vec<Spanned<Token>>>();

        let expected = vec![
            (Token::Keyword(Keyword::Match), 0..5),
            (Token::Identifier("a".to_string()), 6..7),
            (Token::LBrace, 8..9),
            (Token::Int(1), 10..11),
            (Token::Arrow, 12..14),
            (Token::Int(2), 15..16),
            (Token::Comma, 16..17),
            (Token::Int(2), 18..19),
            (Token::Arrow, 20..22),
            (Token::Int(3), 23..24),
            (Token::RBrace, 25..26),
        ];
        assert_eq!(tokens.len(), expected.len());
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_type_enum() {
        let src = "type Color = Red | Green | Blue";
        let lexer = tokenize(src);
        let tokens = lexer.map(|t| t.unwrap()).collect::<Vec<Spanned<Token>>>();

        let expected = vec![
            (Token::Keyword(Keyword::Type), 0..4),
            (Token::Identifier("Color".to_string()), 5..10),
            (Token::Eq, 11..12),
            (Token::Identifier("Red".to_string()), 13..16),
            (Token::Bar, 17..18),
            (Token::Identifier("Green".to_string()), 19..24),
            (Token::Bar, 25..26),
            (Token::Identifier("Blue".to_string()), 27..31),
        ];
        assert_eq!(tokens.len(), expected.len());
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_type_struct() {
        let src = "type Point = { x: int, y: int }";
        let lexer = tokenize(src);
        let tokens = lexer.map(|t| t.unwrap()).collect::<Vec<Spanned<Token>>>();

        let expected = vec![
            (Token::Keyword(Keyword::Type), 0..4),
            (Token::Identifier("Point".to_string()), 5..10),
            (Token::Eq, 11..12),
            (Token::LBrace, 13..14),
            (Token::Identifier("x".to_string()), 15..16),
            (Token::Colon, 16..17),
            (Token::Keyword(Keyword::Int), 18..21),
            (Token::Comma, 21..22),
            (Token::Identifier("y".to_string()), 23..24),
            (Token::Colon, 24..25),
            (Token::Keyword(Keyword::Int), 26..29),
            (Token::RBrace, 30..31),
        ];
        assert_eq!(tokens.len(), expected.len());
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_if() {
        let src = "if a or c { b } else { d }";
        let lexer = tokenize(src);
        let tokens = lexer.map(|t| t.unwrap()).collect::<Vec<Spanned<Token>>>();

        let expected = vec![
            (Token::Keyword(Keyword::If), 0..2),
            (Token::Identifier("a".to_string()), 3..4),
            (Token::BinOp(BinOp::Or), 5..7),
            (Token::Identifier("c".to_string()), 8..9),
            (Token::LBrace, 10..11),
            (Token::Identifier("b".to_string()), 12..13),
            (Token::RBrace, 14..15),
            (Token::Keyword(Keyword::Else), 16..20),
            (Token::LBrace, 21..22),
            (Token::Identifier("d".to_string()), 23..24),
            (Token::RBrace, 25..26),
        ];
        assert_eq!(tokens.len(), expected.len());
        assert_eq!(tokens, expected);
    }
}
