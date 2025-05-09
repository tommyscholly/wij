use std::iter::Peekable;

use crate::ast::Spanned;

type LexItem = char;

#[derive(Debug)]
pub enum LexError {
    UnexpectedChar(LexItem),
    UnexpectedKeyword(String),
    UnexpectedEOF,
}

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
}

macro_rules! advance_single_token {
    ($self:expr, $token_type:expr) => {{
        $self.chars.next();
        $self.current += 1;
        let token = ($token_type, $self.start..$self.current);
        $self.start += 1;
        return Ok(token);
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
                '=' => {
                    advance_single_token!(self, Token::Eq)
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
                '-' => {
                    self.chars.next();
                    if self.chars.peek() == Some(&'>') {
                        advance_single_token!(self, Token::Arrow)
                    } else {
                        return Err(LexError::UnexpectedChar('-'));
                    }
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
                        return Err(LexError::UnexpectedChar(*c));
                    }
                }
            }
        }

        Err(LexError::UnexpectedEOF)
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
            Err(_) => Ok((Token::Identifier(kw_var), self.start..self.current)),
        }
    }
}

impl<T: Iterator<Item = LexItem>> Iterator for Lexer<T> {
    type Item = Spanned<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.chars.peek().is_none() {
            None
        } else {
            let tok = self.next_token();
            match tok {
                Ok(token) => Some(token),
                Err(err) => panic!("Lexer error: {:?}", err),
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
        let tokens = lexer.collect::<Vec<Spanned<Token>>>();

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
        let tokens = lexer.collect::<Vec<Spanned<Token>>>();

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
        let tokens = lexer.collect::<Vec<Spanned<Token>>>();

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
        let tokens = lexer.collect::<Vec<Spanned<Token>>>();

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
        let tokens = lexer.collect::<Vec<Spanned<Token>>>();

        let expected = vec![
            (Token::Keyword(Keyword::Match), 0..5),
            (Token::Identifier("a".to_string()), 6..7),
            (Token::LBrace, 8..9),
            (Token::Int(1), 10..11),
            (Token::Arrow, 12..13),
            (Token::Int(2), 14..15),
            (Token::Comma, 15..16),
            (Token::Int(2), 17..18),
            (Token::Arrow, 19..20),
            (Token::Int(3), 21..22),
            (Token::RBrace, 23..24),
        ];
        assert_eq!(tokens.len(), expected.len());
        assert_eq!(tokens, expected);
    }
}
