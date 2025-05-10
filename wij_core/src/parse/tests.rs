use super::lex::*;
use crate::ast::{BinOp, Spanned};

#[test]
fn test_let() {
    let src = "let a: int = 1;";
    let lexer = tokenize(src);
    let tokens = lexer.map(Result::unwrap).collect::<Vec<Spanned<Token>>>();

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
    let tokens = lexer.map(Result::unwrap).collect::<Vec<Spanned<Token>>>();

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
    let tokens = lexer.map(Result::unwrap).collect::<Vec<Spanned<Token>>>();

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
    let tokens = lexer.map(Result::unwrap).collect::<Vec<Spanned<Token>>>();

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
    let tokens = lexer.map(Result::unwrap).collect::<Vec<Spanned<Token>>>();

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
    let tokens = lexer.map(Result::unwrap).collect::<Vec<Spanned<Token>>>();

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
    let tokens = lexer.map(Result::unwrap).collect::<Vec<Spanned<Token>>>();

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
    let tokens = lexer.map(Result::unwrap).collect::<Vec<Spanned<Token>>>();

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
