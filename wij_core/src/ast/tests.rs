use super::Literal::*;
use super::*;
use crate::tokenize;

#[test]
fn test_parse_fn() {
    let src = "fn main() {}";
    let lexer = tokenize(src);
    let toks = lexer.map(Result::unwrap).collect();
    let parser = Parser::new(toks);

    let decls = parser
        .map(Result::unwrap)
        .collect::<Vec<Spanned<Declaration>>>();
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
    let lexer = tokenize(src);
    let toks = lexer.map(Result::unwrap).collect();
    let parser = Parser::new(toks);

    let decls = parser
        .map(Result::unwrap)
        .collect::<Vec<Spanned<Declaration>>>();
    let expected = vec![(
        Declaration::Function {
            name: "main".to_string(),
            arguments: vec![
                (
                    Var {
                        name: "a".to_string(),
                        ty: Type::Int,
                    },
                    8..14,
                ),
                (
                    Var {
                        name: "b".to_string(),
                        ty: Type::Int,
                    },
                    16..22,
                ),
            ],
            body: Statement::Block(vec![(
                Statement::Let {
                    var: Var {
                        name: "c".to_string(),
                        ty: Type::Int,
                    },
                    value: Some((Expression::Literal(Int(1)), 51..52)),
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

#[test]
fn test_return_fn() {
    let src = "fn main() {return 1;}";
    let lexer = tokenize(src);
    let toks = lexer.map(Result::unwrap).collect();
    let parser = Parser::new(toks);

    let decls = parser
        .map(Result::unwrap)
        .collect::<Vec<Spanned<Declaration>>>();
    let expected = vec![(
        Declaration::Function {
            name: "main".to_string(),
            arguments: vec![],
            body: Statement::Block(vec![(
                Statement::Return(Some((Expression::Literal(Int(1)), 18..19))),
                11..19,
            )]),
            ret_type: None,
        },
        0..19,
    )];
    assert_eq!(decls.len(), expected.len());
    assert_eq!(decls, expected);
}

#[test]
fn test_if_else() {
    let src = "fn main() { if a { return 1; } else { return 2; }}";
    let lexer = tokenize(src);
    let toks = lexer.map(Result::unwrap).collect();
    let parser = Parser::new(toks);

    let decls = parser
        .map(Result::unwrap)
        .collect::<Vec<Spanned<Declaration>>>();

    let expected = vec![(
        Declaration::Function {
            name: "main".to_string(),
            arguments: vec![],
            body: Statement::Block(vec![(
                Statement::If {
                    condition: (Expression::Ident("a".to_string()), 15..16),
                    then_block: Box::new((
                        Statement::Block(vec![(
                            Statement::Return(Some((Expression::Literal(Int(1)), 26..27))),
                            19..27,
                        )]),
                        17..27,
                    )),
                    else_block: Some(Box::new((
                        Statement::Block(vec![(
                            Statement::Return(Some((Expression::Literal(Int(2)), 45..46))),
                            38..46,
                        )]),
                        36..46,
                    ))),
                },
                12..46,
            )]),
            ret_type: None,
        },
        0..46,
    )];

    assert_eq!(decls.len(), expected.len());
    assert_eq!(decls, expected);
}

#[test]
fn test_type_enum() {
    let src = "type Test = Red;";
    let lexer = tokenize(src);
    let toks = lexer.map(Result::unwrap).collect();
    let parser = Parser::new(toks);

    let decls = parser
        .map(Result::unwrap)
        .collect::<Vec<Spanned<Declaration>>>();
    let expected = vec![(
        Declaration::Enum {
            name: "Test".to_string(),
            variants: vec![("Red".to_string(), 12..15)],
        },
        0..15,
    )];
    assert_eq!(decls.len(), expected.len());
    assert_eq!(decls, expected);
}

#[test]
fn test_type_record() {
    let src = "type Test = { a: int, b: int }";

    let lexer = tokenize(src);
    let toks = lexer.map(Result::unwrap).collect();
    let parser = Parser::new(toks);

    let decls = parser
        .map(Result::unwrap)
        .collect::<Vec<Spanned<Declaration>>>();
    let expected = vec![(
        Declaration::Record {
            name: "Test".to_string(),
            fields: vec![
                (
                    Var {
                        name: "a".to_string(),
                        ty: Type::Int,
                    },
                    14..20,
                ),
                (
                    Var {
                        name: "b".to_string(),
                        ty: Type::Int,
                    },
                    22..28,
                ),
            ],
        },
        0..28,
    )];
    assert_eq!(decls.len(), expected.len());
    assert_eq!(decls, expected);
}

#[test]
fn test_fn_call() {
    let src = "fn main() { let a: int = add(1, 2); }";
    let lexer = tokenize(src);
    let toks = lexer.map(Result::unwrap).collect();
    let parser = Parser::new(toks);

    let decls = parser
        .map(Result::unwrap)
        .collect::<Vec<Spanned<Declaration>>>();
    let expected = vec![(
        Declaration::Function {
            name: "main".to_string(),
            arguments: vec![],
            body: Statement::Block(vec![(
                Statement::Let {
                    var: Var {
                        name: "a".to_string(),
                        ty: Type::Int,
                    },
                    value: Some((
                        Expression::FnCall(
                            "add".to_string(),
                            vec![
                                (Expression::Literal(Int(1)), 29..30),
                                (Expression::Literal(Int(2)), 32..33),
                            ],
                        ),
                        25..34,
                    )),
                },
                16..34,
            )]),
            ret_type: None,
        },
        0..34,
    )];
    assert_eq!(decls.len(), expected.len());
    assert_eq!(decls, expected);
}

#[test]
fn test_fn_call_as_stmt() {
    let src = "fn main() { add(1, 2); }";
    let lexer = tokenize(src);
    let toks = lexer.map(Result::unwrap).collect();
    let parser = Parser::new(toks);

    let decls = parser
        .map(Result::unwrap)
        .collect::<Vec<Spanned<Declaration>>>();
    let expected = vec![(
        Declaration::Function {
            name: "main".to_string(),
            arguments: vec![],
            body: Statement::Block(vec![(
                Statement::Expression(Box::new((
                    Expression::FnCall(
                        "add".to_string(),
                        vec![
                            (Expression::Literal(Int(1)), 16..17),
                            (Expression::Literal(Int(2)), 19..20),
                        ],
                    ),
                    12..21,
                ))),
                12..21,
            )]),
            ret_type: None,
        },
        0..21,
    )];
    assert_eq!(decls.len(), expected.len());
    assert_eq!(decls, expected);
}

#[test]
fn test_bin_op_precedence() {
    let src = "fn main() { return 1 + 2 * 3; }";
    let lexer = tokenize(src);
    let toks = lexer.map(Result::unwrap).collect();
    let parser = Parser::new(toks);

    let decls = parser
        .map(Result::unwrap)
        .collect::<Vec<Spanned<Declaration>>>();
    let expected = vec![(
        Declaration::Function {
            name: "main".to_string(),
            arguments: vec![],
            body: Statement::Block(vec![(
                Statement::Return(Some((
                    Expression::BinOp(
                        BinOp::Add,
                        Box::new((Expression::Literal(Int(1)), 19..20)),
                        Box::new((
                            Expression::BinOp(
                                BinOp::Mul,
                                Box::new((Expression::Literal(Int(2)), 23..24)),
                                Box::new((Expression::Literal(Int(3)), 27..28)),
                            ),
                            23..28,
                        )),
                    ),
                    19..28,
                ))),
                12..28,
            )]),
            ret_type: None,
        },
        0..28,
    )];
    assert_eq!(decls.len(), expected.len());
    assert_eq!(decls, expected);
}

#[test]
fn test_expr_prec() {
    let src = "n == 0 or n == 1";
    let tokens = tokenize(src).map(Result::unwrap).collect();
    let mut parser = Parser::new(tokens);
    let (expr, _) = Expression::parse(&mut parser).unwrap();
    let expected = Expression::BinOp(
        BinOp::Or,
        Box::new((
            Expression::BinOp(
                BinOp::EqEq,
                Box::new((Expression::Ident("n".to_string()), 0..1)),
                Box::new((Expression::Literal(Int(0)), 5..6)),
            ),
            0..6,
        )),
        Box::new((
            Expression::BinOp(
                BinOp::EqEq,
                Box::new((Expression::Ident("n".to_string()), 10..11)),
                Box::new((Expression::Literal(Int(1)), 15..16)),
            ),
            10..16,
        )),
    );
    assert_eq!(expr, expected);
}

#[test]
fn test_bool_fn() {
    let src = "fn test() -> bool { return true; }";
    let lexer = tokenize(src);
    let toks = lexer.map(Result::unwrap).collect();
    let parser = Parser::new(toks);

    let decls = parser
        .map(Result::unwrap)
        .collect::<Vec<Spanned<Declaration>>>();
    let expected = vec![(
        Declaration::Function {
            name: "test".to_string(),
            arguments: vec![],
            body: Statement::Block(vec![(
                Statement::Return(Some((Expression::Literal(Bool(true)), 27..31))),
                20..31,
            )]),
            ret_type: Some(Type::Bool),
        },
        0..31,
    )];
    assert_eq!(decls.len(), expected.len());
    assert_eq!(decls, expected);
}
