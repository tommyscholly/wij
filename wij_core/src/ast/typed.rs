#![allow(unused)]
use std::collections::HashMap;

use crate::ast::Type;

use super::{BinOp, Declaration, Expression, Literal, Span, Spanned, Statement, Var};

#[derive(Debug)]
pub enum TypeErrorKind {
    UndefinedVariable(String),
    UndefinedType(String),
    UndefinedFunction(String),
    IdentUsedAsFn(String),
    FunctionArityMismatch {
        expected: u32,
        found: u32,
    },
    TypeMismatch {
        expected: Type,
        found: Type,
    },
    IncompatibleTypes {
        operation: BinOp,
        lhs: Type,
        rhs: Type,
    },
}

pub struct TypeError {
    kind: TypeErrorKind,
    span: Span,
}

impl TypeError {
    pub fn new(kind: TypeErrorKind, span: Span) -> Self {
        Self { kind, span }
    }
}

pub type TypeResult<T> = Result<T, TypeError>;

pub type VarId = u32;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionSignature {
    param_types: Vec<Type>,
    ret_type: Type,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypedVar {
    pub id: VarId,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StatementKind {
    Let {
        var: TypedVar,
        value: Option<TypedExpression>,
    },
    Return(Option<TypedExpression>),
    Block(Vec<TypedStatement>),
    If {
        condition: TypedExpression,
        then_block: Box<TypedStatement>,
        else_block: Option<Box<TypedStatement>>,
    },
    // Match {
    //     value: TypedExpression,
    //     cases: Vec<MatchCase>,
    // },
    Expression(Box<TypedExpression>), // a subclass of expressions being executed for side effects
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypedStatement {
    pub kind: StatementKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ExpressionKind {
    Literal(Literal),
    Ident(String),
    BinOp(BinOp, Box<TypedExpression>, Box<TypedExpression>),
    FnCall(String, Vec<TypedExpression>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypedExpression {
    pub kind: ExpressionKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct EnumVariant {
    pub name: String,
    pub fields: Vec<TypedVar>,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum DeclKind {
    Function {
        name: String,
        arguments: Vec<TypedVar>,
        body: TypedStatement,
        ret_type: Option<Type>,
    },
    Record {
        name: String,
        fields: Vec<(String, Type)>,
    },
    Enum {
        name: String,
        variants: Vec<EnumVariant>,
    },
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypedDecl {
    pub kind: DeclKind,
    pub ty: Type,
    pub span: Span,
}

#[derive(Default)]
pub struct TyCtx {
    var_id_map: HashMap<String, VarId>,
    var_id_type: HashMap<VarId, Type>,
    user_def_types: HashMap<String, Type>,
    decls: Vec<TypedDecl>,
}

impl TyCtx {
    pub fn new() -> TyCtx {
        TyCtx::default()
    }

    fn var_id(&self, name: &str) -> VarId {
        assert!(self.var_id_map.contains_key(name));
        *self.var_id_map.get(name).unwrap()
    }

    fn var_ty(&self, id: VarId) -> &Type {
        assert!(self.var_id_type.contains_key(&id));
        self.var_id_type.get(&id).unwrap()
    }

    fn insert_user_def_type(&mut self, name: String, ty: Type) {
        self.user_def_types.insert(name, ty);
    }

    fn get_user_def_type(&self, name: &str) -> Option<Type> {
        self.user_def_types.get(name).cloned()
    }
}

fn type_var(ctx: &mut TyCtx, var: Spanned<Var>) -> TypeResult<TypedVar> {
    let (var, span) = var;
    let id = ctx.var_id_map.len() as VarId;
    ctx.var_id_map.insert(var.name.clone(), id);
    Ok(TypedVar {
        id,
        ty: var.ty,
        span,
    })
}

fn type_expr(ctx: &mut TyCtx, expr: Spanned<Expression>) -> TypeResult<TypedExpression> {
    let expr = match expr.0 {
        Expression::Literal(lit) => TypedExpression {
            ty: match lit {
                Literal::Int(_) => Type::Int,
                Literal::Bool(_) => Type::Bool,
                Literal::String(_) => Type::String,
            },
            kind: ExpressionKind::Literal(lit),
            span: expr.1,
        },
        Expression::Ident(ident) => TypedExpression {
            ty: ctx.var_ty(ctx.var_id(&ident)).clone(),
            kind: ExpressionKind::Ident(ident),
            span: expr.1,
        },
        Expression::BinOp(op, lhs, rhs) => {
            let ty_lhs = type_expr(ctx, *lhs)?;
            let ty_rhs = type_expr(ctx, *rhs)?;
            let bop_ty = match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => Type::Int,
                BinOp::And
                | BinOp::Or
                | BinOp::EqEq
                | BinOp::NEq
                | BinOp::Gt
                | BinOp::GtEq
                | BinOp::Lt
                | BinOp::LtEq => Type::Bool,
            };

            if ty_lhs.ty != ty_rhs.ty {
                return Err(TypeError::new(
                    TypeErrorKind::IncompatibleTypes {
                        operation: op,
                        lhs: ty_lhs.ty,
                        rhs: ty_rhs.ty,
                    },
                    expr.1,
                ));
            }

            if bop_ty == Type::Int && ty_lhs.ty != Type::Int {
                return Err(TypeError::new(
                    TypeErrorKind::TypeMismatch {
                        expected: Type::Int,
                        found: ty_lhs.ty,
                    },
                    expr.1,
                ));
            }

            TypedExpression {
                kind: ExpressionKind::BinOp(op, Box::new(ty_lhs), Box::new(ty_rhs)),
                ty: bop_ty,
                span: expr.1,
            }
        }
        Expression::FnCall(name, args) => {
            let mut ty_args = vec![];
            for arg in args {
                ty_args.push(type_expr(ctx, arg)?);
            }

            let fn_type = match ctx.get_user_def_type(&name) {
                None => {
                    return Err(TypeError::new(
                        TypeErrorKind::UndefinedFunction(name),
                        expr.1,
                    ));
                }
                Some(t) => t,
            };

            if let Type::Fn(fn_) = fn_type {
                let FunctionSignature {
                    param_types,
                    ret_type,
                } = *fn_;

                if param_types.len() != ty_args.len() {
                    return Err(TypeError::new(
                        TypeErrorKind::FunctionArityMismatch {
                            expected: param_types.len() as u32,
                            found: ty_args.len() as u32,
                        },
                        expr.1,
                    ));
                }

                for (arg, ty_arg) in param_types.iter().zip(ty_args.iter()) {
                    if arg != &ty_arg.ty {
                        return Err(TypeError::new(
                            TypeErrorKind::IncompatibleTypes {
                                operation: BinOp::EqEq,
                                lhs: arg.clone(),
                                rhs: ty_arg.ty.clone(),
                            },
                            expr.1,
                        ));
                    }
                }
            } else {
                return Err(TypeError::new(TypeErrorKind::IdentUsedAsFn(name), expr.1));
            }

            TypedExpression {
                kind: ExpressionKind::FnCall(name, ty_args),
                ty: Type::Unit,
                span: expr.1,
            }
        }
    };

    Ok(expr)
}

fn type_stmt(ctx: &mut TyCtx, stmt: Spanned<Statement>) -> TypeResult<TypedStatement> {
    let stmt = match stmt.0 {
        Statement::Let { var, value } => {
            let val = if let Some(expr) = value {
                Some(type_expr(ctx, expr)?)
            } else {
                None
            };

            TypedStatement {
                kind: StatementKind::Let {
                    var: type_var(ctx, var)?,
                    value: val,
                },
                ty: Type::Unit,
                span: stmt.1,
            }
        }
        Statement::Return(value) => {
            let val = if let Some(expr) = value {
                Some(type_expr(ctx, expr)?)
            } else {
                None
            };

            TypedStatement {
                kind: StatementKind::Return(val),
                ty: Type::Unit,
                span: stmt.1,
            }
        }
        Statement::Block(stmts) => {
            let mut ty_stmts = vec![];
            for stmt in stmts {
                ty_stmts.push(type_stmt(ctx, stmt)?);
            }

            TypedStatement {
                kind: StatementKind::Block(ty_stmts),
                ty: Type::Unit,
                span: stmt.1,
            }
        }
        Statement::If {
            condition,
            then_block,
            else_block,
        } => {
            let else_block = if let Some(stmt) = else_block {
                Some(Box::new(type_stmt(ctx, *stmt)?))
            } else {
                None
            };

            TypedStatement {
                kind: StatementKind::If {
                    condition: type_expr(ctx, condition)?,
                    then_block: Box::new(type_stmt(ctx, *then_block)?),
                    else_block,
                },
                ty: Type::Unit,
                span: stmt.1,
            }
        }
        Statement::Expression(expr) => {
            let expr = type_expr(ctx, *expr)?;
            TypedStatement {
                ty: expr.ty.clone(),
                kind: StatementKind::Expression(Box::new(expr)),
                span: stmt.1,
            }
        }

        Statement::Match { value, cases } => todo!(),
    };

    Ok(stmt)
}

pub fn type_decl(ctx: &mut TyCtx, decl: Spanned<Declaration>) -> TypeResult<TypedDecl> {
    let decl = match decl.0 {
        Declaration::Function {
            name,
            arguments,
            body,
            ret_type,
        } => {
            let mut args = vec![];
            for var in arguments {
                args.push(type_var(ctx, var)?);
            }

            TypedDecl {
                ty: ctx
                    .get_user_def_type(&name)
                    .expect("function type should have been parsed"),
                kind: DeclKind::Function {
                    name,
                    arguments: args,
                    body: type_stmt(ctx, body)?,
                    ret_type,
                },
                span: decl.1,
            }
        }
        Declaration::Enum { name, variants } => TypedDecl {
            ty: Type::Unit,
            kind: DeclKind::Enum {
                name,
                variants: variants
                    .into_iter()
                    .map(|var| EnumVariant {
                        name: var.0,
                        fields: vec![],
                        span: var.1,
                    })
                    .collect(),
            },
            span: decl.1,
        },

        Declaration::Record { name, fields } => {
            let fields: Vec<(String, Type)> = fields
                .into_iter()
                .map(|(var, _)| (var.name, var.ty))
                .collect();

            TypedDecl {
                ty: ctx
                    .get_user_def_type(&name)
                    .expect("record type should have been parsed"),
                kind: DeclKind::Record { name, fields },
                span: decl.1,
            }
        }
    };

    Ok(decl)
}

fn register_types(ctx: &mut TyCtx, decls: &Vec<Declaration>) {
    for decl in decls {
        match decl {
            Declaration::Record { name, fields } => {
                let fields: Vec<(String, Type)> = fields
                    .iter()
                    .map(|(var, _)| (var.name.clone(), var.ty.clone()))
                    .collect();
                let type_ = Type::Record(fields);
                ctx.insert_user_def_type(name.to_string(), type_);
            }
            Declaration::Enum { name, .. } => {
                todo!()
            }
            _ => {}
        }
    }

    for decl in decls {
        if let Declaration::Function {
            name,
            arguments,
            body,
            ret_type,
        } = decl
        {
            let param_types: Vec<Type> = arguments.iter().map(|(var, _)| var.ty.clone()).collect();

            let ret_type = ret_type.clone().unwrap_or(Type::Unit);
            let signature = FunctionSignature {
                param_types,
                ret_type,
            };
            ctx.insert_user_def_type(name.to_string(), Type::Fn(Box::new(signature)));
        }
    }
}

pub fn type_check(decls: Vec<Declaration>) -> bool {
    let mut ctx = TyCtx::new();

    register_types(&mut ctx, &decls);

    todo!()
}
