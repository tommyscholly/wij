#![allow(unused)]
use std::{collections::HashMap, fmt::Display};

use crate::{
    AstError,
    ast::{self, Type},
};

use super::{BinOp, Declaration, Expression, Literal, Path, Span, Spanned, Statement, Var};

#[derive(Debug)]
pub enum TypeErrorKind {
    UndefinedVariable(String),
    UndefinedType(String),
    UndefinedFunction(String),
    Unassignable,
    IdentUsedAsFn(String),
    DuplicateModule,
    ModuleNotFound,
    InvalidType(Type),
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

impl Display for TypeErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TypeErrorKind::*;
        match self {
            InvalidType(ty) => write!(f, "Invalid type `{}`", ty),
            UndefinedVariable(ident) => write!(f, "Undefined variable `{}`", ident),
            UndefinedType(ident) => write!(f, "Undefined type `{}`", ident),
            UndefinedFunction(ident) => write!(f, "Undefined function `{}`", ident),
            Unassignable => write!(f, "Cannot assign to this expression"),
            IdentUsedAsFn(ident) => write!(f, "Identifier `{}` used as function", ident),
            FunctionArityMismatch { expected, found } => {
                write!(f, "Expected {} arguments, but found {}", expected, found)
            }
            TypeMismatch { expected, found } => {
                write!(f, "Expected `{}` but found `{}`", expected, found)
            }
            IncompatibleTypes {
                operation,
                lhs,
                rhs,
            } => write!(
                f,
                "Incompatible types for operation `{}`: {} and {}",
                operation, lhs, rhs
            ),
            DuplicateModule => write!(f, "Duplicate module"),
            ModuleNotFound => write!(f, "Module not found"),
        }
    }
}

pub struct TypeError {
    kind: TypeErrorKind,
    notes: Vec<(String, Span)>,
    span: Span,
}

impl TypeError {
    pub fn new(kind: TypeErrorKind, span: Span) -> Self {
        Self {
            kind,
            span,
            notes: Vec::new(),
        }
    }

    pub fn add_note(mut self, note: String, span: Span) -> Self {
        self.notes.push((note, span));
        self
    }
}

impl AstError for TypeError {
    fn span(&self) -> Option<Span> {
        Some(self.span.clone())
    }

    fn reason(&self) -> String {
        self.kind.to_string()
    }

    fn notes(&self) -> Vec<(String, Span)> {
        self.notes.clone()
    }
}

pub type TypeResult<T> = Result<T, TypeError>;

pub type VarId = u32;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionSignature {
    pub param_types: Vec<Type>,
    pub ret_type: Type,
}

impl Display for FunctionSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) -> {}",
            self.param_types
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<String>>()
                .join(", "),
            self.ret_type
        )
    }
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
    Assignment(Box<TypedExpression>, Box<TypedExpression>),
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
    FieldAccess(Box<TypedExpression>, String),
    BinOp(BinOp, Box<TypedExpression>, Box<TypedExpression>),
    FnCall(String, Vec<TypedExpression>),
    RecordInit(String, Vec<(String, TypedExpression)>),
    DataConstructor(String, Option<Box<TypedExpression>>),
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
    pub data: Option<Spanned<Type>>,
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
    ForeignDeclarations(Vec<FunctionSignature>),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypedDecl {
    pub kind: DeclKind,
    pub ty: Type,
    pub span: Span,
    pub visible: bool,
}

impl TypedDecl {
    fn name(&self) -> Option<&str> {
        match &self.kind {
            DeclKind::Function { name, .. } => Some(name),
            DeclKind::Record { name, .. } => Some(name),
            DeclKind::Enum { name, .. } => Some(name),
            DeclKind::ForeignDeclarations(..) => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Module {
    pub name: String,
    pub decls: Vec<TypedDecl>,
    // todo: idxs of decls that are exported?
    pub exports: Vec<TypedDecl>,
}

impl Module {
    pub fn new(name: String) -> Module {
        Module {
            name,
            decls: vec![],
            exports: vec![],
        }
    }

    pub fn combine(&mut self, mut other: Module) {
        self.decls.append(&mut other.decls);
        self.exports.append(&mut other.exports);
    }
}

#[derive(Default, Clone, Debug)]
pub struct TyCtx {
    var_id_map: HashMap<String, VarId>,
    var_id_type: HashMap<VarId, Type>,
    user_def_types: HashMap<String, Type>,
    data_constructors: HashMap<String, Option<Type>>,
    decls: Vec<TypedDecl>,
}

impl TyCtx {
    pub fn new() -> TyCtx {
        TyCtx::default()
    }

    fn var_id(&self, name: &str) -> Option<&VarId> {
        self.var_id_map.get(name)
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

    fn insert_data_constructor(&mut self, name: String, ty: Option<Type>) {
        self.data_constructors.insert(name, ty);
    }
}

#[derive(Debug, Clone)]
pub struct ScopedCtx<'a> {
    ctx: TyCtx,
    parent: Option<&'a ScopedCtx<'a>>,
}

impl<'a> ScopedCtx<'a> {
    fn new() -> ScopedCtx<'a> {
        ScopedCtx {
            ctx: TyCtx::new(),
            parent: None,
        }
    }

    fn from_tyctx(ctx: TyCtx) -> ScopedCtx<'a> {
        ScopedCtx { ctx, parent: None }
    }

    fn child(&'a self) -> ScopedCtx<'a> {
        ScopedCtx {
            ctx: TyCtx::new(),
            parent: Some(self),
        }
    }

    fn var_id(&self, name: &str) -> Option<&VarId> {
        match self.ctx.var_id_map.get(name) {
            Some(id) => Some(id),
            None => match self.parent {
                Some(parent) => parent.var_id(name),
                None => None,
            },
        }
    }

    fn var_ty(&self, id: VarId) -> &Type {
        match self.ctx.var_id_type.get(&id) {
            Some(ty) => ty,
            None => match self.parent {
                Some(parent) => parent.var_ty(id),
                None => panic!("var not found: {}", id),
            },
        }
    }

    fn insert_var(&mut self, name: String, ty: Type) -> VarId {
        let id = self.ctx.var_id_map.len() as VarId;
        self.ctx.var_id_map.insert(name, id);
        self.ctx.var_id_type.insert(id, ty);

        id
    }

    fn get_user_def_type(&self, name: &str) -> Option<Type> {
        match self.ctx.get_user_def_type(name) {
            Some(ty) => Some(ty),
            None => match self.parent {
                Some(parent) => parent.get_user_def_type(name),
                None => None,
            },
        }
    }

    fn insert_user_def_type(&mut self, name: String, ty: Type) {
        self.ctx.user_def_types.insert(name, ty);
    }

    fn resolve_type(&self, ty: Type) -> Type {
        if let Type::UserDef(name) = &ty {
            return match self.get_user_def_type(name) {
                Some(ty) => ty,
                None => match self.parent {
                    Some(parent) => parent.resolve_type(ty),
                    None => ty,
                },
            };
        }

        ty
    }
}

fn type_var(ctx: &mut ScopedCtx, var: Spanned<Var>, inferred_ty: Type) -> TypeResult<TypedVar> {
    let (mut var, span) = var;
    match (&var.ty, inferred_ty) {
        (Some(ty), ity) => {
            let ty = ctx.resolve_type(ty.clone());
            let ity = ctx.resolve_type(ity);
            if ty != ity {
                return Err(TypeError::new(
                    TypeErrorKind::TypeMismatch {
                        expected: ty.clone(),
                        found: ity.clone(),
                    },
                    span,
                ));
            }
        }
        (None, ty) => {
            var.ty = Some(ty);
        }
    };

    let id = ctx.insert_var(var.name.clone(), var.ty.clone().unwrap());
    Ok(TypedVar {
        id,
        ty: var.ty.unwrap(),
        span,
    })
}

fn type_expr(ctx: &mut ScopedCtx, expr: Spanned<Expression>) -> TypeResult<TypedExpression> {
    let expr = match expr.0 {
        Expression::RecordInit(record_name, assignments) => {
            let mut ty_assignments = vec![];
            for assignment in assignments {
                let (name, expr) = assignment;
                let ty_expr = type_expr(ctx, expr)?;
                ty_assignments.push((name, ty_expr));
            }

            let record_ty = ctx.get_user_def_type(&record_name);
            match record_ty {
                Some(ty) => TypedExpression {
                    ty: ty.clone(),
                    kind: ExpressionKind::RecordInit(record_name, ty_assignments),
                    span: expr.1,
                },
                None => {
                    return Err(TypeError::new(
                        TypeErrorKind::UndefinedType(record_name),
                        expr.1,
                    ));
                }
            }
        }
        Expression::Literal(lit) => TypedExpression {
            ty: match lit {
                Literal::Int(_) => Type::Int,
                Literal::Bool(_) => Type::Bool,
                Literal::Str(_) => Type::Str,
            },
            kind: ExpressionKind::Literal(lit),
            span: expr.1,
        },
        Expression::Ident(ident) => TypedExpression {
            ty: match ctx.var_id(&ident) {
                Some(id) => ctx.var_ty(*id).clone(),
                None => {
                    return Err(TypeError::new(
                        TypeErrorKind::UndefinedVariable(ident),
                        expr.1,
                    ));
                }
            },
            kind: ExpressionKind::Ident(ident),
            span: expr.1,
        },
        Expression::BinOp(op, lhs, rhs) => {
            let ty_lhs = type_expr(ctx, *lhs)?;
            let ty_rhs = type_expr(ctx, *rhs)?;
            let bop_ty = match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => Type::Int,
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

            let ret_ty = if let Type::Fn(fn_) = fn_type {
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
                            TypeErrorKind::TypeMismatch {
                                expected: arg.clone(),
                                found: ty_arg.ty.clone(),
                            },
                            expr.1,
                        ));
                    }
                }
                ret_type
            } else {
                return Err(TypeError::new(TypeErrorKind::IdentUsedAsFn(name), expr.1));
            };

            TypedExpression {
                kind: ExpressionKind::FnCall(name, ty_args),
                ty: ret_ty,
                span: expr.1,
            }
        }
        Expression::FieldAccess(expr, field_name) => {
            let expr_span = expr.1.clone();
            let ty_expr = type_expr(ctx, *expr)?;
            println!("ty: {:?}", ty_expr.ty);
            let ty = if let Type::UserDef(name) = &ty_expr.ty {
                &ctx.get_user_def_type(name)
                    .expect("userdef types should have been registered")
            } else {
                &ty_expr.ty
            };
            let field_ty = if let Type::Record(fields) = ty {
                println!("fields: {:?}", fields);
                fields.iter().find_map(|(var, ty)| {
                    if *var == field_name {
                        Some(ty.clone())
                    } else {
                        None
                    }
                })
            } else {
                None
            };
            if field_ty.is_none() {
                let mut err = TypeError::new(
                    TypeErrorKind::InvalidType(ty_expr.ty.clone()),
                    expr_span.clone(),
                )
                .add_note(
                    format!("Cannot access field {field_name} on type {}", ty_expr.ty),
                    expr_span,
                );
                return Err(err);
            }
            TypedExpression {
                ty: field_ty.unwrap(),
                kind: ExpressionKind::FieldAccess(Box::new(ty_expr), field_name),
                span: expr_span,
            }
        }
        Expression::DataConstruction(construct_name, data) => {
            let (dacon_ty, enum_ty) = match ctx.get_user_def_type(&construct_name) {
                Some(Type::DataConstructor(dacon, data, ty_name)) => (
                    Type::DataConstructor(dacon, data, ty_name.clone()),
                    Type::UserDef(ty_name),
                ),
                _ => {
                    return Err(TypeError::new(
                        TypeErrorKind::UndefinedType(construct_name),
                        expr.1,
                    ));
                }
            };

            let data = if let Some(data) = data {
                Some(Box::new(type_expr(ctx, *data)?))
            } else {
                None
            };

            TypedExpression {
                kind: ExpressionKind::DataConstructor(construct_name, data),
                ty: enum_ty,
                span: expr.1,
            }
        }
    };

    Ok(expr)
}

fn type_stmt(ctx: &mut ScopedCtx, stmt: Spanned<Statement>) -> TypeResult<TypedStatement> {
    let stmt = match stmt.0 {
        Statement::Assignment(lhs, rhs) => {
            let ty_lhs = type_expr(ctx, *lhs)?;
            let ty_rhs = type_expr(ctx, *rhs)?;

            if ty_lhs.ty != ty_rhs.ty {
                let mut err = TypeError::new(
                    TypeErrorKind::TypeMismatch {
                        expected: ty_lhs.ty.clone(),
                        found: ty_rhs.ty.clone(),
                    },
                    stmt.1,
                )
                .add_note(format!("This has type {}", ty_lhs.ty), ty_lhs.span)
                .add_note(format!("This has type {}", ty_rhs.ty), ty_rhs.span);
                return Err(err);
            }

            if !matches!(
                ty_lhs.kind,
                ExpressionKind::Ident(_) | ExpressionKind::FieldAccess(_, _)
            ) {
                return Err(TypeError::new(TypeErrorKind::Unassignable, stmt.1));
            }

            TypedStatement {
                ty: ty_lhs.ty.clone(),
                kind: StatementKind::Assignment(Box::new(ty_lhs), Box::new(ty_rhs)),
                span: stmt.1,
            }
        }
        Statement::Let { var, value } => {
            // TODO: cleanup these clones
            let (val, inferred_ty) = if let Some(expr) = value {
                let ty_expr = type_expr(ctx, expr)?;
                let inferred_ty = ty_expr.ty.clone();
                (Some(ty_expr), inferred_ty)
            } else {
                (None, Type::Any)
            };

            let var_ty = var.0.ty.clone();
            let ty_var = match type_var(ctx, var, inferred_ty) {
                Ok(ty_var) => ty_var,
                Err(err) => match val {
                    Some(val) => {
                        return Err(TypeError::new(
                            TypeErrorKind::TypeMismatch {
                                expected: var_ty.clone().unwrap(),
                                found: val.ty.clone(),
                            },
                            stmt.1,
                        )
                        .add_note(format!("This has type {}", var_ty.unwrap()), err.span)
                        .add_note(format!("This has type {}", val.ty), val.span));
                    }
                    None => return Err(err),
                },
            };

            TypedStatement {
                kind: StatementKind::Let {
                    var: ty_var,
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
            let mut child_ctx = ctx.child();
            for stmt in stmts {
                ty_stmts.push(type_stmt(&mut child_ctx, stmt)?);
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
            let mut else_ctx = ctx.child();
            let else_block = if let Some(stmt) = else_block {
                Some(Box::new(type_stmt(&mut else_ctx, *stmt)?))
            } else {
                None
            };

            let mut then_ctx = ctx.child();
            let then_block = Box::new(type_stmt(&mut then_ctx, *then_block)?);
            TypedStatement {
                kind: StatementKind::If {
                    condition: type_expr(ctx, condition)?,
                    then_block,
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

pub fn type_decl(ctx: &mut ScopedCtx, decl: Spanned<Declaration>) -> TypeResult<TypedDecl> {
    let decl = match decl.0 {
        Declaration::Function {
            name,
            arguments,
            body,
            ret_type,
        } => {
            let mut body_ctx = ctx.child();
            let mut args = vec![];
            for var in arguments {
                let ity = var.0.ty.clone().unwrap();
                args.push(type_var(&mut body_ctx, var, ity)?);
            }

            TypedDecl {
                ty: match ctx.get_user_def_type(&name) {
                    Some(ty) => ty,
                    None => return Err(TypeError::new(TypeErrorKind::IdentUsedAsFn(name), decl.1)),
                },
                kind: DeclKind::Function {
                    name,
                    arguments: args,
                    body: type_stmt(&mut body_ctx, body)?,
                    ret_type,
                },
                span: decl.1,
                visible: false,
            }
        }
        Declaration::Enum { name, variants } => TypedDecl {
            ty: Type::Unit,
            kind: DeclKind::Enum {
                name,
                variants: variants
                    .into_iter()
                    .map(|var| EnumVariant {
                        name: var.0.name,
                        data: var.0.data,
                        span: var.1,
                    })
                    .collect(),
            },
            span: decl.1,
            visible: false,
        },

        Declaration::Record { name, fields } => {
            let fields: Vec<(String, Type)> = fields
                .into_iter()
                .map(|(var, _)| (var.name, var.ty.unwrap()))
                .collect();

            TypedDecl {
                ty: ctx
                    .get_user_def_type(&name)
                    .expect("record type should have been parsed"),
                kind: DeclKind::Record { name, fields },
                span: decl.1,
                visible: false,
            }
        }
        Declaration::ForeignDeclarations(fds) => {
            let mut ty_fds = vec![];
            for (fd, _) in fds {
                let name = fd.name;
                let sig = fd.sig;
                ctx.insert_user_def_type(name, Type::Fn(Box::new(sig.clone())));
                ty_fds.push(sig);
            }

            TypedDecl {
                ty: Type::Unit,
                kind: DeclKind::ForeignDeclarations(ty_fds),
                span: decl.1,
                visible: false,
            }
        }
        Declaration::Public(decl) => {
            let mut ty_decl = type_decl(ctx, *decl)?;
            ty_decl.visible = true;
            ty_decl
        }
        Declaration::Use(_) => todo!(),
        Declaration::Module(_) => unreachable!(),
    };

    Ok(decl)
}

fn register_types(ctx: &mut TyCtx, decls: &Vec<Spanned<Declaration>>, imports: Vec<TypedDecl>) {
    for decl in decls {
        match &decl.0 {
            Declaration::Record { name, fields } => {
                let fields: Vec<(String, Type)> = fields
                    .iter()
                    .map(|(var, _)| (var.name.clone(), var.ty.clone().unwrap()))
                    .collect();
                let type_ = Type::Record(fields);
                ctx.insert_user_def_type(name.to_string(), type_);
            }
            Declaration::Enum {
                name: enum_name,
                variants,
            } => {
                for (variant, _) in variants {
                    let ast::EnumVariant { name, data, .. } = variant;
                    let data = data.as_ref().map(|data| Box::new(data.0.clone()));
                    ctx.insert_user_def_type(
                        name.to_string(),
                        Type::DataConstructor(name.to_string(), data, enum_name.to_string()),
                    );
                }
            }
            _ => {}
        }
    }

    for decl in decls {
        let (name, arguments, body, ret_type) = match &decl.0 {
            Declaration::Function {
                name,
                arguments,
                body,
                ret_type,
            } => (name, arguments, body, ret_type),
            Declaration::Public(decl) => match &decl.0 {
                Declaration::Function {
                    name,
                    arguments,
                    body,
                    ret_type,
                } => (name, arguments, body, ret_type),
                _ => continue,
            },
            _ => continue,
        };

        let param_types: Vec<Type> = arguments
            .iter()
            .map(|(var, _)| var.ty.clone().unwrap())
            .collect();

        let ret_type = ret_type.clone().unwrap_or(Type::Unit);
        let signature = FunctionSignature {
            param_types,
            ret_type,
        };
        println!("registered fn {}: {}", name, signature);
        ctx.insert_user_def_type(name.to_string(), Type::Fn(Box::new(signature)));
    }

    for decl in imports {
        if let Some(name) = decl.name() {
            ctx.insert_user_def_type(name.to_string(), decl.ty.clone());
        }
    }
}

type ModuleUseDecls = (Module, Vec<Spanned<Path>>, Vec<Spanned<Declaration>>);

// Filters decls into decls, modules and uses
// Should only ever be one module
fn extract_module_uses(decls: Vec<Spanned<Declaration>>) -> TypeResult<ModuleUseDecls> {
    let mut filt_decls: Vec<Spanned<Declaration>> = vec![];
    let mut modules = vec![];
    let mut uses = vec![];
    for decl in decls {
        match decl.0 {
            Declaration::Module(name) => {
                modules.push((name, decl.1));
            }
            Declaration::Use(name) => {
                uses.push((name, decl.1));
            }
            _ => {
                filt_decls.push(decl);
            }
        }
    }

    if modules.len() > 1 {
        return Err(TypeError::new(
            TypeErrorKind::DuplicateModule,
            modules[0].1.start..modules[1].1.end,
        ));
    } else if modules.is_empty() {
        return Err(TypeError::new(
            TypeErrorKind::ModuleNotFound,
            Span::default(),
        ));
    }

    let module = Module {
        name: modules[0].0.clone(),
        decls: vec![],
        exports: vec![],
    };

    Ok((module, uses, filt_decls))
}

pub fn type_check(decls: Vec<Spanned<Declaration>>, imports: Vec<TypedDecl>) -> TypeResult<Module> {
    let mut ctx = TyCtx::new();

    let (mut module, uses, decls) = extract_module_uses(decls)?;

    register_types(&mut ctx, &decls, imports);

    let mut ctx = ScopedCtx::from_tyctx(ctx);
    let mut ty_decls = vec![];
    let mut exports = vec![];
    for (i, decl) in decls.into_iter().enumerate() {
        let ty_decl = type_decl(&mut ctx, decl)?;
        if ty_decl.visible {
            exports.push(ty_decl.clone());
        }
        ty_decls.push(ty_decl);
    }

    module.decls = ty_decls;
    module.exports = exports;

    Ok(module)
}
