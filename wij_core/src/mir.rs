// This MIR is a CFG
// From the MIR, we then lower into our target IR depending on the codegen backend
// Once we lower to MIR, any errors here are ICE
#![allow(dead_code)]

use std::collections::HashMap;

use crate::{
    Module,
    ast::{
        Type,
        typed::{DeclKind, FunctionSignature, StatementKind, TypedStatement, TypedVar},
    },
};

enum MIRType {
    Int,
    Bool,
    Record(Vec<MIRType>),
    Fn(Vec<MIRType>, Box<MIRType>),
}

impl From<Type> for MIRType {
    fn from(value: Type) -> Self {
        match value {
            Type::Int => MIRType::Int,
            Type::Bool => MIRType::Bool,
            Type::Fn(sig) => {
                let args = sig
                    .param_types
                    .iter()
                    .map(|ty| MIRType::from(ty.clone()))
                    .collect();
                let ret = Box::new(MIRType::from(sig.ret_type.clone()));
                MIRType::Fn(args, ret)
            }
            _ => todo!(),
        }
    }
}

#[derive(Clone, Copy)]
struct BlockID(u32);
#[derive(Clone, Copy)]
struct LocalID(u32);
#[derive(Clone, Copy)]
struct ParamID(u32);
#[derive(Clone, Copy)]
struct FnID(u32);

#[derive(Debug, Default)]
struct LabelBuilder {
    block_id_ctr: u32,
    local_id_ctr: u32,
    param_id_ctr: u32,
    fn_id_ctr: u32,
}

impl LabelBuilder {
    fn new() -> Self {
        Self::default()
    }

    fn block_id(&mut self) -> BlockID {
        let id = self.block_id_ctr;
        self.block_id_ctr += 1;
        BlockID(id)
    }

    fn local_id(&mut self) -> LocalID {
        let id = self.local_id_ctr;
        self.local_id_ctr += 1;
        LocalID(id)
    }

    fn param_id(&mut self) -> ParamID {
        let id = self.param_id_ctr;
        self.param_id_ctr += 1;
        ParamID(id)
    }

    fn fn_id(&mut self) -> FnID {
        let id = self.fn_id_ctr;
        self.fn_id_ctr += 1;
        FnID(id)
    }
}

enum Data {
    Local(LocalID, MIRType),
    Param(ParamID, MIRType),
}

enum Trans {
    Goto {
        target: BlockID,
    },
    Call {
        target: BlockID,
        args: Vec<Data>,
        return_point: Option<BlockID>,
    },
    Return {
        value: Option<Data>,
    },
}

enum Value {
    Immediate(i32),
    Data(Data),
    // Array loading can also be used for records
    Array(Vec<Value>),
}

enum Statement {
    // When codegening, we will always emit a joining block, so the false_block will point to the
    // joining block if there is no actual false block
    If {
        cond: Data,
        true_block: BlockID,
        false_block: BlockID,
    },
    // We need to have some sort of escape analysis to demote to a heap allocation
    // Register (1)
    // Stack (2)
    // Heap (3)
    Allocate {
        local: LocalID,
        ty: MIRType,
    },
    Store {
        location: Data,
        value: Value,
    },
    Load {
        src: Data,
        dst: Data,
    },
    // Loops should drop into a block, which will end in a Trans::Goto to the start
    Loop(BlockID),
}

impl Statement {
    fn lower(cfg: &mut Cfg, stmt: TypedStatement) -> Self {
        todo!()
    }
}

struct Block {
    id: BlockID,
    params: Vec<ParamID>,
    stmts: Vec<Statement>,
    trans: Trans,
}

type BlockPairing = (Vec<TypedStatement>, Option<Trans>);
impl Block {
    fn split_block_by_trans(cfg: &mut Cfg, mut stmts: Vec<TypedStatement>) -> Vec<BlockPairing> {
        let mut blocks = vec![];
        let mut current_stmts: Vec<TypedStatement> = vec![];

        while let Some(stmt) = stmts.pop() {}

        blocks
    }

    fn lower(cfg: &mut Cfg, body: TypedStatement) -> Vec<Self> {
        let StatementKind::Block(stmts) = body.kind else {
            unreachable!()
        };

        let block_id = cfg.label_builder.block_id();
        let mir_stmts: Vec<Statement> = stmts
            .into_iter()
            .map(|stmt| Statement::lower(cfg, stmt))
            .collect();

        todo!()
    }
}

struct Function {
    id: FnID,
    params: Vec<ParamID>,
    entry_block: BlockID,
    blocks: Vec<Block>,
    visibility: bool,
}

impl Function {
    fn lower(
        cfg: &mut Cfg,
        fn_id: FnID,
        arguments: Vec<TypedVar>,
        body: TypedStatement,
        ret_type: Option<Type>,
    ) -> Self {
        let mir_args: Vec<MIRType> = arguments
            .into_iter()
            .map(|var| MIRType::from(var.ty))
            .collect();

        todo!()
    }
}

pub struct Cfg {
    label_builder: LabelBuilder,
    fn_id_map: HashMap<String, FnID>,
    functions: Vec<Function>,
    user_def_types: HashMap<String, MIRType>,
    // constraint: this MIRType is guaranteed to be MIRType::Fn
    foreign_imports: HashMap<String, MIRType>,
}

impl Cfg {
    fn new() -> Cfg {
        Cfg {
            label_builder: LabelBuilder::new(),
            functions: vec![],
            fn_id_map: HashMap::new(),
            user_def_types: HashMap::new(),
            foreign_imports: HashMap::new(),
        }
    }

    fn convert_type_decl(&mut self, decl: DeclKind) {
        match decl {
            DeclKind::Record { name, fields } => {
                let mir_fields = fields
                    .into_iter()
                    .map(|(_, ty)| MIRType::from(ty))
                    .collect();
                let mir_type = MIRType::Record(mir_fields);
                self.user_def_types.insert(name, mir_type);
            }
            DeclKind::Enum { .. } => {}
            _ => unreachable!(),
        }
    }

    // this should be enough information for us to emit LLVM instructions for external functions
    fn register_foreign_imports(&mut self, fds: Vec<(String, FunctionSignature)>) {
        for (fn_name, fd) in fds {
            let fn_ty = Type::Fn(Box::new(fd));
            let mir_fn_ty = MIRType::from(fn_ty);
            self.foreign_imports.insert(fn_name, mir_fn_ty);
        }
    }

    fn assign_fn_id(&mut self, fn_name: String) -> FnID {
        let fn_id = self.label_builder.fn_id();
        self.fn_id_map.insert(fn_name, fn_id);
        fn_id
    }
}

pub fn generate_mir(module: Module) -> Cfg {
    let mut cfg = Cfg::new();

    for decl in module.decls.into_iter() {
        match decl.kind {
            DeclKind::Record { .. } => cfg.convert_type_decl(decl.kind),
            DeclKind::Enum { .. } => cfg.convert_type_decl(decl.kind),
            DeclKind::ForeignDeclarations(ds) => cfg.register_foreign_imports(ds),
            DeclKind::Procedures(_, _) => todo!(),
            DeclKind::Function {
                name,
                arguments,
                body,
                ret_type,
            } => {
                let fn_id = cfg.assign_fn_id(name);
                let mir_fn = Function::lower(&mut cfg, fn_id, arguments, body, ret_type);
                cfg.functions.push(mir_fn);
            }
        }
    }

    todo!()
}
