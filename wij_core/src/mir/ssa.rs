use std::collections::{HashMap, HashSet};

use crate::ast::{BinOp, Literal, Type, typed::*};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MIRType {
    Byte,
    Int,
    Usize,
    Bool,
    Str,
    Unit,
    Array(Box<MIRType>),
    Record(Vec<(String, MIRType)>),
    Fn(Vec<MIRType>, Box<MIRType>),
    Ptr,
}

fn convert_type(ty: &Type) -> MIRType {
    match ty {
        Type::Byte => MIRType::Byte,
        Type::Int => MIRType::Int,
        Type::Usize => MIRType::Usize,
        Type::Bool => MIRType::Bool,
        Type::Str => MIRType::Str,
        Type::Unit => MIRType::Unit,
        Type::Array(elem_ty) => MIRType::Array(Box::new(convert_type(elem_ty))),
        Type::Record(fields) => MIRType::Record(
            fields
                .iter()
                .map(|(name, ty)| (name.clone(), convert_type(ty)))
                .collect(),
        ),
        Type::Fn(sig) => {
            let param_types = sig.param_types.iter().map(convert_type).collect();
            MIRType::Fn(param_types, Box::new(convert_type(&sig.ret_type)))
        }
        Type::UserDef(_name) => {
            // todo: fix this by passing in the program
            todo!()
        }
        Type::Ptr(_) => MIRType::Ptr,
        _ => MIRType::Unit, // todo: handle other types
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ValueID(pub u32);
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct BlockID(pub u32);
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct FnID(pub u32);

#[derive(Debug, Clone)]
struct Value {
    id: ValueID,
    #[allow(unused)]
    ty: MIRType,
}

#[derive(Debug, Clone)]
pub enum Operation {
    IntConst(i32),
    UsizeConst(usize),
    BoolConst(bool),
    // unsure if i will actually be doing str constants
    // or if we need to basically allocate a str struct that has procedures on it
    StrConst(String),

    Parameter(usize), // parameter idx

    Alloca(MIRType),         // stack allocation
    Load(ValueID),           // (address)
    Store(ValueID, ValueID), // (address, value)

    // array/record operations, very similar to LLVM
    GetElementPtr(ValueID, Vec<ValueID>), // base + indices
    #[allow(unused)]
    ExtractValue(ValueID, usize), // for records
    #[allow(unused)]
    InsertValue(ValueID, ValueID, usize), // for records (base, new_value, field_idx)

    BinOp {
        op: BinOpKind,
        lhs: ValueID,
        rhs: ValueID,
    },

    Call {
        function: FnID,
        args: Vec<ValueID>,
    },

    Phi {
        incoming: Vec<(BlockID, ValueID)>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl From<&BinOp> for BinOpKind {
    fn from(op: &BinOp) -> Self {
        match op {
            BinOp::Add => BinOpKind::Add,
            BinOp::Sub => BinOpKind::Sub,
            BinOp::Mul => BinOpKind::Mul,
            BinOp::Div => BinOpKind::Div,
            BinOp::Mod => BinOpKind::Mod,
            BinOp::And => BinOpKind::And,
            BinOp::Or => BinOpKind::Or,
            BinOp::EqEq => BinOpKind::Eq,
            BinOp::NEq => BinOpKind::Ne,
            BinOp::Gt => BinOpKind::Gt,
            BinOp::GtEq => BinOpKind::Ge,
            BinOp::Lt => BinOpKind::Lt,
            BinOp::LtEq => BinOpKind::Le,
        }
    }
}

// terminator instruction for a basic block
#[derive(Debug, Clone)]
pub enum Terminator {
    Return(Option<ValueID>),
    // goto, but called br in llvm naming style
    Branch(BlockID),
    CondBranch {
        condition: ValueID,
        true_block: BlockID,
        false_block: BlockID,
    },
    #[allow(unused)]
    Switch {
        value: ValueID,
        cases: Vec<(i32, BlockID)>,
        default: BlockID,
    },
    #[allow(unused)]
    Call {
        function: FnID,
        args: Vec<ValueID>,
        return_to: BlockID,
    },
    Unreachable,
}

#[derive(Debug)]
pub struct Block {
    pub id: BlockID,
    pub instructions: Vec<(ValueID, Operation)>,
    pub terminator: Terminator,
    // unsure if i'm doing analysis, but if i am, this will be needed
    #[allow(unused)]
    pub(crate) predecessors: HashSet<BlockID>,
}

#[derive(Debug)]
pub struct Function {
    pub id: FnID,
    pub name: String,
    pub params: Vec<(String, MIRType)>,

    pub return_type: Option<MIRType>,

    pub entry_block: BlockID,
    pub blocks: HashMap<BlockID, Block>,

    // unsure if i'm doing analysis, but dominator tree is needed for most cfg analysis
    #[allow(unused)]
    pub(crate) dominators: HashMap<BlockID, HashSet<BlockID>>,

    // map var names to value ids
    pub symbols: HashMap<String, ValueID>,
}

#[derive(Debug, Default)]
pub struct Program {
    pub name: String,
    pub functions: HashMap<FnID, Function>,
    // allows faster fn lookup, but more importantly, recursive functions
    pub fns_by_name: HashMap<String, FnID>,

    pub entry_function: Option<FnID>,

    pub types: HashMap<String, MIRType>,
    pub externals: HashMap<String, (FnID, FunctionType)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType {
    pub params: Vec<MIRType>,
    pub return_type: Option<MIRType>,
}

#[derive(Default, Debug)]
struct SSABuilder {
    next_value_id: u32,
    next_block_id: u32,
    next_fn_id: u32,

    current_function: Option<FnID>,
    current_block: Option<BlockID>,

    var_defs: HashMap<String, ValueID>,

    #[allow(unused)]
    incomplete_phis: Vec<(ValueID, String)>,

    entry_fn_name: Option<String>,
}

impl SSABuilder {
    fn new(entry_fn_name: Option<String>) -> Self {
        Self {
            entry_fn_name,
            ..Default::default()
        }
    }

    fn build_module(&mut self, typed_module: Module, program: &mut Program) {
        self.register_types(program, &typed_module);

        for decl in &typed_module.decls {
            match &decl.kind {
                DeclKind::Procedures(_, fns) => {
                    for fn_decl in fns {
                        let DeclKind::Function {
                            name,
                            arguments,
                            body,
                            ret_type,
                        } = &fn_decl.kind
                        else {
                            panic!("help")
                        };

                        println!("lowering procedure: {}", name);
                        self.lower_function(program, name, arguments, body, ret_type.clone());
                    }
                }
                DeclKind::Function {
                    name,
                    arguments,
                    body,
                    ret_type,
                } => {
                    println!("lowering fn: {}", name);
                    let fn_id =
                        self.lower_function(program, name, arguments, body, ret_type.clone());

                    // todo: fix this to look for main as the entry function
                    if program.entry_function.is_none() {
                        match &self.entry_fn_name {
                            Some(fnname) => {
                                if fnname == name {
                                    program.entry_function = Some(fn_id);
                                }
                            }
                            None => {
                                program.entry_function = Some(fn_id);
                            }
                        }
                    }
                }
                DeclKind::ForeignDeclarations(fds) => {
                    for (name, sig) in fds {
                        self.register_external(program, name, sig);
                    }
                }
                _ => {}
            }
        }
    }

    fn lower_function(
        &mut self,
        program: &mut Program,
        name: &str,
        arguments: &[TypedVar],
        body: &TypedStatement,
        ret_type: Option<Type>,
    ) -> FnID {
        let fn_id = FnID(self.next_fn_id);
        self.next_fn_id += 1;

        let entry_block = self.new_block();

        let mut function = Function {
            id: fn_id,
            name: name.to_string(),
            params: Vec::new(),
            return_type: ret_type.as_ref().map(convert_type),
            entry_block,
            blocks: HashMap::new(),
            dominators: HashMap::new(),
            symbols: HashMap::new(),
        };

        let mut entry = Block {
            id: entry_block,
            instructions: Vec::new(),
            terminator: Terminator::Unreachable, // filled in later
            predecessors: HashSet::new(),
        };

        self.current_function = Some(fn_id);
        self.current_block = Some(entry_block);
        self.var_defs.clear();

        for (idx, arg) in arguments.iter().enumerate() {
            let param_ty = convert_type(&arg.ty);
            function.params.push((arg.id.clone(), param_ty.clone()));

            let param_value = self.new_value(param_ty);
            entry
                .instructions
                .push((param_value.id, Operation::Parameter(idx)));

            self.var_defs.insert(arg.id.clone(), param_value.id);
            function.symbols.insert(arg.id.clone(), param_value.id);
        }

        program.fns_by_name.insert(name.to_string(), fn_id);
        function.blocks.insert(entry_block, entry);
        self.create_new_block(&mut function);
        self.lower_statement(program, &mut function, body);

        // if a block has no terminator, it is a return that has no value
        if let Some(block_id) = self.current_block {
            if let Some(block) = function.blocks.get_mut(&block_id) {
                if matches!(block.terminator, Terminator::Unreachable) {
                    block.terminator = Terminator::Return(None);
                }
            }
        }

        // all phis should be resolvable by the end of a function
        self.resolve_phis(&mut function);

        program.functions.insert(fn_id, function);
        fn_id
    }

    fn create_new_block(&mut self, function: &mut Function) -> BlockID {
        let block_id = self.new_block();
        if let Some(curr_block_id) = self.current_block {
            if let Some(block) = function.blocks.get_mut(&curr_block_id) {
                if matches!(block.terminator, Terminator::Unreachable) {
                    block.terminator = Terminator::Branch(block_id);
                }
            }
        }

        let block = Block {
            id: block_id,
            instructions: Vec::new(),
            terminator: Terminator::Unreachable,
            predecessors: HashSet::from_iter(vec![self.current_block.unwrap()]),
        };
        function.blocks.insert(block_id, block);
        self.current_block = Some(block_id);

        block_id
    }

    fn lower_statement(
        &mut self,
        program: &Program,
        function: &mut Function,
        stmt: &TypedStatement,
    ) {
        match &stmt.kind {
            StatementKind::Let { var, value } => {
                if let Some(expr) = value {
                    let value_id = self.lower_expression(program, function, expr);

                    self.var_defs.insert(var.id.clone(), value_id);
                    function.symbols.insert(var.id.clone(), value_id);
                } else {
                    // todo: if there is no initializer, should be allocating memory?
                    // i suppose in a gc'd language that is fine? since we don't care about
                    // optimial memory allocation
                    let alloc_id = self.add_instruction_to_current_block(
                        function,
                        Operation::Alloca(convert_type(&var.ty)),
                        MIRType::Ptr,
                    );

                    self.var_defs.insert(var.id.clone(), alloc_id);
                    function.symbols.insert(var.id.clone(), alloc_id);
                }
            }

            StatementKind::Block(statements) => {
                for sub_stmt in statements {
                    self.lower_statement(program, function, sub_stmt);
                }
            }

            StatementKind::Return(expr_opt) => {
                let value_id = expr_opt
                    .as_ref()
                    .map(|expr| self.lower_expression(program, function, expr));

                if let Some(block_id) = self.current_block {
                    if let Some(block) = function.blocks.get_mut(&block_id) {
                        block.terminator = Terminator::Return(value_id);
                    }
                }

                self.current_block = Some(self.new_block());
            }

            StatementKind::If {
                condition,
                then_block,
                else_block,
            } => {
                let cond_value = self.lower_expression(program, function, condition);

                let then_block_id = self.new_block();
                let else_block_id = self.new_block();
                let merge_block_id = self.new_block();

                let current_block_id = self.current_block.unwrap();
                if let Some(block) = function.blocks.get_mut(&current_block_id) {
                    block.terminator = Terminator::CondBranch {
                        condition: cond_value,
                        true_block: then_block_id,
                        false_block: else_block_id,
                    };
                }

                function.blocks.insert(
                    then_block_id,
                    Block {
                        id: then_block_id,
                        instructions: Vec::new(),
                        terminator: Terminator::Branch(merge_block_id),
                        predecessors: [current_block_id].into_iter().collect(),
                    },
                );

                function.blocks.insert(
                    else_block_id,
                    Block {
                        id: else_block_id,
                        instructions: Vec::new(),
                        terminator: Terminator::Branch(merge_block_id),
                        predecessors: [current_block_id].into_iter().collect(),
                    },
                );

                function.blocks.insert(
                    merge_block_id,
                    Block {
                        id: merge_block_id,
                        instructions: Vec::new(),
                        terminator: Terminator::Unreachable,
                        predecessors: [then_block_id, else_block_id].into_iter().collect(),
                    },
                );

                let outer_vars = self.var_defs.clone();

                self.current_block = Some(then_block_id);
                self.lower_statement(program, function, then_block);
                let then_vars = self.var_defs.clone();

                self.current_block = Some(else_block_id);
                if let Some(else_stmt) = else_block {
                    self.lower_statement(program, function, else_stmt);
                }
                let else_vars = self.var_defs.clone();

                self.current_block = Some(merge_block_id);

                let mut all_modified_vars = HashSet::new();
                for (var, _) in then_vars
                    .iter()
                    .filter(|(k, v)| outer_vars.get(*k) != Some(v))
                {
                    all_modified_vars.insert(var.clone());
                }

                for (var, _) in else_vars
                    .iter()
                    .filter(|(k, v)| outer_vars.get(*k) != Some(v))
                {
                    all_modified_vars.insert(var.clone());
                }

                for var in all_modified_vars {
                    let var_type = if let Some(_value_id) =
                        then_vars.get(&var).or_else(|| else_vars.get(&var))
                    {
                        // todo: fix these place holders
                        MIRType::Int // placeholder
                    } else {
                        MIRType::Int // placeholder 
                    };

                    let phi_value_id = self.new_value(var_type.clone()).id;

                    if let Some(block) = function.blocks.get_mut(&merge_block_id) {
                        let incoming = vec![
                            (
                                then_block_id,
                                *then_vars
                                    .get(&var)
                                    .unwrap_or(&ValueID(self.next_value_id - 1)),
                            ),
                            (
                                else_block_id,
                                *else_vars
                                    .get(&var)
                                    .unwrap_or(&ValueID(self.next_value_id - 1)),
                            ),
                        ];

                        block
                            .instructions
                            .push((phi_value_id, Operation::Phi { incoming }));
                    }

                    self.var_defs.insert(var.clone(), phi_value_id);
                }
            }

            StatementKind::Expression(expr) => {
                self.lower_expression(program, function, expr);
            }
            _ => {
                println!("need to handle stmt: {:?}", stmt);
                todo!()
            }
        }
    }

    fn lower_expression(
        &mut self,
        program: &Program,
        function: &mut Function,
        expr: &TypedExpression,
    ) -> ValueID {
        match &expr.kind {
            ExpressionKind::Literal(lit) => match lit {
                Literal::Int(val) => self.add_instruction_to_current_block(
                    function,
                    Operation::IntConst(*val),
                    MIRType::Int,
                ),
                Literal::Usize(val) => self.add_instruction_to_current_block(
                    function,
                    Operation::UsizeConst(*val),
                    MIRType::Usize,
                ),
                Literal::Bool(val) => self.add_instruction_to_current_block(
                    function,
                    Operation::BoolConst(*val),
                    MIRType::Bool,
                ),
                Literal::Str(val) => self.add_instruction_to_current_block(
                    function,
                    Operation::StrConst(val.clone()),
                    MIRType::Str,
                ),
            },

            ExpressionKind::Ident(name) => *self.var_defs.get(name).unwrap_or_else(|| {
                panic!("Undefined variable: {}", name);
            }),

            ExpressionKind::BinOp(op, lhs, rhs) => {
                let lhs_value = self.lower_expression(program, function, lhs);
                let rhs_value = self.lower_expression(program, function, rhs);

                let op_kind = BinOpKind::from(op);

                self.add_instruction_to_current_block(
                    function,
                    Operation::BinOp {
                        op: op_kind,
                        lhs: lhs_value,
                        rhs: rhs_value,
                    },
                    convert_type(&expr.ty),
                )
            }

            ExpressionKind::FnCall(func_name, args) => {
                let arg_values = args
                    .iter()
                    .map(|arg| self.lower_expression(program, function, arg))
                    .collect();

                let fn_id = program
                    .fns_by_name
                    .get(func_name.as_str())
                    .unwrap_or_else(|| {
                        println!("{:?}", program.fns_by_name);
                        panic!(
                            "function {} should exist if we are trying to call it",
                            func_name.as_str(),
                        )
                    });

                self.add_instruction_to_current_block(
                    function,
                    Operation::Call {
                        function: *fn_id,
                        args: arg_values,
                    },
                    convert_type(&expr.ty),
                )
            }

            ExpressionKind::Array(elements) => {
                let elem_type = if let MIRType::Array(elem_ty) = convert_type(&expr.ty) {
                    *elem_ty
                } else {
                    unreachable!();
                };

                let array_ptr = self.add_instruction_to_current_block(
                    function,
                    Operation::Alloca(MIRType::Array(Box::new(elem_type.clone()))),
                    MIRType::Ptr,
                );

                for (i, elem) in elements.iter().enumerate() {
                    let elem_value = self.lower_expression(program, function, elem);

                    let idx_value = self.add_instruction_to_current_block(
                        function,
                        Operation::IntConst(i as i32),
                        MIRType::Int,
                    );

                    let elem_ptr = self.add_instruction_to_current_block(
                        function,
                        Operation::GetElementPtr(array_ptr, vec![idx_value]),
                        MIRType::Ptr,
                    );

                    self.add_instruction_to_current_block(
                        function,
                        Operation::Store(elem_ptr, elem_value),
                        MIRType::Unit,
                    );
                }

                array_ptr
            }

            ExpressionKind::Idx(array, index) => {
                let array_value = self.lower_expression(program, function, array);
                let index_value = self.lower_expression(program, function, index);

                let elem_ptr = self.add_instruction_to_current_block(
                    function,
                    Operation::GetElementPtr(array_value, vec![index_value]),
                    MIRType::Ptr,
                );

                self.add_instruction_to_current_block(
                    function,
                    Operation::Load(elem_ptr),
                    convert_type(&expr.ty),
                )
            }

            ExpressionKind::FieldAccess(record, field_name) => {
                let record_value = self.lower_expression(program, function, record);
                let Type::UserDef(record_name) = &expr.ty else {
                    // this should be unreachable due to type checking
                    unreachable!();
                };

                let field_idx = self
                    .record_field_idx(program, record_name, field_name)
                    .unwrap() as i32;

                // deref the ptr
                let idx_zero = self.add_instruction_to_current_block(
                    function,
                    Operation::IntConst(0),
                    MIRType::Int,
                );

                // get field idx
                let idx_one = self.add_instruction_to_current_block(
                    function,
                    Operation::IntConst(field_idx),
                    MIRType::Int,
                );

                let field_ptr = self.add_instruction_to_current_block(
                    function,
                    Operation::GetElementPtr(record_value, vec![idx_zero, idx_one]),
                    MIRType::Ptr,
                );

                self.add_instruction_to_current_block(
                    function,
                    Operation::Load(field_ptr),
                    convert_type(&expr.ty),
                )
            }
            ExpressionKind::RecordInit(record_type, assignments) => {
                let record_mir_ty = program
                    .types
                    .get(record_type)
                    .unwrap_or_else(|| panic!("unbound type {}", record_type));

                let record_ptr = self.add_instruction_to_current_block(
                    function,
                    Operation::Alloca(record_mir_ty.clone()),
                    MIRType::Ptr,
                );

                for (field_name, expr) in assignments {
                    let field_value = self.lower_expression(program, function, expr);
                    let field_idx = self
                        .record_field_idx(program, record_type, field_name)
                        .unwrap_or_else(|| {
                            panic!("field {} not found in record {}", field_name, record_type)
                        });

                    let insert_val = Operation::InsertValue(record_ptr, field_value, field_idx);

                    self.add_instruction_to_current_block(function, insert_val, MIRType::Unit);
                }

                record_ptr
            }
            ExpressionKind::Self_ => {
                let self_ptr = function.symbols.get("self").unwrap_or_else(|| {
                    panic!("self not found in function {}", function.name);
                });

                *self_ptr
            }
            e => {
                println!("need to handle expr: {:#?}", e);

                todo!()
            }
        }
    }

    fn add_instruction_to_current_block(
        &mut self,
        function: &mut Function,
        operation: Operation,
        ty: MIRType,
    ) -> ValueID {
        let value_id = self.new_value(ty).id;

        if let Some(block_id) = self.current_block {
            if let Some(block) = function.blocks.get_mut(&block_id) {
                block.instructions.push((value_id, operation));
            } else {
                let block = Block {
                    id: block_id,
                    instructions: vec![(value_id, operation)],
                    terminator: Terminator::Unreachable, // tmp, filled in above
                    predecessors: HashSet::new(),
                };
                function.blocks.insert(block_id, block);
            }
        }

        value_id
    }

    fn register_external(&mut self, program: &mut Program, name: &str, sig: &FunctionSignature) {
        let param_types = sig.param_types.iter().map(convert_type).collect();
        let return_type = convert_type(&sig.ret_type);
        // believe externals still need fn_ids
        let external_fn_id = FnID(self.next_fn_id);
        self.next_fn_id += 1;

        program.externals.insert(
            name.to_string(),
            (
                external_fn_id,
                FunctionType {
                    params: param_types,
                    return_type: Some(return_type),
                },
            ),
        );

        program.fns_by_name.insert(name.to_string(), external_fn_id);
    }

    fn record_field_idx(
        &self,
        program: &Program,
        record_name: &str,
        field_name: &str,
    ) -> Option<usize> {
        match program.types.get(record_name) {
            Some(MIRType::Record(fields)) => fields.iter().position(|(name, _)| name == field_name),
            _ => None,
        }
    }

    fn register_types(&mut self, program: &mut Program, typed_module: &Module) {
        for decl in &typed_module.decls {
            match &decl.kind {
                DeclKind::Record { name, fields } => {
                    let mir_fields = fields
                        .iter()
                        .map(|(field_name, ty)| (field_name.clone(), convert_type(ty)))
                        .collect();

                    program
                        .types
                        .insert(name.clone(), MIRType::Record(mir_fields));
                }
                DeclKind::Enum {
                    name: _,
                    variants: _,
                } => {
                    // need to figure out how I am representing enums
                    todo!()
                }
                _ => {}
            }
        }
    }

    fn resolve_phis(&mut self, _function: &mut Function) {
        // todo
        #[allow(clippy::needless_return)]
        return;
    }

    fn new_value(&mut self, ty: MIRType) -> Value {
        let id = ValueID(self.next_value_id);
        self.next_value_id += 1;
        Value { id, ty }
    }

    fn new_block(&mut self) -> BlockID {
        let id = BlockID(self.next_block_id);
        self.next_block_id += 1;
        id
    }
}

pub fn build_ssa(mut modules: Vec<Module>) -> Program {
    let mut program = Program {
        name: "todo".to_string(),
        functions: HashMap::new(),
        fns_by_name: HashMap::new(),
        entry_function: None,
        types: HashMap::new(),
        externals: HashMap::new(),
    };

    let mut ssa_builder = SSABuilder::new(Some("main".to_string()));
    while let Some(typed_module) = modules.pop() {
        ssa_builder.build_module(typed_module, &mut program);
    }
    program
}
