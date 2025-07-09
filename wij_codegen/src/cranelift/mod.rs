#![allow(unused_imports)]

use std::collections::HashMap;
use std::collections::VecDeque;

use cranelift::codegen;
use cranelift::codegen::ir::immediates::Offset32;
use cranelift::codegen::ir::stackslot::StackSize;
use cranelift::frontend;

use codegen::entity::EntityRef;
use codegen::ir::{
    AbiParam, Function, InstBuilder, Signature, UserFuncName, instructions::BlockArg, types::*,
};
use codegen::isa::CallConv;
use codegen::settings::{self, Flags};
use codegen::verifier::verify_function;
use cranelift::prelude::Block;
use cranelift::prelude::IntCC;
use cranelift::prelude::MemFlags;
use cranelift::prelude::StackSlotData;
use cranelift::prelude::StackSlotKind;
use cranelift::prelude::Value;
use cranelift_module::DataDescription;
use cranelift_module::FuncId;
use cranelift_module::Linkage;
use cranelift_module::{Module, default_libcall_names};
use cranelift_object::{ObjectBuilder, ObjectModule};
use frontend::{FunctionBuilder, FunctionBuilderContext, Variable};

use wij_core::SizeOf;
use wij_core::ssa::BinOpKind;
use wij_core::ssa::BlockID;
use wij_core::ssa::FnID;
use wij_core::ssa::ValueID;
use wij_core::{
    Program, ssa::Block as SSABlock, ssa::Function as SSAFunction, ssa::MIRType,
    ssa::Operation as SSAOperation, ssa::Terminator as SSATerminator,
};

#[derive(Default, Debug)]
struct ProgramCtx {
    // maps fnid to funcid
    fnid_to_funcid: HashMap<u32, FuncId>,
    // todo: maybe better way of finding out what functions return?
    fnid_to_ret_ty: HashMap<u32, Type>,
    variables: HashMap<u32, Variable>,
}

impl ProgramCtx {
    fn declare_variable(
        &mut self,
        varid: u32,
        builder: &mut FunctionBuilder,
        ty: Type,
    ) -> Variable {
        let var = Variable::new(varid as usize);
        builder.declare_var(var, ty);
        self.variables.insert(varid, var);
        var
    }

    fn get_variable(&self, varid: u32) -> Variable {
        self.variables[&varid]
    }
}

struct CraneliftProgram {
    program_name: String,
    fbctx: FunctionBuilderContext,
    module: ObjectModule,
    pctx: ProgramCtx,
}

struct FunctionTranslator<'ctx> {
    builder: FunctionBuilder<'ctx>,
    module: &'ctx mut ObjectModule,
    pctx: &'ctx mut ProgramCtx,
    // fun fact, our BlockId maps directly to Cranelift Blocks
    // both are just newtype patterns over u32
    block_map: HashMap<u32, Block>,
    block_queue: VecDeque<SSABlock>,
    func: SSAFunction,
}

impl CraneliftProgram {
    fn new(program_name: &str) -> Self {
        let flags = Flags::new(settings::builder());

        let isa_builder = cranelift_native::builder().expect("arch isnt supported");
        let isa = isa_builder.finish(flags).expect("isa builder not finished");

        let object_builder = ObjectBuilder::new(isa, program_name, default_libcall_names())
            .expect("object builder not supported");
        let module = ObjectModule::new(object_builder);

        let fbctx = FunctionBuilderContext::new();
        let pctx = ProgramCtx::default();
        Self {
            program_name: program_name.to_string(),
            fbctx,
            module,
            pctx,
        }
    }

    fn output(self) -> anyhow::Result<()> {
        let object = self.module.finish();
        let object_file = format!("{}.o", self.program_name);
        std::fs::write(&object_file, object.emit()?)?;

        std::process::Command::new("cc")
            .arg("wij_intrinsics/intrinsics.c")
            .arg(&object_file)
            .arg("-o")
            .arg(self.program_name)
            .output()?;

        Ok(())
    }

    // for want of a better word than compile
    fn compile(&mut self, program: Program) {
        self.declare_externals(&program);

        let mut function_sigs = HashMap::new();
        for (fnid, func) in program.functions.iter() {
            let param_tys: Vec<Type> = func.params.iter().map(|(_, ty)| ty.to_type()).collect();
            let mut function_signature = self.module.make_signature();
            for ty in &param_tys {
                function_signature.params.push(AbiParam::new(*ty));
            }

            if let Some(ret_ty) = &func.return_type {
                let ret_ty = ret_ty.to_type();
                self.pctx.fnid_to_ret_ty.insert(fnid.0, ret_ty);
                function_signature.returns.push(AbiParam::new(ret_ty));
            }

            // todo: add actual linkage determination based on pub/priv
            let linkage = Linkage::Export;
            let func_id = self
                .module
                .declare_function(&func.name, linkage, &function_signature)
                // todo: error handling
                .unwrap();

            self.pctx.fnid_to_funcid.insert(fnid.0, func_id);

            function_sigs.insert(*fnid, function_signature);
        }

        for (fnid, func) in program.functions {
            self.compile_function(fnid.0, func, function_sigs.remove(&fnid).unwrap());
        }
    }

    fn declare_externals(&mut self, program: &Program) {
        for (fnname, (fnid, fnty)) in &program.externals {
            let linkage = Linkage::Import;
            let mut sig = self.module.make_signature();
            let param_tys: Vec<Type> = fnty.params.iter().map(|ty| ty.to_type()).collect();
            for ty in &param_tys {
                sig.params.push(AbiParam::new(*ty));
            }

            if let Some(ret_ty) = &fnty.return_type {
                let ret_ty = ret_ty.to_type();
                self.pctx.fnid_to_ret_ty.insert(fnid.0, ret_ty);
                sig.returns.push(AbiParam::new(ret_ty));
            }

            let func_id = self
                .module
                .declare_function(fnname, linkage, &sig)
                // todo: error handling
                .unwrap();
            println!("declared extern: {}:{} -> {}", fnname, fnid.0, func_id);
            self.pctx.fnid_to_funcid.insert(fnid.0, func_id);
        }
    }

    fn compile_function(&mut self, fnid: u32, func: SSAFunction, function_signature: Signature) {
        let func_id = *self.pctx.fnid_to_funcid.get(&fnid).unwrap();

        let mut fctx = self.module.make_context();
        fctx.func.signature = function_signature;

        let fb = FunctionBuilder::new(&mut fctx.func, &mut self.fbctx);
        let mut fntrans = FunctionTranslator::new(fb, &mut self.module, &mut self.pctx, func);
        fntrans.translate_function();

        println!("{}", fntrans.builder.func.display());

        if let Err(errors) = verify_function(fntrans.builder.func, fntrans.module.isa()) {
            panic!("Function verification failed:\n{}", errors);
        }

        fntrans.builder.finalize();
        self.module.define_function(func_id, &mut fctx).unwrap();
    }
}

impl<'ctx> FunctionTranslator<'ctx> {
    fn new(
        builder: FunctionBuilder<'ctx>,
        module: &'ctx mut ObjectModule,
        pctx: &'ctx mut ProgramCtx,
        func: SSAFunction,
    ) -> Self {
        Self {
            builder,
            module,
            pctx,
            func,
            block_map: HashMap::new(),
            block_queue: VecDeque::new(),
        }
    }

    fn translate_function(&mut self) {
        let block = self.define_block(self.func.entry_block);
        self.builder.append_block_params_for_function_params(block);
        self.builder.switch_to_block(block);

        for (idx, (param_name, ty)) in self.func.params.iter().enumerate() {
            // need to get the param_id because all references to the param is the id
            let param_id = self.func.symbols.get(param_name).unwrap().id.0;
            let param_var = self
                .pctx
                .declare_variable(param_id, &mut self.builder, ty.to_type());
            let param_val = self.builder.block_params(block)[idx];
            self.builder.def_var(param_var, param_val);
        }

        // let entry_block = self.func.blocks.remove(&self.func.entry_block).unwrap();
        // self.block_queue.push_back(entry_block);
        while let Some(block) = self.block_queue.pop_front() {
            self.translate_block(block);
        }

        if !self.func.blocks.is_empty() {
            // if a block is unreachable in a function, i think it's just an optimization opportunity
            // println!("certain blocks were unreachable: {:#?}", self.func.blocks);
        }
    }

    fn translate_block(&mut self, ssablock: SSABlock) {
        let block = *self.block_map.get(&ssablock.id.0).unwrap();
        self.builder.switch_to_block(block);

        for (val_id, operation) in ssablock.instructions {
            self.translate_operation(val_id.0, operation);
        }
        self.translate_terminator(ssablock.terminator);
        self.builder.seal_block(block);
    }

    fn translate_operation(&mut self, val_id: u32, oper: SSAOperation) {
        use SSAOperation::*;
        match oper {
            IntConst(i) => {
                let ty = MIRType::Int.to_type();
                let const_val = self.builder.ins().iconst(ty, i as i64);
                let var = self.pctx.declare_variable(val_id, &mut self.builder, ty);
                self.builder.def_var(var, const_val);
            }
            UsizeConst(u) => {
                let ty = MIRType::Usize.to_type();
                let const_val = self.builder.ins().iconst(ty, u as i64);
                let var = self.pctx.declare_variable(val_id, &mut self.builder, ty);
                self.builder.def_var(var, const_val);
            }
            BoolConst(b) => {
                let val = if b { 1 } else { 0 };
                let ty = MIRType::Bool.to_type();
                let const_val = self.builder.ins().iconst(ty, val);
                let var = self.pctx.declare_variable(val_id, &mut self.builder, ty);
                self.builder.def_var(var, const_val);
            }
            Parameter(_id) => {
                // don't think this needs to be handled
            }
            Call { function, args } => {
                let call_result = self.generate_call(function, args);

                // if there is no return type, i believe we don't need to define a null var lol
                if let Some(var_ty) = self.pctx.fnid_to_ret_ty.get(&function.0) {
                    let var = self
                        .pctx
                        .declare_variable(val_id, &mut self.builder, *var_ty);
                    self.builder.def_var(var, call_result);
                }
            }
            Alloca(ty) => match ty {
                MIRType::Record(fields) => {
                    let size = fields
                        .iter()
                        .map(|(_, ty)| ty.size_of() as StackSize)
                        .sum::<StackSize>();

                    let data = StackSlotData::new(StackSlotKind::ExplicitSlot, size, 16); // TODO:
                    // understand align_shift more
                    let record_slot = self.builder.create_sized_stack_slot(data);

                    let ty = MIRType::Ptr.to_type();
                    let addr = self.builder.ins().stack_addr(ty, record_slot, 0);
                    let var = self.pctx.declare_variable(val_id, &mut self.builder, ty);
                    self.builder.def_var(var, addr);
                }
                MIRType::Array(_elem_ty) => {
                    todo!()
                }
                _ => {
                    todo!()
                }
            },
            InsertValue(base, value, idx) => {
                let base_var = self.pctx.get_variable(base.0);
                let value_var = self.pctx.get_variable(value.0);
                let base_val = self.builder.use_var(base_var);
                let value_val = self.builder.use_var(value_var);
                let offset = Offset32::new(idx as i32);
                let _field_val =
                    self.builder
                        .ins()
                        .store(MemFlags::new(), value_val, base_val, offset);
            }
            ExtractValue(base, idx, field_type) => {
                let base_var = self.pctx.get_variable(base.0);
                let base_val = self.builder.use_var(base_var);
                let offset = Offset32::new(idx as i32);
                let field_val = self.builder.ins().load(
                    field_type.to_type(),
                    MemFlags::new(),
                    base_val,
                    offset,
                );
                let var =
                    self.pctx
                        .declare_variable(val_id, &mut self.builder, field_type.to_type());
                self.builder.def_var(var, field_val);
            }
            Store(lhs, rhs) => {
                let lhs_var = self.pctx.get_variable(lhs.0);
                let rhs_var = self.pctx.get_variable(rhs.0);
                let lhs_val = self.builder.use_var(lhs_var);
                let rhs_val = self.builder.use_var(rhs_var);
                let _store_val = self
                    .builder
                    .ins()
                    .store(MemFlags::new(), rhs_val, lhs_val, 0);
            }
            Load(ptr, ty) => {
                let ptr_var = self.pctx.get_variable(ptr.0);
                let ptr_val = self.builder.use_var(ptr_var);
                let ty = ty.to_type();
                let field_val = self.builder.ins().load(ty, MemFlags::new(), ptr_val, 0);
                let var = self.pctx.declare_variable(val_id, &mut self.builder, ty);
                self.builder.def_var(var, field_val);
            }
            GetElementPtr(base, indices) => {
                let base_var = self.pctx.get_variable(base.0);
                let base_val = self.builder.use_var(base_var);
                let mut ptr = base_val;

                for idx in indices {
                    let idx_val = self.builder.ins().iconst(I64, idx as i64 * 8);
                    ptr = self.builder.ins().iadd(ptr, idx_val);
                }

                let ty = MIRType::Ptr.to_type();
                let var = self.pctx.declare_variable(val_id, &mut self.builder, ty);
                self.builder.def_var(var, ptr);
            }
            Assign(lhs, rhs) => {
                let lhs_var = self.pctx.get_variable(lhs.0);
                let rhs_var = self.pctx.get_variable(rhs.0);
                let rhs_val = self.builder.use_var(rhs_var);

                self.builder.def_var(lhs_var, rhs_val);
            }
            StrConst(s) => {
                let string_id = self
                    .module
                    .declare_data(&format!("str{}", val_id), Linkage::Local, false, false)
                    .unwrap();

                let mut string_data = DataDescription::new();
                string_data.define(s.as_bytes().into());

                self.module
                    .define_data(string_id, &string_data)
                    .expect("to define string literal");

                let str_ptr = self
                    .module
                    .declare_data_in_func(string_id, self.builder.func);

                // Get pointer to the string data
                let data_ptr = self.builder.ins().global_value(I64, str_ptr);
                let str_len = self.builder.ins().iconst(I32, s.len() as i64);

                // Simple approach - create a struct with ptr + len
                // Allocate space for a string struct (ptr=8 bytes + len=4 bytes + padding=4 bytes = 16 bytes)
                let string_struct_size = 16u32;
                let data = StackSlotData::new(StackSlotKind::ExplicitSlot, string_struct_size, 8);
                let string_slot = self.builder.create_sized_stack_slot(data);
                let string_addr = self.builder.ins().stack_addr(I64, string_slot, 0);

                // Store pointer at offset 0
                self.builder
                    .ins()
                    .store(MemFlags::new(), data_ptr, string_addr, 0);
                // Store length at offset 8 (as i32)
                self.builder
                    .ins()
                    .store(MemFlags::new(), str_len, string_addr, 8);

                let var = self.pctx.declare_variable(val_id, &mut self.builder, I64);
                self.builder.def_var(var, string_addr);
            }
            BinOp { op, lhs, rhs } => {
                let lhs_var = self.pctx.get_variable(lhs.0);
                let rhs_var = self.pctx.get_variable(rhs.0);
                let lhs_val = self.builder.use_var(lhs_var);
                let rhs_val = self.builder.use_var(rhs_var);

                match op {
                    BinOpKind::Eq => {
                        let eq_val = self.builder.ins().icmp(IntCC::Equal, lhs_val, rhs_val);
                        let var = self.pctx.declare_variable(val_id, &mut self.builder, I8);
                        self.builder.def_var(var, eq_val);
                    }
                    BinOpKind::Add => {
                        let add_val = self.builder.ins().iadd(lhs_val, rhs_val);
                        let var = self.pctx.declare_variable(val_id, &mut self.builder, I32);
                        self.builder.def_var(var, add_val);
                    }
                    BinOpKind::Sub => {
                        let sub_val = self.builder.ins().isub(lhs_val, rhs_val);
                        let var = self.pctx.declare_variable(val_id, &mut self.builder, I32);
                        self.builder.def_var(var, sub_val);
                    }
                    BinOpKind::Mul => {
                        let mul_val = self.builder.ins().imul(lhs_val, rhs_val);
                        let var = self.pctx.declare_variable(val_id, &mut self.builder, I32);
                        self.builder.def_var(var, mul_val);
                    }
                    BinOpKind::Or => {
                        let or_val = self.builder.ins().bor(lhs_val, rhs_val);
                        let var = self.pctx.declare_variable(val_id, &mut self.builder, I8);
                        self.builder.def_var(var, or_val);
                    }
                    BinOpKind::And => {
                        let and_val = self.builder.ins().band(lhs_val, rhs_val);
                        let var = self.pctx.declare_variable(val_id, &mut self.builder, I32);
                        self.builder.def_var(var, and_val);
                    }
                    o => panic!("unimplemented bop kind: {:?}", o),
                }
            }
            c => panic!("unimplemented: {:?}", c),
        }
    }

    fn generate_call(&mut self, function: FnID, args: Vec<ValueID>) -> Value {
        let args = args
            .iter()
            .map(|arg| {
                let var = self.pctx.get_variable(arg.0);
                self.builder.use_var(var)
            })
            .collect::<Vec<Value>>();

        let funcid = self
            .pctx
            .fnid_to_funcid
            .get(&function.0)
            .unwrap_or_else(|| panic!("function {} not found", function.0));

        let func_ref = self.module.declare_func_in_func(*funcid, self.builder.func);
        let call = self.builder.ins().call(func_ref, &args);
        let call_results = self.builder.inst_results(call);
        if call_results.is_empty() {
            return self.builder.ins().iconst(I32, 0);
        }

        call_results[0]
    }

    // helper function to define a future block when we encounter it, and add it to the queue
    fn define_block(&mut self, id: BlockID) -> Block {
        let block = self.builder.create_block();
        let ssablock = self.func.blocks.remove(&id).unwrap();

        self.block_map.insert(id.0, block);
        self.block_queue.push_back(ssablock);

        block
    }

    fn translate_terminator(&mut self, terminator: SSATerminator) {
        match terminator {
            SSATerminator::Return(value) => {
                if let Some(value) = value {
                    let var = self.pctx.get_variable(value.0);
                    let val = self.builder.use_var(var);
                    self.builder.ins().return_(&[val]);
                } else {
                    self.builder.ins().return_(&[]);
                }
            }
            SSATerminator::Branch(block_id) => {
                let block_target = self.define_block(block_id);
                self.builder.ins().jump(block_target, &[]);
            }
            SSATerminator::CondBranch {
                condition,
                true_block,
                false_block,
            } => {
                let cond_val = self.pctx.get_variable(condition.0);
                let cond_val = self.builder.use_var(cond_val);
                let true_block_target = self.define_block(true_block);
                let false_block_target = self.define_block(false_block);
                self.builder
                    .ins()
                    .brif(cond_val, true_block_target, &[], false_block_target, &[]);
            }
            SSATerminator::Call {
                function,
                args,
                return_to,
            } => {
                let call_result = self.generate_call(function, args);

                let return_to = self.block_map.get(&return_to.0).unwrap();
                self.builder
                    .ins()
                    .jump(*return_to, &[BlockArg::Value(call_result)]);
            }
            SSATerminator::Switch {
                value: _,
                cases: _,
                default: _,
            } => todo!(),
            SSATerminator::Unreachable => unreachable!(),
        }
    }
}

trait ToType {
    fn to_type(&self) -> Type;
}

impl ToType for MIRType {
    fn to_type(&self) -> Type {
        match self {
            MIRType::Byte => I8,
            MIRType::Int => I32,
            MIRType::Bool => I8,
            MIRType::Unit => I8,
            // all i64s are ptrs
            MIRType::Array(_elem_ty) => I64,
            MIRType::Record(_fields) => I64,
            MIRType::Fn(_args, _ret_ty) => I64,
            MIRType::Str => I64,
            MIRType::Ptr => I64,
            MIRType::Usize => {
                // todo: need to actually determine for my TARGET architecture, and not the one that
                // the compiler is RUNNING on
                // this will come into play when trying to cross compile to arm, if im doing 32 bit
                if cfg!(target_pointer_width = "64") {
                    I64
                } else {
                    I32
                }
            }
        }
    }
}

pub(crate) fn compile(program: Program, program_name: &str) {
    let mut cranelift_prog = CraneliftProgram::new(program_name);
    cranelift_prog.compile(program);
    cranelift_prog.output().unwrap();
}
