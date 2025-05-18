// This MIR is a CFG
// From the MIR, we then lower into our target IR depending on the codegen backend
// Once we lower to MIR, any errors here are ICE
#![allow(dead_code)]

enum MIRType {}

struct BlockID(u32);
struct LocalID(u32);
struct ParamID(u32);
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

struct Block {
    id: BlockID,
    params: Vec<ParamID>,
    stmts: Vec<Statement>,
    trans: Trans,
}

struct Function {
    id: FnID,
    params: Vec<ParamID>,
    entry_block: BlockID,
    blocks: Vec<Block>,
}
