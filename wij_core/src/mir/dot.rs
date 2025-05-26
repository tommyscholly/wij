use std::fmt::Write;

use super::ssa::{Block, Function, Operation, Terminator};
use crate::{Graphviz, Program};

impl Graphviz for Program {
    fn dot(&self) -> String {
        let mut dot = String::new();

        writeln!(dot, "digraph Program {{").unwrap();
        writeln!(dot, "    node [shape=box, fontname=\"Courier\"];").unwrap();
        writeln!(dot, "    edge [fontname=\"Arial\"];").unwrap();
        writeln!(dot, "    compound=true;").unwrap();
        writeln!(dot).unwrap();

        for (i, function) in self.functions.values().enumerate() {
            writeln!(dot, "    subgraph cluster_{} {{", i).unwrap();
            writeln!(
                dot,
                "        label=\"Function: {}\";",
                escape_for_dot(&function.name)
            )
            .unwrap();
            writeln!(dot, "        style=rounded;").unwrap();
            writeln!(dot, "        color=blue;").unwrap();
            writeln!(dot).unwrap();

            for block in function.blocks.values() {
                function.add_block_to_dot_with_prefix(&mut dot, block, &format!("f{}_", i));
            }

            writeln!(dot, "    }}").unwrap();
            writeln!(dot).unwrap();
        }

        for (i, function) in self.functions.values().enumerate() {
            for block in function.blocks.values() {
                function.add_edges_to_dot_with_prefix(&mut dot, block, &format!("f{}_", i));
            }
        }

        writeln!(dot, "}}").unwrap();
        dot
    }
}

impl Function {
    fn add_edges_to_dot_with_prefix(&self, dot: &mut String, block: &Block, prefix: &str) {
        let block_name = format!("{}bb{}", prefix, block.id.0);

        match &block.terminator {
            Terminator::Branch(target) => {
                writeln!(dot, "    {} -> {}bb{};", block_name, prefix, target.0).unwrap();
            }
            Terminator::CondBranch {
                true_block,
                false_block,
                ..
            } => {
                writeln!(
                    dot,
                    "    {} -> {}bb{} [label=\"true\", color=green];",
                    block_name, prefix, true_block.0
                )
                .unwrap();
                writeln!(
                    dot,
                    "    {} -> {}bb{} [label=\"false\", color=red];",
                    block_name, prefix, false_block.0
                )
                .unwrap();
            }
            Terminator::Call { return_to, .. } => {
                writeln!(
                    dot,
                    "    {} -> {}bb{} [style=dashed];",
                    block_name, prefix, return_to.0
                )
                .unwrap();
            }
            Terminator::Switch { cases, default, .. } => {
                for (case_val, target) in cases {
                    writeln!(
                        dot,
                        "    {} -> {}bb{} [label=\"{}\"];",
                        block_name, prefix, target.0, case_val
                    )
                    .unwrap();
                }
                writeln!(
                    dot,
                    "    {} -> {}bb{} [label=\"default\", style=dashed];",
                    block_name, prefix, default.0
                )
                .unwrap();
            }
            _ => {}
        }
    }

    fn add_block_to_dot_with_prefix(&self, dot: &mut String, block: &Block, prefix: &str) {
        let block_name = format!("{}bb{}", prefix, block.id.0);

        let mut label = String::new();
        write!(label, "Block {}", block.id.0).unwrap();

        if block.id == self.entry_block && !self.params.is_empty() {
            write!(label, "\\lParams: ").unwrap();
            for (i, (name, ty)) in self.params.iter().enumerate() {
                if i > 0 {
                    write!(label, ", ").unwrap();
                }
                let type_str = format!("{:?}", ty);
                write!(
                    label,
                    "%{}: {}",
                    escape_for_dot(name),
                    escape_for_dot(&type_str)
                )
                .unwrap();
            }
        }

        label.push_str("\\l");

        for (value_id, operation) in &block.instructions {
            write!(label, "%{} = ", value_id.0).unwrap();
            match operation {
                Operation::IntConst(val) => write!(label, "{}", val).unwrap(),
                Operation::BoolConst(val) => write!(label, "{}", val).unwrap(),
                Operation::StrConst(val) => {
                    write!(label, "\\\"{}\\\"", escape_string_literal_for_dot(val)).unwrap()
                }
                Operation::Parameter(idx) => write!(label, "param {}", idx).unwrap(),
                Operation::Alloca(ty) => {
                    let type_str = format!("{:?}", ty);
                    write!(label, "alloca {}", escape_for_dot(&type_str)).unwrap();
                }
                Operation::Load(addr, _) => write!(label, "load %{}", addr.0).unwrap(),
                Operation::Store(addr, val) => {
                    write!(label, "store %{}, %{}", val.0, addr.0).unwrap()
                }
                Operation::GetElementPtr(base, indices) => {
                    write!(label, "getelementptr(%{}", base.0).unwrap();
                    for idx in indices {
                        write!(label, ", %{}", idx).unwrap();
                    }
                    write!(label, ")").unwrap();
                }

                Operation::BinOp { op, lhs, rhs } => {
                    let op_str = format!("{:?}", op);
                    write!(label, "{} %{}, %{}", escape_for_dot(&op_str), lhs.0, rhs.0).unwrap();
                }
                Operation::Call { function, args } => {
                    write!(label, "call @fn{}(", function.0).unwrap();
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(label, ", ").unwrap();
                        }
                        write!(label, "%{}", arg.0).unwrap();
                    }
                    write!(label, ")").unwrap();
                }
                Operation::Phi { incoming } => {
                    write!(label, "phi ").unwrap();
                    for (i, (block_id, value_id)) in incoming.iter().enumerate() {
                        if i > 0 {
                            write!(label, ", ").unwrap();
                        }
                        write!(label, "[%{}, bb{}]", value_id.0, block_id.0).unwrap();
                    }
                }
                Operation::InsertValue(base, value, index) => {
                    write!(label, "InsertValue(%{}, %{}, {})", base.0, value.0, index).unwrap();
                }
                _ => {
                    let debug_str = format!("{:?}", operation);
                    write!(label, "{}", escape_for_dot(&debug_str)).unwrap();
                }
            }
            label.push_str("\\l");
        }

        label.push_str("───────────\\l");
        match &block.terminator {
            Terminator::Return(None) => label.push_str("return\\l"),
            Terminator::Return(Some(val)) => {
                write!(label, "return %{}\\l", val.0).unwrap();
            }
            Terminator::Branch(target) => {
                write!(label, "br bb{}\\l", target.0).unwrap();
            }
            Terminator::CondBranch {
                condition,
                true_block,
                false_block,
            } => {
                write!(
                    label,
                    "br %{}, bb{}, bb{}\\l",
                    condition.0, true_block.0, false_block.0
                )
                .unwrap();
            }
            Terminator::Call {
                function,
                args,
                return_to,
            } => {
                write!(label, "call @fn{}(", function.0).unwrap();
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(label, ", ").unwrap();
                    }
                    write!(label, "%{}", arg.0).unwrap();
                }
                write!(label, ") -> bb{}\\l", return_to.0).unwrap();
            }
            _ => {
                let debug_str = format!("{:?}", block.terminator);
                write!(label, "{}\\l", escape_for_dot(&debug_str)).unwrap();
            }
        }

        let style = if block.id == self.entry_block {
            "style=filled, fillcolor=lightgreen"
        } else if matches!(block.terminator, Terminator::Return(_)) {
            "style=filled, fillcolor=lightcoral"
        } else if matches!(block.terminator, Terminator::CondBranch { .. }) {
            "style=filled, fillcolor=lightblue"
        } else {
            "style=filled, fillcolor=white"
        };

        writeln!(
            dot,
            "        {} [label=\"{}\", {}];",
            block_name, label, style
        )
        .unwrap();
    }
}

fn escape_string_literal_for_dot(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\\' => "\\\\".to_string(),  // One backslash becomes two
            '"' => "\\\"".to_string(),   // Escape quotes
            '\n' => "\\\\n".to_string(), // Newline becomes \n literal
            '\r' => "\\\\r".to_string(),
            '\t' => "\\\\t".to_string(),
            '{' => "\\{".to_string(),
            '}' => "\\}".to_string(),
            '<' => "\\<".to_string(),
            '>' => "\\>".to_string(),
            '|' => "\\|".to_string(),
            c if c.is_control() => format!("\\\\x{:02x}", c as u8),
            c => c.to_string(),
        })
        .collect()
}

fn escape_for_dot(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\\' => "\\\\".to_string(),
            '"' => "\\\"".to_string(),
            '\n' => "\\\\n".to_string(),
            '\r' => "\\\\r".to_string(),
            '\t' => "\\\\t".to_string(),
            '{' => "\\{".to_string(),
            '}' => "\\}".to_string(),
            '<' => "\\<".to_string(),
            '>' => "\\>".to_string(),
            '|' => "\\|".to_string(),
            c if c.is_control() => format!("\\\\x{:02x}", c as u8),
            c => c.to_string(),
        })
        .collect()
}
