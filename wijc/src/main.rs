use std::{
    fs,
    path::PathBuf,
    process::{Command, exit},
};

use ariadne::{ColorGenerator, Label, Report, ReportKind, Source};
use clap::Parser as Clap;
use rand::{Rng, rng};

use wij_core::{
    Graphviz, Module, Parser, Program, WijError, build_ssa, tokenize, type_check, use_analysis,
};

#[derive(Clap)]
struct Options {
    file: String,
    #[clap(short, long, required = true)]
    core_path: String,
    #[clap(short, long)]
    lex: bool,
    #[clap(short, long)]
    parse: bool,
    #[clap(short, long)]
    tychk: bool,
    #[clap(short, long)]
    debug: bool,
}

fn report_error(file: &str, contents: &str, top_level_msg: &str, e: impl WijError) {
    let mut rand_state = [0u16; 3];
    rng().fill(&mut rand_state);
    let mut colors = ColorGenerator::from_state(rand_state, 0.5);

    let span = match e.span() {
        Some(span) => span,
        None => 0..0,
    };
    let mut report =
        Report::build(ReportKind::Error, (file, span.clone())).with_message(top_level_msg);

    if e.notes().is_empty() {
        report = report.with_label(
            Label::new((file, span))
                .with_message(e.reason())
                .with_color(colors.next()),
        );
    } else {
        report = report.with_note(e.reason());
    }

    for (msg, span) in e.notes() {
        report = report.with_label(
            Label::new((file, span))
                .with_message(msg)
                .with_color(colors.next()),
        );
    }

    report
        .finish()
        .print((file, Source::from(contents)))
        .unwrap();
}

// This is a special type alias, where the directly compiled file is always first
// and any dependent modules are appended
pub type ResultingModules = Vec<Module>;

fn compile_file(file: &str, options: &Options) -> Option<ResultingModules> {
    let src = std::fs::read_to_string(file).unwrap();
    let src = src.trim();
    let tokens = tokenize(src)
        .map(|t| match t {
            Ok(t) => t,
            Err(e) => {
                report_error(file, src, "Lex Error", e);
                exit(1);
            }
        })
        .collect();
    if options.lex {
        println!("{:#?}", tokens);
        return None;
    }
    let parser = Parser::new(tokens);
    let mut prog = Vec::new();
    for decl in parser {
        match decl {
            Ok(decl) => prog.push(decl),
            Err(e) => {
                report_error(file, src, "Parse Error", e);
                return None;
            }
        }
    }
    if options.parse {
        println!("{:#?}", prog);
        return None;
    }

    let module_uses = use_analysis::extract_module_uses(&prog);
    let mut additional_modules = Vec::new();
    let mut imports = Vec::new();

    for module_import in module_uses {
        let module_name = module_import.join(":");
        let module_file_path =
            use_analysis::resolve_file_from_path(file, &options.core_path, module_import);

        let module_files = use_analysis::files_in_module(&module_file_path);
        match module_files {
            Ok(module_files) => {
                let module = compile_module(module_name, module_files, options);
                imports.append(&mut module.exports.clone());

                additional_modules.push(module);
            }
            Err(e) => {
                panic!("Error loading module {module_name}: {e}");
            }
        }
    }

    let module = match type_check(prog, imports) {
        Ok((module, _)) => module,
        Err(e) => {
            report_error(file, src, "Type Error", e);
            return None;
        }
    };

    if options.tychk {
        println!("{:#?}", module);
        return None;
    }

    let mut resulting_modules = vec![module];
    resulting_modules.append(&mut additional_modules);

    Some(resulting_modules)
}

fn compile_module(module_name: String, module_files: Vec<PathBuf>, options: &Options) -> Module {
    let mut module = Module::new(module_name);
    for file in module_files {
        let file_mods = compile_file(file.to_str().unwrap(), options).unwrap();
        for file_mod in file_mods.into_iter() {
            module.combine(file_mod);
        }
    }

    module
}

fn main() {
    let options = Options::parse();

    if let Some(modules) = compile_file(&options.file, &options) {
        let ssa_mod: Program = build_ssa(modules);

        if options.debug {
            let dot_content = ssa_mod.dot();
            let output_path = &ssa_mod.name;
            fs::write(format!("{}.dot", output_path), dot_content)
                .expect("Failed to write DOT file");

            Command::new("dot")
                .args([
                    "-Tpng",
                    &format!("{}.dot", output_path),
                    "-o",
                    &format!("{}.png", output_path),
                ])
                .output()
                .expect("Failed to execute dot command");
        }
    }
}
