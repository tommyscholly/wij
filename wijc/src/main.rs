use std::{path::PathBuf, process::exit};

use ariadne::{ColorGenerator, Label, Report, ReportKind, Source};
use clap::Parser as Clap;

use wij_core::{AstError, Module, Parser, tokenize, type_check, use_analysis};

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
}

fn report_error(file: &str, contents: &str, top_level_msg: &str, e: impl AstError) {
    let mut colors = ColorGenerator::new();

    let a = colors.next();

    let span = match e.span() {
        Some(span) => span,
        None => 0..0,
    };
    Report::build(ReportKind::Error, (file, span.clone()))
        .with_message(top_level_msg)
        .with_label(
            Label::new((file, span))
                .with_message(e.reason())
                .with_color(a),
        )
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
        Ok(p) => p,
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
    let mut dependent_modules = Vec::new();
    for file in module_files {
        let mut file_mods = compile_file(file.to_str().unwrap(), options).unwrap();
        let file_module = file_mods.swap_remove(0);
        module.combine(file_module);
        // file_mods becomes just the dependent modules after the first
        dependent_modules.append(&mut file_mods);
    }

    println!("{:?}", module.exports);

    module
}

fn main() {
    let options = Options::parse();

    if let Some(module) = compile_file(&options.file, &options) {
        println!("{:#?}", module);
    }
}
