use std::process::exit;

use ariadne::{ColorGenerator, Label, Report, ReportKind, Source};
use clap::Parser as Clap;

use wij_core::{AstError, ParseError, Parser, tokenize};

#[derive(Clap)]
struct Options {
    file: String,
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

fn main() {
    let options = Options::parse();

    let src = std::fs::read_to_string(&options.file).unwrap();
    let src = src.trim();
    let tokens = tokenize(src)
        .map(|t| match t {
            Ok(t) => t,
            Err(e) => {
                report_error(&options.file, src, "Lex Error", e);
                exit(1);
            }
        })
        .collect();
    let parser = Parser::new(tokens);
    let mut prog = Vec::new();
    for decl in parser {
        match decl {
            Ok(decl) => prog.push(decl),
            Err(e) => {
                report_error(&options.file, src, "Parse Error", e);
                return;
            }
        }
    }

    println!("{:#?}", prog);
}
