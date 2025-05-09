use ariadne::{ColorGenerator, Label, Report, ReportKind, Source};
use clap::Parser as Clap;

use wij_core::{ParseError, Parser, tokenize};

#[derive(Clap)]
struct Options {
    file: String,
}

fn report_parse_error(file: &str, contents: &str, e: ParseError) {
    let mut colors = ColorGenerator::new();

    let a = colors.next();

    let span = match e.span {
        Some(span) => span,
        None => 0..0,
    };
    Report::build(ReportKind::Error, (file, span.clone()))
        .with_message("Parse Error")
        .with_label(
            Label::new((file, span))
                .with_message(e.reason.unwrap_or("".to_string()))
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
    let tokens = tokenize(src).collect();
    let parser = Parser::new(tokens);
    let mut prog = Vec::new();
    for decl in parser {
        match decl {
            Ok(decl) => prog.push(decl),
            Err(e) => {
                report_parse_error(&options.file, src, e);
                return;
            }
        }
    }

    println!("{:#?}", prog);
}
