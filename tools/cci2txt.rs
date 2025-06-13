//! ```cargo
//! [dependencies]
//! serde = { version = "1.0", features = ["derive"] }
//! serde_json = "1.0"
//! clap = { version = "4.0", features = ["derive"] }
//! ```

// rust-script tools/cci2txt.rs --input-folder /tmp --output-file /tmp/chinese_output.txt
use clap::Parser;
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

#[derive(Parser)]
#[command(author, version, about = "Convert JSONL files to filtered TXT (Chinese majority only)")]
struct Args {
    #[arg(short, long, help = "Input folder containing JSONL files")]
    input_folder: String,

    #[arg(short, long, help = "Output text file path")]
    output_file: String,

    #[arg(short, long, default_value_t = 0.7, help = "Chinese ratio threshold (0.0 to 1.0)")]
    threshold: f64,
}

fn chinese_ratio(text: &str) -> f64 {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    let s = n/3;
    let e = n*2/3;

    let slice = &chars[s..e];

    let chinese_count = slice
        .iter()
        .filter(|&&c| c >= '\u{4e00}' && c <= '\u{9fa5}')
        .count();

    chinese_count as f64 / (e - s) as f64
}


fn process_jsonl_file<P: AsRef<Path>>(file_path: P, threshold: f64) -> Vec<String> {
    let mut texts = Vec::new();
    let file = match File::open(&file_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("无法打开文件 {}: {}", file_path.as_ref().display(), e);
            return texts;
        }
    };
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("无法读取行: {}", e);
                continue;
            }
        };

        let data: serde_json::Result<serde_json::Value> = serde_json::from_str(line.trim());

        match data {
            Ok(json) => {
                if let Some(text) = json.get("text").and_then(|t| t.as_str()) {
                    if threshold == 0.0 {
                        texts.push(text.to_string());
                        continue;
                    }
                    if chinese_ratio(text) >= threshold {
                        texts.push(text.to_string());
                    }
                }
            }
            Err(e) => {
                eprintln!("警告：无法解析一行 JSON 数据: {}", e);
            }
        }
    }

    texts
}

fn main() {
    let args = Args::parse();

    let input_folder = args.input_folder;
    let output_file = args.output_file;
    let threshold = args.threshold;

    let mut all_texts = Vec::new();

    let paths = match fs::read_dir(&input_folder) {
        Ok(paths) => paths,
        Err(e) => {
            eprintln!("无法读取文件夹 {}: {}", input_folder, e);
            return;
        }
    };

    for path in paths {
        let path = match path {
            Ok(p) => p.path(),
            Err(e) => {
                eprintln!("无法读取路径: {}", e);
                continue;
            }
        };

        if let Some(ext) = path.extension() {
            if ext == "jsonl" {
                println!("正在处理文件: {:?}", path);
                let texts = process_jsonl_file(&path, threshold);
                all_texts.extend(texts);
            }
        }
    }

    let mut output = match File::create(&output_file) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("无法创建输出文件 {}: {}", output_file, e);
            return;
        }
    };

    for text in &all_texts {
        if let Err(e) = writeln!(output, "{}\n", text) {
            eprintln!("写入文件时出错: {}", e);
        }
    }

    println!(
        "完成！共提取并写入 {} 条中文占比 ≥70% 的文本到 {}",
        all_texts.len(),
        output_file
    );
}