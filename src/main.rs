use std::fs;
use std::path::Path;
use std::collections::HashMap;
use clap::Parser;

/// CLI Tool: Organize files in a directory by extension
#[derive(Parser)]
#[command(version = "1.0", author = "Your Name", about = "Organizes files by extension")]
struct Args {
    /// Directory to organize
    #[arg(short, long)]
    dir: String,
}

fn main() {
    let args = Args::parse();
    let dir_path = Path::new(&args.dir);

    if !dir_path.exists() {
        eprintln!("Error: Directory does not exist!");
        return;
    }

    // Create a hashmap to store file extensions and their target folders
    let mut extension_map: HashMap<&str, &str> = HashMap::new();
    extension_map.insert("jpg", "Images");
    extension_map.insert("png", "Images");
    extension_map.insert("txt", "Documents");
    extension_map.insert("pdf", "Documents");
    extension_map.insert("mp4", "Videos");

    // Iterate through files in the directory
    for entry in fs::read_dir(dir_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
            if let Some(target_folder) = extension_map.get(extension) {
                let target_path = dir_path.join(target_folder);

                // Create folder if it doesn't exist
                fs::create_dir_all(&target_path).unwrap();

                // Move file
                let new_path = target_path.join(path.file_name().unwrap());
                fs::rename(&path, &new_path).unwrap();
                println!("Moved: {} -> {:?}", path.display(), new_path);
            }
        }
    }
}
