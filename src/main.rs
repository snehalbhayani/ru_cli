use blake3::{hash, Hash};
use std::collections::HashMap;
use std::collections::HashSet;
use std::io::{self, Write};

fn handle_exit(s: &str) -> bool {
    if s.eq_ignore_ascii_case("exit") {
        println!("Exiting...");
        return true;
    } else {
        return false;
    }
}

fn tokenize_and_index(input: &str, history: &mut HashMap<String, HashSet<String>>) {
    for i in 1..=input.len() {
        // Iterate over slice lengths
        let slice = &input[0..i];
        println!("{}", slice);
        let curr_val = history.entry(slice.to_string()).or_insert(HashSet::new());
        curr_val.insert(input.to_string());
    }
}

fn handle_tab(input: &str) {
    // We need to handle tab and display the matching list of commands from the history
}

fn main() {
    // We need two main types of memory. One is the heap which holds the history
    let mut buffer: String = String::new(); // Holds current input
    let mut history: HashMap<String, HashSet<String>> = HashMap::new(); // Stores tokenized lines

    loop {
        print!("> ");
        io::stdout().flush().unwrap(); // Flush stdout to display prompt immediately

        buffer.clear(); // Clear buffer for new input
        io::stdin()
            .read_line(&mut buffer)
            .expect("Failed to read line");

        let input: String = buffer.trim().to_string(); // Remove trailing newline

        if handle_exit(&input) {
            break;
        }
        tokenize_and_index(&input, &mut history);
    }
    for key in history.keys() {
        println!("key: {:?}", key);
        println!("val: {:?}", history.get(key));
    }
}
