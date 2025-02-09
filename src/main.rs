use std::io;
use std::collections::{HashMap, HashSet};
use tokio;
mod rnn;
mod db;
mod utilities;
mod terminal;


async fn startup_training () {
    let db_path = "my_cli/history.db"; // Change to your SQLite database path
    let commands = rnn::load_commands(db_path);
    println!("Loaded {:?} commands from the history", commands.len());
    let (char_to_idx, idx_to_char) = rnn::build_vocab(&commands);
    println!("Vocab built with {:?} characters", char_to_idx.len());
    let hidden_size = 40;    
    let model = rnn::train_model(1000, 5, hidden_size, &char_to_idx, &idx_to_char, commands);
    tokio::time::sleep(tokio::time::Duration::from_millis(10000000)).await; // Simulate async delay

}


#[tokio::main]
async fn main() {
    
    // Run training asynchronously
    let handle: tokio::task::JoinHandle<()> = tokio::spawn(async {
        startup_training().await;
    });

    // let start_seq = "oo2";
    // println!("{:?}", char_to_idx.get(&'3'));
    // let predicted_text = rnn::generate_text(&model, &start_seq, &char_to_idx, &idx_to_char, 1 as usize, hidden_size);
    // println!("{:?}", predicted_text);

    // First load the history of commands
    let conn: rusqlite::Connection = db::setup_db();
    let history_data: Result<Vec<String>, rusqlite::Error> = db::get_full_history(&conn);
    let mut history_ptr: u32 = 0;
    let mut buffer: String = String::new();
    let mut history: HashMap<String, Vec<String>> = HashMap::new(); // Stores tokenized lines
    for result in history_data.unwrap() {
        utilities::tokenize_and_index(&result, &mut history);    
    }

    let _res: Result<(), io::Error> = terminal::listen_key_strokes(&mut buffer, &mut history, &conn);    
    
    println!("Waiting for training to complete...");
}
