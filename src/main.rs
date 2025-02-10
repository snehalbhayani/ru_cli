use rusqlite::{params, Connection, Result};
use std::io;
use std::collections::HashMap;
use tokio;
mod rnn;
mod db;
mod utilities;
mod terminal;
mod modes;

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
    // Startup the session in the default conversation mode
    let mode = modes::Mode::DiX;
    let engine = mode.engine();

    let session_id = db::insert_session(&conn, mode.id() as i32, 0); // The default session is always the text session
    println!("{:?}", session_id);
    let mut buffer: String = String::new();

    let history_data: Vec<String> = db::get_full_history(&conn, mode.id() as i32);
    let mut history: HashMap<String, Vec<String>> = HashMap::new(); // Stores tokenized lines
    for result in history_data {
        utilities::tokenize_and_index(&result, &mut history);    
    }

    let _res: Result<(), io::Error> = terminal::listen_key_strokes(&mut buffer, &mut history, &conn, mode.id() as i32);    

    db::update_session(&conn, session_id, 1); // Set the session to inactive

}
