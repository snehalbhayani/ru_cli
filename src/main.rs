// use std::collections::HashMap;
// use std::collections::HashSet;
// use std::fmt::write;
// use std::io::{self, Write, stdin, stdout};
// use rnn::train_model;
// use termion::cursor::DetectCursorPos;
// use termion::event::Key;
// use termion::clear::{CurrentLine};
// use termion::input::TermRead;
// use termion::raw::IntoRawMode;
// mod rnn;
// use termion::{screen::{ToAlternateScreen, ToMainScreen}, color};
// mod db;

// fn handle_exit(s: &str) -> bool {
//     if s.eq_ignore_ascii_case("exit") {
//         println!("Exiting...");
//         return true;
//     } else {
//         return false;
//     }
// }

// fn tokenize_and_index(input: &str, history: &mut HashMap<String, Vec<String>>) {
//     for i in 1..=input.len() {
//         // Iterate over slice lengths
//         let slice = &input[0..i];
//         let curr_val: &mut Vec<String> = history.entry(slice.to_string()).or_insert(Vec::new());
//         curr_val.push(input.to_string());
//     }
// }

// fn extract_char_from_key(ky: Key) -> Option<char>
// {
//     // We need to handle character that was pressed 
//     let c0: Option<char> = match ky {
//         Key::Char(c) => Some(c),
//         _ => None,
//     };
//     c0
// }

// fn look_for_cmd_chars(kyc: Option<char>, s: &mut String)
// {

//     if let Some(c) = kyc {
//     (*s).push_str(&c.to_string());
//     } 
// }

// fn find_command_frequency(command: &String, hist: &HashMap<String, Vec<String>>, conn:&rusqlite::Connection) -> i32
// {
//     let history_data: Result<Vec<String>, rusqlite::Error> = db::get_full_history(conn);
//     let commands: Vec<String> = history_data.unwrap();
//     let count: usize = commands.iter().filter(|&c| c == command).count();
//     return count as i32;

// }

// fn listen_key_strokes(buf: &mut String, hist: &mut HashMap<String, Vec<String>>, history_ptr: & mut u32, conn:&rusqlite::Connection) -> io::Result<()>
// {
//     let stdin: io::Stdin = stdin();
//     let mut stdout: termion::raw::RawTerminal<io::Stdout> = stdout().into_raw_mode().unwrap();
//     stdout.flush().unwrap();


//     write!(stdout, ">> ").unwrap();
//     stdout.flush()?;

//     //detecting keydown events
//     for c in stdin.keys() {        
//         match c.unwrap() {
//             Key::BackTab => {
//                 write!(stdout, "{}", termion::clear::AfterCursor).unwrap();
//                 stdout.flush()?;

//                 // Find the value associated with the key
//                 if let Some(values) = hist.get(buf) {
//                     // Fix the length of history
//                     // *history_ptr = unsafe { Box::from_raw(values.len() as u128)};
//                     *history_ptr = values.len() as u32-1;
//                     write!(stdout, "{}", color::Fg(color::Red)).unwrap();                 
//                     // Safely access element
//                     if let Some(value) = values.get(*history_ptr as usize) {
//                         write!(stdout, "{}", value).unwrap();
//                         write!(stdout, "{}", termion::cursor::Left(value.len() as u16)).unwrap();    
//                         *history_ptr = *history_ptr - 1;
//                     }                  

//                     write!(stdout, "{}", color::Fg(color::Reset)).unwrap();
//                     stdout.flush()?;
//                 }                
//             },
//             Key::Up => {

//                 // Find the value associated with the key
//                 if let Some(values) = hist.get(buf) {
//                     write!(stdout, "{}", termion::clear::AfterCursor).unwrap();    
//                     write!(stdout, "{}", color::Fg(color::Red)).unwrap();                 
//                     // Safely access element
//                     if let Some(value) = values.get(*history_ptr as usize) {
//                         write!(stdout, "{}", value).unwrap();
//                         write!(stdout, "{}", termion::cursor::Left(value.len() as u16)).unwrap();    
//                         if *history_ptr == 0 {
//                             *history_ptr = values.len() as u32-1;
//                         }
//                         *history_ptr = *history_ptr - 1;
//                     }                  

//                     write!(stdout, "{}", color::Fg(color::Reset)).unwrap();
//                     stdout.flush()?;                
//                 }

//             },
//             Key::Backspace => {
//                 let cur_pos = stdout.cursor_pos().unwrap();       
//                 write!(stdout, "{}", termion::cursor::Goto(4, cur_pos.1)).unwrap();
//                 stdout.flush()?;
//                 (*buf).pop();
//                 write!(stdout, "{}{}", buf,termion::clear::AfterCursor).unwrap();
//                 stdout.flush()?;
//             },            
//             Key::Char('\n') => {
//                 let cur_pos: (u16, u16) = stdout.cursor_pos().unwrap();                
//                 write!(stdout, "\n{}>> ", termion::cursor::Goto(1,cur_pos.1 + 1)).unwrap();
//                 stdout.flush()?;
//                 tokenize_and_index(buf, hist);
//                 let frequency: i32 = find_command_frequency(buf, hist, conn) ;
//                 db::insert_command(conn, buf, frequency);
//                 (*buf).clear();
//                 write!(stdout, "{}", termion::clear::AfterCursor).unwrap();
//                 stdout.flush()?;

//             },
//             Key::Ctrl('q') => {
//                 write!(stdout, "{}", termion::clear::AfterCursor).unwrap();
//                 stdout.flush()?;
//                 break;
//             },
//             ky => {                       
//                 // write!(stdout, "{}", ToMainScreen).unwrap();
//                 // stdout.flush()?;
//                 let kyc: Option<char> = extract_char_from_key(ky);
//                 if let Some(c) = kyc {
//                     stdout.write_all(&[c as u8])?;
//                     stdout.flush()?;
//                     write!(stdout, "{}", termion::clear::AfterCursor).unwrap();
//                     stdout.flush()?;
//                     }
//                 look_for_cmd_chars(kyc, buf);
//             },
//         }        
//     };

//     Ok(())
// }

// // Generate (input, next char) pairs for training
// fn generate_training_data(history: &HashMap<String, Vec<String>>, seq_len: usize) -> (Vec<Vec<char>>, Vec<char>) {
//     let mut inputs = Vec::new();
//     let mut targets = Vec::new();

//     for (_, commands) in history.iter() {
//         for cmd in commands {
//             let chars: Vec<char> = cmd.chars().collect();
//             if chars.len() > seq_len {
//                 for i in 0..chars.len() - seq_len {
//                     inputs.push(chars[i..i + seq_len].to_vec());
//                     targets.push(chars[i + seq_len]);
//                 }
//             }
//         }
//     }

//     (inputs, targets)
// }

// fn main() {

//     let conn: rusqlite::Connection = db::setup_db();
//     let history_data: Result<Vec<String>, rusqlite::Error> = db::get_full_history(&conn);
//     let mut history_ptr: u32 = 0;
//     let seq_len: usize = 3;

//     // plot::plot_sample();
//     // We need two main types of memory. 
//     // 1. buffer : holds the command line buffer
//     let mut buffer: String = String::new();
//     // 2. Holds the history
//     let mut history: HashMap<String, Vec<String>> = HashMap::new(); // Stores tokenized lines
//     for result in history_data.unwrap() {
//         tokenize_and_index(&result, &mut history);    
//     }
//     let (train_inputs, train_targets) = generate_training_data(&history, seq_len);
//     println!("{:?}", train_inputs);
//     train_model(history.clone(), 10, seq_len, 2, 5, train_inputs, train_targets);

//     let _res: Result<(), io::Error> = listen_key_strokes(&mut buffer, &mut history, &mut history_ptr, &conn);
    
//     // // Iterate through keys and print the values
//     // for (key, values) in &history {
//     //     println!("Key: {}", key);
//     //     for value in values {
//     //         println!("  Value: {}", value);
//     //     }
//     // }
   
// }

// use std::collections::HashMap;
// use std::collections::HashSet;
// use std::fmt::write;
// use std::io::{self, Write, stdin, stdout};
// use rnn::train_model;
// mod rnn;
// mod db;


// fn tokenize_and_index(input: &str, history: &mut HashMap<String, Vec<String>>) {
//     for i in 1..=input.len() {
//         // Iterate over slice lengths
//         let slice = &input[0..i];
//         let curr_val: &mut Vec<String> = history.entry(slice.to_string()).or_insert(Vec::new());
//         curr_val.push(input.to_string());
//     }
// }

// // Generate (input, next char) pairs for training
// fn generate_training_data(history: &HashMap<String, Vec<String>>, seq_len: usize) -> (Vec<Vec<char>>, Vec<char>) {
//     let mut inputs = Vec::new();
//     let mut targets = Vec::new();

//     for (_, commands) in history.iter() {
//         for cmd in commands {
//             let chars: Vec<char> = cmd.chars().collect();
//             if chars.len() > seq_len {
//                 for i in 0..chars.len() - seq_len {
//                     inputs.push(chars[i..i + seq_len].to_vec());
//                     targets.push(chars[i + seq_len]);
//                 }
//             }
//         }
//     }

//     (inputs, targets)
// }


// fn main() {

//     let conn: rusqlite::Connection = db::setup_db();
//     let history_data: Result<Vec<String>, rusqlite::Error> = db::get_full_history(&conn);
//     let mut history_ptr: u32 = 0;
//     let seq_len: usize = 3;

//     // plot::plot_sample();
//     // We need two main types of memory. 
//     // 1. buffer : holds the command line buffer
//     let mut buffer: String = String::new();
//     // 2. Holds the history
//     let mut history: HashMap<String, Vec<String>> = HashMap::new(); // Stores tokenized lines
//     for result in history_data.unwrap() {
//         tokenize_and_index(&result, &mut history);    
//     }
//     let (train_inputs, train_targets) = generate_training_data(&history, seq_len);
//     println!("{:?} {:?}", train_inputs, train_targets);
//     train_model(history.clone(), 10, seq_len, 2, 5, train_inputs, train_targets);

// }


// rnn.rs

use tch::nn::{self, LSTMState, Module, OptimizerConfig, LSTM, RNN};
use tch::{Device, Kind, Tensor};

#[derive(Debug)]
struct RnnModel {
    lstm: LSTM,
    fc: nn::Linear,
}

impl RnnModel {
    fn new(vs: &nn::Path, input_size: i64, hidden_size: i64, output_size: i64) -> Self {
        let lstm = nn::lstm(vs, input_size, hidden_size, Default::default());
        let fc = nn::linear(vs, hidden_size, output_size, Default::default());
        RnnModel { lstm, fc }
    }

    fn forward(&self, input: &Tensor, state: &LSTMState) -> (Tensor, LSTMState) {
        let (output, new_state) = self.lstm.seq_init(input, state);
        let output = self.fc.forward(&output);
        (output, new_state)
    }
}

pub fn predict(test_input: Tensor, model: &RnnModel, device: Device) -> Tensor {
    // Initialize LSTM state (h and c) with zeros for each batch
    let state = LSTMState((
        Tensor::zeros(&[1, test_input.size()[1], 20], (Kind::Float, device)), // h (batch_size, seq_len, hidden_size)
        Tensor::zeros(&[1, test_input.size()[1], 20], (Kind::Float, device)), // c (batch_size, seq_len, hidden_size)
    ));

    // Forward pass to get the model output
    let output: (Tensor, LSTMState) = model.forward(&test_input, &state);

    output.0
}
fn main() {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    
    // Hyperparameters (reduced for testing)
    let input_size = 10; // reasonable size
    let hidden_size = 20;
    let output_size = 10;
    let batch_size = 2;  // reduced batch size
    let seq_len = 2;    // shorter sequence length
    let learning_rate = 0.01;
    let epochs = 1000;

    let model = RnnModel::new(&vs.root(), input_size, hidden_size, output_size);
    let mut optimizer = nn::Adam::default().build(&vs, learning_rate).unwrap();

    // Generate dummy training data (with smaller sizes)
    let train_inputs = Tensor::randn(&[seq_len, batch_size, input_size], (Kind::Float, device));
    let train_targets = Tensor::randint(output_size, &[seq_len, batch_size], (Kind::Int64, device));

    for epoch in 1..=epochs {
        // Zero gradients
        optimizer.zero_grad();

        // Initialize LSTM state
        let state = LSTMState((
            Tensor::zeros(&[1, batch_size, hidden_size], (Kind::Float, device)), // h
            Tensor::zeros(&[1, batch_size, hidden_size], (Kind::Float, device)), // c
        ));

        // Forward pass
        let (output, _) = model.forward(&train_inputs, &state);
        
        // Reshape output and compute loss
        let loss = output.view([-1, output_size]) // Flatten for softmax
            .cross_entropy_for_logits(&train_targets.view([-1]));

        // Backpropagation
        loss.backward();
        optimizer.step();

        // Print loss
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:?}", epoch, loss.double_value(&[]));
        }
    }

    println!("Training complete!");

    let test_input: Tensor = Tensor::randn(&[seq_len, batch_size, input_size], (Kind::Float, device)); // Random test input

    // Make a prediction
    let prediction = predict(test_input, &model, device);
    println!("Predicted output: {:?}", prediction);

}
