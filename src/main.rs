use std::path::Path;
use std::collections::{HashMap, HashSet};
use std::fmt::write;
use std::io::{self, Write, stdin, stdout};
use clap::builder::Str;
use termion::cursor::DetectCursorPos;
use termion::event::Key;
use termion::clear::{CurrentLine};
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use termion::{screen::{ToAlternateScreen, ToMainScreen}, color};
use rand::seq::SliceRandom;
use rusqlite::{params, Connection};
use tch::{nn::{self, Module, OptimizerConfig, RNN, LSTMState}, Device, Tensor};
mod rnn;
mod db;
mod utilities;

// const ASCII_CHARS: &[char] = &[
//     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b',
//     'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
//     't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '(', ')'
// ];

// #[derive(Debug)]
// struct RnnModel {
//     lstm: LSTM,
//     fc: nn::Linear,
// }

// impl RnnModel {
//     fn new(vs: &nn::Path, input_size: i64, hidden_size: i64, output_size: i64) -> Self {
//         let lstm = nn::lstm(vs, input_size, hidden_size, Default::default());
//         let fc = nn::linear(vs, hidden_size, output_size, Default::default());
//         RnnModel { lstm, fc }
//     }

//     fn forward(&self, input: &Tensor, state: &LSTMState) -> (Tensor, LSTMState) {
//         let (output, new_state) = self.lstm.seq_init(input, state);
//         let output = self.fc.forward(&output);
//         (output, new_state)
//     }
// }




// // fn tokenize_and_index(input: &str, history: &mut HashMap<String, Vec<String>>) {
// //     for i in 1..=input.len() {
// //         // Iterate over slice lengths
// //         let slice = &input[0..i];
// //         let curr_val: &mut Vec<String> = history.entry(slice.to_string()).or_insert(Vec::new());
// //         curr_val.push(input.to_string());
// //     }
// // }

// // fn extract_char_from_key(ky: Key) -> Option<char>
// // {
// //     // We need to handle character that was pressed 
// //     let c0: Option<char> = match ky {
// //         Key::Char(c) => Some(c),
// //         _ => None,
// //     };
// //     c0
// // }

// // fn look_for_cmd_chars(kyc: Option<char>, s: &mut String)
// // {

// //     if let Some(c) = kyc {
// //     (*s).push_str(&c.to_string());
// //     } 
// // }

// // fn find_command_frequency(command: &String, hist: &HashMap<String, Vec<String>>, conn:&rusqlite::Connection) -> i32
// // {
// //     let history_data: Result<Vec<String>, rusqlite::Error> = db::get_full_history(conn);
// //     let commands: Vec<String> = history_data.unwrap();
// //     let count: usize = commands.iter().filter(|&c| c == command).count();
// //     return count as i32;

// // }

fn listen_key_strokes(buf: &mut String, hist: &mut HashMap<String, Vec<String>>, history_ptr: & mut u32, conn:&rusqlite::Connection) -> io::Result<()>
{
    let stdin: io::Stdin = stdin();
    let mut stdout: termion::raw::RawTerminal<io::Stdout> = stdout().into_raw_mode().unwrap();
    stdout.flush().unwrap();


    write!(stdout, ">> ").unwrap();
    stdout.flush()?;

    //detecting keydown events
    for c in stdin.keys() {        
        match c.unwrap() {
            Key::BackTab => {
                write!(stdout, "{}", termion::clear::AfterCursor).unwrap();
                stdout.flush()?;

                // Find the value associated with the key
                if let Some(values) = hist.get(buf) {
                    // Fix the length of history
                    // *history_ptr = unsafe { Box::from_raw(values.len() as u128)};
                    *history_ptr = values.len() as u32-1;
                    write!(stdout, "{}", color::Fg(color::Red)).unwrap();                 
                    // Safely access element
                    if let Some(value) = values.get(*history_ptr as usize) {
                        write!(stdout, "{}", value).unwrap();
                        write!(stdout, "{}", termion::cursor::Left(value.len() as u16)).unwrap();    
                        *history_ptr = *history_ptr - 1;
                    }                  

                    write!(stdout, "{}", color::Fg(color::Reset)).unwrap();
                    stdout.flush()?;
                }                
            },
            Key::Up => {

                // Find the value associated with the key
                if let Some(values) = hist.get(buf) {
                    write!(stdout, "{}", termion::clear::AfterCursor).unwrap();    
                    write!(stdout, "{}", color::Fg(color::Red)).unwrap();                 
                    // Safely access element
                    if let Some(value) = values.get(*history_ptr as usize) {
                        write!(stdout, "{}", value).unwrap();
                        write!(stdout, "{}", termion::cursor::Left(value.len() as u16)).unwrap();    
                        if *history_ptr == 0 {
                            *history_ptr = values.len() as u32-1;
                        }
                        *history_ptr = *history_ptr - 1;
                    }                  

                    write!(stdout, "{}", color::Fg(color::Reset)).unwrap();
                    stdout.flush()?;                
                }

            },
            Key::Backspace => {
                let cur_pos = stdout.cursor_pos().unwrap();       
                write!(stdout, "{}", termion::cursor::Goto(4, cur_pos.1)).unwrap();
                stdout.flush()?;
                (*buf).pop();
                write!(stdout, "{}{}", buf,termion::clear::AfterCursor).unwrap();
                stdout.flush()?;
            },            
            Key::Char('\n') => {
                let cur_pos: (u16, u16) = stdout.cursor_pos().unwrap();                
                write!(stdout, "\n{}>> ", termion::cursor::Goto(1,cur_pos.1 + 1)).unwrap();
                stdout.flush()?;
                utilities::tokenize_and_index(buf, hist);
                let frequency: i32 = utilities::find_command_frequency(buf, hist, conn) ;
                db::insert_command(conn, buf, frequency);
                (*buf).clear();
                write!(stdout, "{}", termion::clear::AfterCursor).unwrap();
                stdout.flush()?;

            },
            Key::Ctrl('q') => {
                write!(stdout, "{}", termion::clear::AfterCursor).unwrap();
                stdout.flush()?;
                break;
            },
            ky => {                       
                // write!(stdout, "{}", ToMainScreen).unwrap();
                // stdout.flush()?;
                let kyc: Option<char> = utilities::extract_char_from_key(ky);
                if let Some(c) = kyc {
                    stdout.write_all(&[c as u8])?;
                    stdout.flush()?;
                    write!(stdout, "{}", termion::clear::AfterCursor).unwrap();
                    stdout.flush()?;
                    }
                    utilities::look_for_cmd_chars(kyc, buf);
            },
        }        
    };

    Ok(())
}


// fn predict_next_char(hidden_size: i64, input_size: i64, batch_size: i64, device: Device, model:&RnnModel, test_tn: Tensor)->char {
    
//     let state = LSTMState((
//         Tensor::zeros(&[1, batch_size, hidden_size], (Kind::Float, device)), // h
//         Tensor::zeros(&[1, batch_size, hidden_size], (Kind::Float, device)), // c
//     ));
//     // Forward pass
//     let (output, _) = model.forward(&test_tn, &state);
//     // println!("{:?}", output.size())
//     let slice = output.get(2).get(0);
//     let probabilities = slice.softmax(-1, Kind::Float);
//     let most_probable_index_tensor = probabilities.argmax(-1, false);
//     let most_probable_index: i64 = most_probable_index_tensor.int64_value(&[]);
//     println!("{:?}", most_probable_index);
//     // println!("The most probable next char is: {:?}", utilities::index_to_char(most_probable_index as usize, ASCII_CHARS.to_vec()));
//     let c = utilities::index_to_char(most_probable_index as usize, ASCII_CHARS.to_vec());
//     c
// }

// fn main() {
//     // First load the history of commands
//     let conn: rusqlite::Connection = db::setup_db();
//     let history_data: Result<Vec<String>, rusqlite::Error> = db::get_full_history(&conn);
//     let mut history_ptr: u32 = 0;
//     let mut buffer: String = String::new();
//     let mut history: HashMap<String, Vec<String>> = HashMap::new(); // Stores tokenized lines
//     for result in history_data.unwrap() {
//         utilities::tokenize_and_index(&result, &mut history);    
//     }
//     // Collect unique characters
//     // let vocab:HashSet<char> = history.values().flat_map(|commands| commands.iter().flat_map(|cmd| cmd.chars())).collect();
//     // let vocab: Vec<char> = vocab.into_iter().collect();
//     // 
//     let device = Device::cuda_if_available();
//     let mut vs = nn::VarStore::new(device);
    
//     // Hyperparameters (reduced for testing)
//     let input_size = ASCII_CHARS.len() as i64; // reasonable size
//     let hidden_size = 34;
//     let output_size = input_size;
//     let seq_len: i64 = 3;    // shorter sequence length
//     let learning_rate: f64 = 0.01;
//     let epochs = 20000;
//     let seq_len2 = seq_len as usize;
//     let model_path = "./my_cli/model.safetensors";
//     let model: RnnModel = RnnModel::new(&vs.root(), input_size, hidden_size, output_size);            
//     let mut optimizer = nn::Adam::default().build(&vs, learning_rate).unwrap();
//     let mut model_trained = false;
//     // Check if the model exists before loading
//     // if Path::new(model_path).exists() {
//     //     println!("Loading existing model...");
//     //     vs.load(model_path);
//     //     model_trained = true;
//     // } else {
//     //     println!("No saved model found, starting fresh.");
//     // }
//     let mut batch_size: i64 = 0;

//     if history.len() > 0 {
//         if !model_trained {
//             let mut training_data: Tensor = Tensor::empty(&[seq_len, 1, input_size], (Kind::Float, device));
//             let mut target_data: Tensor = Tensor::empty(&[seq_len, 1], (Kind::Int64, device));
        
//             for (_, commands) in history.iter() {
//                 for cmd in commands {
//                     let chars: Vec<char> = cmd.chars().collect();
//                     if chars.len() > seq_len2 {
//                         for i in 0..chars.len() - seq_len2 {
//                             let mut training_data_2: Tensor = Tensor::empty(&[1, 1, input_size], (Kind::Float, device));
//                             let mut target_data_2: Tensor = Tensor::empty(&[1, 1], (Kind::Int64, device));
//                             for j in 0..seq_len2 {
//                                 let tn: Tensor = utilities::one_hot_encode_char(chars[i+j], ASCII_CHARS.to_vec(), device).reshape(&[1, 1, input_size]);
//                                 training_data_2 = Tensor::cat(&[training_data_2, tn], 0);
//                                 if let Some(idx) = utilities::char_to_index(chars[i+j+1], ASCII_CHARS.to_vec()) {
                                    
//                                     let tn: Tensor = (idx as i64)*Tensor::ones(&[1, 1], (Kind::Int64, device));                        
//                                     target_data_2 = Tensor::cat(&[target_data_2, tn], 0);
//                                 };
//                             }
//                             // println!("{:?}, {:?}", target_data.size(), target_data_2.size());
//                             training_data = Tensor::cat(&[training_data, training_data_2.slice(0, 1, seq_len+1, 1)], 1);
//                             target_data   = Tensor::cat(&[target_data, target_data_2.slice(0, 1, seq_len+1, 1)], 1);
//                         }
//                     }
//                 }
//             }
//             // target_data.get(1).print();
            
//                 let train_inputs: Tensor = training_data.slice(1, 1, training_data.size()[1], 1);
//                 let train_targets: Tensor = target_data.slice(1, 1, target_data.size()[1], 1);
//                 batch_size = train_inputs.size()[1];  

//                 // println!("{:?}-{:?}", train_inputs.size(), 1);

//                 for epoch in 1..=epochs {
//                 //     // Zero gradients
//                     optimizer.zero_grad();
            
//                     // Initialize LSTM state
//                     let state: LSTMState = LSTMState((
//                         Tensor::zeros(&[1, batch_size, hidden_size], (Kind::Float, device)), // h
//                         Tensor::zeros(&[1, batch_size, hidden_size], (Kind::Float, device)), // c
//                     ));
            
//                     // Forward pass
//                     let (output, _) = model.forward(&train_inputs, &state);
//                     // println!("{:?}--{:?}", train_targets.reshape(&[3*train_targets.size()[1]]).size(),output.view([-1, output_size]).size());
//                     // train_targets.reshape(&[3*train_targets.size()[1]]).get(0).print();
//                     // // Reshape output and compute loss
//                     let loss = output.view([-1, output_size]) // Flatten for softmax
//                         .cross_entropy_for_logits(&train_targets.reshape(&[3*train_targets.size()[1]]));
//                             // &train_targets.reshape(&[3*train_targets.size()[1]]));
            
//                     loss.backward();
//                     optimizer.step();
            
//                     if epoch % 100 == 0 {
//                         println!("Epoch {}: Loss = {:?}", epoch, loss.double_value(&[]));
//                         let test_tn: Tensor = Tensor::cat(&[utilities::one_hot_encode_char('p',ASCII_CHARS.to_vec(), device).reshape(&[1, 1, input_size]), 
//                         utilities::one_hot_encode_char('1',ASCII_CHARS.to_vec(), device).reshape(&[1, 1, input_size]),
//                         utilities::one_hot_encode_char('1',ASCII_CHARS.to_vec(), device).reshape(&[1, 1, input_size])
//                         ], 0).to_kind(Kind::Float);
//                         let c: char = predict_next_char(hidden_size, input_size, batch_size, device, &model, test_tn);
//                         println!("The next char is predicted to be {:?}", c);
//                     }
//                 }


                
//                 println!("Training complete!");    
//         }


//         // print!("{:?}", most_probable_index);
//         // println!("The most probable position is: {:?}", utilities::char_to_index('f', ASCII_CHARS.to_vec()));
//         if !model_trained {
//             vs.save(model_path).unwrap();
//         }
//     }

//     let test_tn: Tensor = Tensor::cat(&[utilities::one_hot_encode_char('p',ASCII_CHARS.to_vec(), device).reshape(&[1, 1, input_size]), 
//     utilities::one_hot_encode_char('1',ASCII_CHARS.to_vec(), device).reshape(&[1, 1, input_size]),
//     utilities::one_hot_encode_char('1',ASCII_CHARS.to_vec(), device).reshape(&[1, 1, input_size])
//     ], 0).to_kind(Kind::Float);
//     let c = predict_next_char(hidden_size, input_size, batch_size, device, &model, test_tn);
//     println!("The next char is predicted to be {:?}", c);

//     let _res: Result<(), io::Error> = listen_key_strokes(&mut buffer, &mut history, &mut history_ptr, &conn);


// }


fn main() {
    let db_path = "my_cli/commands.db"; // Change to your SQLite database path
    let commands = rnn::load_commands(db_path);
    println!("Loaded {:?} commands from the history", commands.len());
    let (char_to_idx, idx_to_char) = rnn::build_vocab(&commands);
    println!("Vocab built with {:?} characters", char_to_idx.len());

    let model = rnn::train_model(1000, 1000, 128, &char_to_idx, &idx_to_char, commands);
    
    
    let start_seq = "is not her";
    let predicted_text = rnn::generate_text(&model, &start_seq, &char_to_idx, &idx_to_char, 1 as usize, 128);
    println!("{:?}", predicted_text);

    // // First load the history of commands
    // let conn: rusqlite::Connection = db::setup_db();
    // let history_data: Result<Vec<String>, rusqlite::Error> = db::get_full_history(&conn);
    // let mut history_ptr: u32 = 0;
    // let mut buffer: String = String::new();
    // let mut history: HashMap<String, Vec<String>> = HashMap::new(); // Stores tokenized lines
    // for result in history_data.unwrap() {
    //     utilities::tokenize_and_index(&result, &mut history);    
    // }

    // let _res: Result<(), io::Error> = listen_key_strokes(&mut buffer, &mut history, &mut history_ptr, &conn);    
}
