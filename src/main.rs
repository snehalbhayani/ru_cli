use std::path::Path;
use std::collections::HashMap;
use std::fmt::write;
use std::io::{self, Write, stdin, stdout};
use termion::cursor::DetectCursorPos;
use termion::event::Key;
use termion::clear::{CurrentLine};
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use termion::{screen::{ToAlternateScreen, ToMainScreen}, color};
use tch::nn::{self, LSTMState, Module, OptimizerConfig, LSTM, RNN};
use tch::{Device, Kind, Tensor};
mod rnn;
mod db;



fn tokenize_and_index(input: &str, history: &mut HashMap<String, Vec<String>>) {
    for i in 1..=input.len() {
        // Iterate over slice lengths
        let slice = &input[0..i];
        let curr_val: &mut Vec<String> = history.entry(slice.to_string()).or_insert(Vec::new());
        curr_val.push(input.to_string());
    }
}

fn extract_char_from_key(ky: Key) -> Option<char>
{
    // We need to handle character that was pressed 
    let c0: Option<char> = match ky {
        Key::Char(c) => Some(c),
        _ => None,
    };
    c0
}

fn look_for_cmd_chars(kyc: Option<char>, s: &mut String)
{

    if let Some(c) = kyc {
    (*s).push_str(&c.to_string());
    } 
}

fn find_command_frequency(command: &String, hist: &HashMap<String, Vec<String>>, conn:&rusqlite::Connection) -> i32
{
    let history_data: Result<Vec<String>, rusqlite::Error> = db::get_full_history(conn);
    let commands: Vec<String> = history_data.unwrap();
    let count: usize = commands.iter().filter(|&c| c == command).count();
    return count as i32;

}

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
                tokenize_and_index(buf, hist);
                let frequency: i32 = find_command_frequency(buf, hist, conn) ;
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
                let kyc: Option<char> = extract_char_from_key(ky);
                if let Some(c) = kyc {
                    stdout.write_all(&[c as u8])?;
                    stdout.flush()?;
                    write!(stdout, "{}", termion::clear::AfterCursor).unwrap();
                    stdout.flush()?;
                    }
                look_for_cmd_chars(kyc, buf);
            },
        }        
    };

    Ok(())
}

// Generate (input, next char) pairs for training
fn generate_training_data(history: &HashMap<String, Vec<String>>, seq_len: usize) -> (Vec<Vec<char>>, Vec<char>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for (_, commands) in history.iter() {
        for cmd in commands {
            let chars: Vec<char> = cmd.chars().collect();
            if chars.len() > seq_len {
                for i in 0..chars.len() - seq_len {
                    inputs.push(chars[i..i + seq_len].to_vec());
                    targets.push(chars[i + seq_len]);
                }
            }
        }
    }

    (inputs, targets)
}


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


fn char_to_index(c: char) -> Option<usize> {
    if c.is_ascii_alphabetic() {
        // Normalize to lowercase and map 'a' to 0, 'b' to 1, ..., 'z' to 25.
        Some(c.to_ascii_lowercase() as usize - 'a' as usize)
    } else if c.is_ascii_digit() {
        // Map '0' to 26, '1' to 27, ..., '9' to 35.
        Some(26 + (c as usize - '0' as usize))
    } else {
        None
    }
}

fn index_to_char(index: usize) -> char {
    match index {
        0..=25 => (b'a' + index as u8) as char, // Map 0-25 to 'a'-'z'
        26..=35 => (b'0' + (index - 26) as u8) as char, // Map 26-35 to '0'-'9'
        _ => panic!("Index {} is out of valid range (0-35)", index), // Error for invalid indexes
    }
}

fn one_hot_encode_char(c: char, device: Device) -> Tensor {
    let vocab_size: i64 = 36; // Full ASCII range

    let mut one_hot: Tensor = Tensor::zeros(&[vocab_size], (Kind::Int16, device));
    if let Some(idx) = char_to_index(c) {
        // Create a one-hot tensor of shape [vocab_size]
        one_hot.index_put_(&[Some(Tensor::from(idx as i64))], &Tensor::from(1 as i16), false);
        
    };
    one_hot

}

fn main() {
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);
    
    // Hyperparameters (reduced for testing)
    let input_size = 36; // reasonable size
    let hidden_size = 40;
    let output_size = 36;
    let batch_size: i64 = 40;  // reduced batch size
    let seq_len: i64 = 2;    // shorter sequence length
    let learning_rate: f64 = 0.01;
    let epochs = 5000;
    let conn: rusqlite::Connection = db::setup_db();
    let history_data: Result<Vec<String>, rusqlite::Error> = db::get_full_history(&conn);
    let mut history_ptr: u32 = 0;
    let seq_len2 = seq_len as usize;
    let mut buffer: String = String::new();
    let mut history: HashMap<String, Vec<String>> = HashMap::new(); // Stores tokenized lines
    let model_path = "./my_cli/model.pt";
    let model: RnnModel = RnnModel::new(&vs.root(), input_size, hidden_size, output_size);            
    let mut optimizer = nn::Adam::default().build(&vs, learning_rate).unwrap();
    let mut model_trained = false;
    // Check if the model exists before loading
    if Path::new(model_path).exists() {
        println!("Loading existing model...");
        vs.load(model_path);
        model_trained = true;
    } else {
        println!("No saved model found, starting fresh.");
    }

    for result in history_data.unwrap() {
        tokenize_and_index(&result, &mut history);    
    }

    if (!model_trained) {
        let mut training_data: Tensor = Tensor::empty(&[seq_len, 1, 36], (Kind::Float, device));
        let mut target_data: Tensor = Tensor::empty(&[seq_len, 1], (Kind::Int64, device));
    
        for (_, commands) in history.iter() {
            for cmd in commands {
                let chars: Vec<char> = cmd.chars().collect();
                if chars.len() > seq_len2 {
                    for i in 0..chars.len() - seq_len2 {
                        let mut training_data_2: Tensor = Tensor::empty(&[1, 1, 36], (Kind::Float, device));
                        let mut target_data_2: Tensor = Tensor::empty(&[1, 1], (Kind::Int64, device));
                        for j in 0..seq_len2 {
                            let tn: Tensor = one_hot_encode_char(chars[i+j], device).reshape(&[1, 1, 36]);
                            training_data_2 = Tensor::cat(&[training_data_2, tn], 0);
                            if let Some(idx) = char_to_index(chars[i+j+1]) {
                                let tn: Tensor = (idx as i64)*Tensor::ones(&[1, 1], (Kind::Int64, device));                        
                                target_data_2 = Tensor::cat(&[target_data_2, tn], 0);
                            };
                        }
                        training_data = Tensor::cat(&[training_data, training_data_2.slice(0, 1, 3, 1)], 1);
                        target_data   = Tensor::cat(&[target_data, target_data_2.slice(0, 1, 3, 1)], 1);
                    }
                }
            }
        }
        // target_data.get(1).print();
        println!("{:?}", training_data.size());
    
        let train_inputs: Tensor = training_data.slice(1, 1, training_data.size()[1], 1);
        let train_targets: Tensor = target_data.slice(1, 1, target_data.size()[1], 1);
        // println!("{:?}--{:?}", train_inputs.size(),train_targets.size());
        
    
        for epoch in 1..=epochs {
        //     // Zero gradients
            optimizer.zero_grad();
    
            // Initialize LSTM state
            let state = LSTMState((
                Tensor::zeros(&[1, batch_size, hidden_size], (Kind::Float, device)), // h
                Tensor::zeros(&[1, batch_size, hidden_size], (Kind::Float, device)), // c
            ));
    
            // Forward pass
            let (output, _) = model.forward(&train_inputs, &state);
            // println!("{:?}--{:?}", train_targets.reshape(&[2*494]), output.view([-1, output_size]).size() );        
            // // Reshape output and compute loss
            let loss = output.view([-1, output_size]) // Flatten for softmax
                .cross_entropy_for_logits(&train_targets.reshape(&[2*train_targets.size()[1]]));
    
        // //     // Backpropagation
            loss.backward();
            optimizer.step();
    
            // Print loss
            if epoch % 100 == 0 {
                println!("Epoch {}: Loss = {:?}", epoch, loss.double_value(&[]));
            }
        }
    
        println!("Training complete!");    
    }

    let test_tn: Tensor = Tensor::cat(&[one_hot_encode_char('4', device).reshape(&[1, 1, 36]), one_hot_encode_char('5', device).reshape(&[1, 1, 36])], 0).to_kind(Kind::Float);
    println!("{:?}", test_tn);
    let state = LSTMState((
        Tensor::zeros(&[1, 2, hidden_size], (Kind::Float, device)), // h
        Tensor::zeros(&[1, 2, hidden_size], (Kind::Float, device)), // c
    ));

    // Forward pass
    let (output, _) = model.forward(&test_tn, &state);
    let predicted_index = output.argmax(2, true);
    predicted_index.print();
    
    let slice = output.get(1).get(0);
    let probabilities = slice.softmax(-1, Kind::Float);
    let most_probable_index_tensor = probabilities.argmax(-1, false);
    let most_probable_index: i64 = most_probable_index_tensor.int64_value(&[]);
    println!("The most probable position is: {:?}", index_to_char(most_probable_index as usize));
    print!("{:?}", most_probable_index);
    println!("The most probable position is: {:?}", char_to_index('f'));


    let _res: Result<(), io::Error> = listen_key_strokes(&mut buffer, &mut history, &mut history_ptr, &conn);

    if (!model_trained) {
        vs.save(model_path).unwrap();
    }

}
