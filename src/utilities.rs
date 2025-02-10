use std::ops::Index;
use std::path::Path;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write, stdin, stdout};
use termion::cursor::DetectCursorPos;
use termion::event::Key;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use termion::color;
use tch::nn::{self, LSTMState, Module, OptimizerConfig, LSTM, RNN};
use tch::{Device, Kind, Tensor};
use crate::db;

pub fn char_to_index(c: char, vocab:Vec<char>) -> Option<usize> {

    if let Some(index) = vocab.iter().position(|&vc| vc == c) {
        Some(index)
    } else {
        None
    }
}

pub fn index_to_char(index: usize, vocab:Vec<char>) -> char {
    // match index {
    //     0..=25 => (b'a' + index as u8) as char, // Map 0-25 to 'a'-'z'
    //     26..=35 => (b'0' + (index - 26) as u8) as char, // Map 26-35 to '0'-'9'
    //     _ => panic!("Index {} is out of valid range (0-35)", index), // Error for invalid indexes
    // }
    let c = vocab.get(index).unwrap();
    *c
}

pub fn one_hot_encode_char(c: char, vocab:Vec<char>, device: Device) -> Tensor {
    
    let vocab_size: i64 = vocab.len() as i64; // Full ASCII range

    let mut one_hot: Tensor = Tensor::zeros(&[vocab_size], (Kind::Int16, device));
    if let Some(idx) = char_to_index(c, vocab) {
        // Create a one-hot tensor of shape [vocab_size]
        let _ = one_hot.index_put_(&[Some(Tensor::from(idx as i64))], &Tensor::from(1 as i16), false);
        
    };
    one_hot

}

pub fn tokenize_and_index(input: &str, history: &mut HashMap<String, Vec<String>>) {
    for i in 1..=input.len() {
        // Iterate over slice lengths
        let slice = &input[0..i];
        let curr_val: &mut Vec<String> = history.entry(slice.to_string()).or_insert(Vec::new());
        curr_val.push(input.to_string());
    }
}

pub fn extract_char_from_key(ky: Key) -> Option<char>
{
    // We need to handle character that was pressed 
    let c0: Option<char> = match ky {
        Key::Char(c) => Some(c),
        _ => None,
    };
    c0
}

pub fn look_for_cmd_chars(kyc: Option<char>, s: &mut String)
{

    if let Some(c) = kyc {
    (*s).push_str(&c.to_string());
    } 
}

