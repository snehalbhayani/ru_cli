use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::write;
use std::io::{self, Write, stdin, stdout};
use termion::cursor::DetectCursorPos;
use termion::event::Key;
use termion::clear::{CurrentLine};
use termion::input::TermRead;
use termion::raw::IntoRawMode;
// mod plot;
use termion::{screen::{ToAlternateScreen, ToMainScreen}, color};
mod db;

fn handle_exit(s: &str) -> bool {
    if s.eq_ignore_ascii_case("exit") {
        println!("Exiting...");
        return true;
    } else {
        return false;
    }
}

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
                // write!(stdout, "{}", ToMainScreen).unwrap();
                // stdout.flush()?;
                let cur_pos: (u16, u16) = stdout.cursor_pos().unwrap();                
                write!(stdout, "\n{}>> ", termion::cursor::Goto(1,cur_pos.1 + 1)).unwrap();
                stdout.flush()?;
                tokenize_and_index(buf, hist);
                db::insert_command(conn, buf);
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

fn main() {

    let conn: rusqlite::Connection = db::setup_db();
    let history_data = db::get_full_history(&conn);
    let mut history_ptr: u32 = 0;


    // plot::plot_sample();
    // We need two main types of memory. 
    // 1. buffer : holds the command line buffer
    let mut buffer: String = String::new();
    // 2. Holds the history
    let mut history: HashMap<String, Vec<String>> = HashMap::new(); // Stores tokenized lines
    for result in history_data.unwrap() {
        tokenize_and_index(&result, &mut history);    
    }
    
    let _res: Result<(), io::Error> = listen_key_strokes(&mut buffer, &mut history, &mut history_ptr, &conn);
    
    // // Iterate through keys and print the values
    // for (key, values) in &history {
    //     println!("Key: {}", key);
    //     for value in values {
    //         println!("  Value: {}", value);
    //     }
    // }
   
}
