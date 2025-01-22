use std::collections::HashMap;
use std::collections::HashSet;
use std::io::{self, Write, stdin, stdout};
use std::ops::Deref;
use termion::clear;
use termion::cursor::DetectCursorPos;
use termion::event::Key;
use termion::clear::{CurrentLine};
use termion::cursor;
use termion::input::TermRead;
use termion::raw::IntoRawMode;

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
        let curr_val = history.entry(slice.to_string()).or_insert(HashSet::new());
        curr_val.insert(input.to_string());
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
fn listen_key_strokes(buf: &mut String, hist: &mut HashMap<String, HashSet<String>>) -> io::Result<()>
{

    let stdin: io::Stdin = stdin();
    let mut stdout: termion::raw::RawTerminal<io::Stdout> = stdout().into_raw_mode().unwrap();
    stdout.flush().unwrap();

    //detecting keydown events
    for c in stdin.keys() {        
        match c.unwrap() {
            Key::Up => {
                println!("Fetching history...");
            },
            Key::Backspace => {
                write!(stdout, "{}", termion::cursor::Left(1)).unwrap();
                stdout.flush()?;
                (*buf).pop();
            },            
            Key::Char('\n') => {
                let cur_pos = stdout.cursor_pos().unwrap();                
                
                write!(stdout, "\n{}", termion::cursor::Goto(1,cur_pos.1 + 1)).unwrap();
                stdout.flush()?;
                tokenize_and_index(buf, hist);

            },
            Key::Ctrl('q') => {
                break;
            },
            ky => {                       
                let kyc: Option<char> = extract_char_from_key(ky);
                if let Some(c) = kyc {
                    stdout.write_all(&[c as u8])?;
                    stdout.flush()?;
                }
                look_for_cmd_chars(kyc, buf);
            },
        }        
    };

    Ok(())
}

fn main() {

    // We need two main types of memory. 
    // 1. buffer : holds the command line buffer
    let mut buffer: String = String::new();
    // 2. Holds the history
    let mut history: HashMap<String, HashSet<String>> = HashMap::new(); // Stores tokenized lines
    
    let _res: Result<(), io::Error> = listen_key_strokes(&mut buffer, &mut history);

    println!("... {:?}", buffer);
   
}
