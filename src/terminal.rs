use std::collections::{HashMap, HashSet};
use std::io::{self, Write, stdin, stdout};
use termion::cursor::{DetectCursorPos, Goto};
use termion::cursor;
use termion::event::Key;
use termion::clear;
use termion::input::TermRead;
use termion::raw::IntoRawMode;
use termion::{screen::{ToAlternateScreen, ToMainScreen}, color, terminal_size};
use crate::utilities;
use crate::db;


pub fn print_welcome_message() {
    // Get stdout and set it to raw mode
    let mut stdout = io::stdout().into_raw_mode().unwrap();
    
    // Get terminal size (width, height)
    let (width, _height) = terminal_size().unwrap_or((80, 24)); // Default to 80x24 if detection fails

    // Calculate center position for "WELCOME"
    let center_x = (width / 2) - ("WELCOME".len() as u16 / 2);

    // Clear the screen
    write!(stdout, "{}{}", clear::All, cursor::Goto(1, 1)).unwrap();

    // Print the welcome message in yellow
    write!(stdout, "{}{}", color::Fg(color::Yellow), cursor::Goto(1, 1)).unwrap();
    for _ in 0..width {
        write!(stdout, "=").unwrap();
    }
    write!(stdout, "{}WELCOME", cursor::Goto(center_x, 2)).unwrap();
    write!(stdout, "\n{}", cursor::Goto(1, 3)).unwrap();

    for _ in 0..width {
        write!(stdout, "=").unwrap();
    }

    write!(stdout, "{}", cursor::Goto(1, 3)).unwrap();
    
    write!(stdout, "{}This is version Rust CLI 1.0.", cursor::Goto(1, 5)).unwrap();
    write!(stdout, "{}Use the command `help` to find more about this tool does.", cursor::Goto(1, 6)).unwrap();
    // Reset color to default
    write!(stdout, "{}", color::Fg(color::Reset)).unwrap();
    write!(stdout, "{}>> ", cursor::Goto(1, 7)).unwrap();

    // Flush to ensure the output is displayed immediately
    stdout.flush().unwrap();
}


pub fn into_raw_mode() -> termion::raw::RawTerminal<io::Stdout>{
    let mut stdout: termion::raw::RawTerminal<io::Stdout> = stdout().into_raw_mode().unwrap();
    stdout.flush().unwrap();
    stdout
}

pub fn listen_key_strokes(buf: &mut String, hist: &mut HashMap<String, Vec<String>>, conn:&rusqlite::Connection) -> io::Result<()>
{
    // Capture every possible keyboard event. 
    // This is where, we listen to all possible events and decide what to do and when to do.
    let stdin: io::Stdin = stdin();
    let mut stdout: termion::raw::RawTerminal<io::Stdout> = into_raw_mode();
    

    // Print welcome message
    print_welcome_message();
    stdout.flush()?;

    //detecting keydown events
    for c in stdin.keys() {        
        match c.unwrap() {
            Key::BackTab => {
                write!(stdout, "{}", termion::clear::AfterCursor).unwrap();
                stdout.flush()?;

                // Find the value associated with the key
                // if let Some(values) = hist.get(buf) {
                //     write!(stdout, "{}", color::Fg(color::Red)).unwrap();                 
                //     if let Some(value) = values.get(*history_ptr as usize) {
                //         write!(stdout, "{}", value).unwrap();
                //         write!(stdout, "{}", termion::cursor::Left(value.len() as u16)).unwrap();    
                //         *history_ptr = *history_ptr - 1;
                //     }                  

                //     write!(stdout, "{}", color::Fg(color::Reset)).unwrap();
                //     stdout.flush()?;
                // }                
            },
            Key::Up => {

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