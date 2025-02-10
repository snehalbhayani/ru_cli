use anyhow::Ok;
use rusqlite::{params, Connection, Result};
use dirs::config_dir;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn get_db_path() -> String {
    // Get and create (if does not exist) the directory to store the database.
    // Simple, we use sqlite3.
    let mut path = config_dir().unwrap();
    path.push("/home/snehal/ru_cli/db");
    fs::create_dir_all(&path).unwrap();
    path.push("ru_cli.db");
    path.to_str().unwrap().to_string()
}

pub fn setup_db() -> Connection {
    // Setup the database connection and create the table(s) which do not exist.
    let db_path = get_db_path();
    let conn = Connection::open(db_path).unwrap();

    // Table 2: Maintain the session information
    conn.execute(
        "CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type INTEGER NOT NULL,
            timestamp INTEGER NOT NULL, 
            state INTEGER DEFAULT 0 NOT NULL
        )",
        [],
    ).unwrap();

    // Table 1: Maintain the command history
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            sess_id INTEGER NOT NULL,
            FOREIGN KEY (sess_id) REFERENCES session(id))",
        [],
    ).unwrap();

    conn
}

pub fn insert_session(conn: &Connection, session_type: i32, session_state: i32)->i32 {
    let timestamp: u64 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    conn.execute(
        "INSERT INTO session (type, timestamp, state) VALUES (?1, ?2, ?3)",
        params![session_type, timestamp, session_state],
    ).unwrap();
    conn.last_insert_rowid() as i32
}

pub fn update_session(conn: &Connection, session_id:i32, session_state: i32) {
    conn.execute(
        "UPDATE session  SET state = ?2 WHERE session.id = ?1",
        params![session_id, session_state],
    ).unwrap();
}

pub fn insert_command(conn: &Connection, command: &str, mode:i32) {
    // Update the command history by searching based on command
    let timestamp: u64 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    let sess_id = conn.query_row(
        "SELECT id FROM session WHERE type = ?1 AND state = 0 LIMIT 1",
        params![mode],
        |row| row.get(0),
    ).unwrap(); // `optional()` to return None if no session is found

    // Ensure we have a valid session ID before proceeding
    let sess_id = match sess_id {
        Some(id) => id,
        None => -1, // Return an empty vector if no session is found
    };
    println!("{:?}", sess_id);

    // Insert the command into history with the retrieved sess_id
    conn.execute(
        "INSERT INTO history (command, timestamp, sess_id) VALUES (?1, ?2, ?3)",
        params![command, timestamp, sess_id],
    ).unwrap();
}

pub fn update_command(conn: &Connection, command: &str, mode:i32) {
    // Update the command history by searching based on command
    let timestamp: u64 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    let sess_id:Option<i32> = conn.query_row(
        "SELECT id FROM session WHERE type = ?1 AND state = 0 LIMIT 1",
        params![mode],
        |row| row.get(0),
    ).unwrap(); // `optional()` to return None if no session is found


    // Ensure we have a valid session ID before proceeding
    let sess_id = match sess_id {
        Some(id) => id,
        None => -1, // Return an empty vector if no session is found
    };
    
     // Insert the command into history with the retrieved sess_id
     conn.execute(
        "INSERT INTO history (command, timestamp, sess_id) VALUES (?1, ?2, ?3)",
        params![command, timestamp, sess_id],
    ).unwrap();

}

// Get the history specific to the current active mode
pub fn get_full_history(conn: &Connection, mode:i32) ->  Vec<String> {

    let sess_id: Option<i32> = conn.query_row(
        "SELECT id FROM session WHERE type = ?1 AND state = 0 LIMIT 1",
        params![mode],
        |row| row.get(0),
    ).unwrap(); // `optional()` to return None if no session is found


    // Ensure we have a valid session ID before proceeding
    let sess_id = match sess_id {
        Some(id) => id,
        None => -1, // Return an empty vector if no session is found
    };
    
    
    let mut stmt = conn.prepare("SELECT command FROM history WHERE sess_id=?1 ORDER BY timestamp DESC LIMIT 100").unwrap();
    let rows = stmt.query_map([sess_id], |row| row.get::<_, String>(0));
    let mut results = Vec::new();

    for row in rows.unwrap() {
        let command = row.unwrap();
        results.push(command);
    }
    results
}



