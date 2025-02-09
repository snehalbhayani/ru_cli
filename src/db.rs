use rusqlite::{params, Connection, Result};
use dirs::config_dir;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn get_db_path() -> String {
    let mut path = config_dir().unwrap();
    path.push("/home/snehal/ru_cli/my_cli");
    fs::create_dir_all(&path).unwrap();
    path.push("history.db");
    path.to_str().unwrap().to_string()
}

pub fn setup_db() -> Connection {
    let db_path = get_db_path();
    let conn = Connection::open(db_path).unwrap();

    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command TEXT NOT NULL,
            timestamp INTEGER NOT NULL, frequency INTEGER NOT NULL
        )",
        [],
    ).unwrap();

    conn
}

pub fn insert_command(conn: &Connection, command: &str, frequency:i32) {
    let timestamp: u64 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    conn.execute(
        "INSERT INTO history (command, timestamp, frequency) VALUES (?1, ?2, ?3)",
        params![command, timestamp, frequency],
    ).unwrap();
}

pub fn update_command(conn: &Connection, command: &str, frequency:i32) {
    // Update the command history by searching based on command
    let timestamp: u64 = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

    conn.execute(
        "INSERT INTO history (command, timestamp, frequency) VALUES (?1, ?2, ?3)",
        params![command, timestamp, frequency],
    ).unwrap();
}

pub fn get_full_history(conn: &Connection) ->  Result<Vec<String>> {
    let mut stmt = conn.prepare("SELECT timestamp, command FROM history ORDER BY timestamp DESC LIMIT 100")?;
    let rows = stmt.query_map([], |row| Ok((row.get::<_, i32>(0)?, row.get::<_, String>(1)?)))?;
    let mut results = Vec::new();

    for row in rows {
        let (_, command) = row?;
        results.push(command);
    }
    Ok(results)
}

pub fn get_last_n_commands(conn: &Connection, n: i32) {
    let mut stmt = conn.prepare("SELECT command FROM history ORDER BY timestamp DESC LIMIT ?1").unwrap();
    let rows = stmt.query_map(params![n], |row| row.get::<_, String>(0)).unwrap();

    println!("Recent Commands:");
    for row in rows {
        println!("{}", row.unwrap());
    }
}

