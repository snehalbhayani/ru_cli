use rusqlite::{params, Connection, Result};


#[derive(Debug)]
pub enum Mode {
    Math,
    Nix,
    DiX,
    Draw
}

impl Mode {
    /// Returns a unique ID for each mode
    pub fn id(&self) -> u8 {
        match self {
            Mode::Math => 1,
            Mode::Nix => 2,
            Mode::DiX => 3,
            Mode::Draw => 4,
        }
    }

    /// Returns the corresponding engine for each mode
    pub fn engine(&self) -> &'static str {
        match self {
            Mode::Math => "MathEngine",
            Mode::Nix => "NixShell",
            Mode::DiX => "ChatEngine",
            Mode::Draw => "GraphicsRenderer",
        }
    }

    pub fn last_command(&self, conn: &Connection) -> Result<Vec<String>> {
        // Returns the last command executed in the current active mode
        let mode: i32 = self.id() as i32; // Convert mode to i32

        // Prepare the SQL statement
        let mut stmt = conn.prepare(
            "SELECT command FROM history WHERE mode = ?1 AND state = 0 ORDER BY timestamp DESC LIMIT 2"
        )?;

        // Query the database and collect results
        let rows = stmt.query_map([mode], |row| row.get::<_, String>(0))?;

        let results: Vec<String> = rows.collect::<Result<Vec<String>>>()?; // Collect and handle errors properly

        Ok(results) // Return results inside Ok
    }

}