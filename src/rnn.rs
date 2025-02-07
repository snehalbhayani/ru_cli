use std::collections::{HashMap, HashSet};
use rusqlite::{params, Connection};
use tch::{nn::{self, Module, OptimizerConfig, RNN, LSTMState}, Device, Tensor};
const SEQ_LEN: usize = 10; // Fixed sequence length

// 1️⃣ Load commands from SQLite
pub fn load_commands(db_path: &str) -> Vec<String> {
    let conn = Connection::open(db_path).expect("Failed to open DB");
    let mut stmt = conn.prepare("SELECT command FROM commands LIMIT 500").expect("Failed to prepare query");
    let rows = stmt
        .query_map([], |row| row.get(0))
        .expect("Failed to execute query");

    rows.filter_map(Result::ok).collect()
    
}


// 2️⃣ Build character vocabulary
pub fn build_vocab(commands: &[String]) -> (HashMap<char, i64>, HashMap<i64, char>) {
    let mut chars: Vec<char> = commands.iter().flat_map(|s| s.chars()).collect();
    chars.sort_unstable();
    chars.dedup();

    let mut char_to_idx = HashMap::new();
    let mut idx_to_char = HashMap::new();

    char_to_idx.insert('<', 0); // Padding token
    for (i, &c) in chars.iter().enumerate() {
        char_to_idx.insert(c, (i + 1) as i64);
        idx_to_char.insert((i + 1) as i64, c);
    }

    (char_to_idx, idx_to_char)
}

// 3️⃣ Convert commands into training data (input-output sequences)
fn create_sequences(commands: &[String], char_to_idx: &HashMap<char, i64>) -> (Tensor, Tensor) {
    let mut input_data = Vec::new();
    let mut target_data = Vec::new();

    for command in commands {
        let encoded: Vec<i64> = command.chars().filter_map(|c| char_to_idx.get(&c).copied()).collect();
        if encoded.len() < SEQ_LEN + 1 {
            continue;
        }

        for i in 0..(encoded.len() - SEQ_LEN) {
            input_data.push(encoded[i..i + SEQ_LEN].to_vec());
            target_data.push(encoded[i + 1..i + SEQ_LEN + 1].to_vec());
        }
    }

    let input_tensor = Tensor::from_slice(&input_data.concat())
    .view([-1, SEQ_LEN as i64])
    .to_device(Device::cuda_if_available());
    
    let target_tensor: Tensor = Tensor::from_slice(&target_data.concat())
        .view([-1, SEQ_LEN as i64])
        .to_device(Device::cuda_if_available());

    (input_tensor, target_tensor)
}

// 4️⃣ Define LSTM Model
#[derive(Debug)]
pub struct LSTMModel {
    embedding: nn::Embedding,
    lstm: nn::LSTM,
    fc: nn::Linear,
}

impl LSTMModel {
    fn new(vs: &nn::Path, vocab_size: i64, embedding_dim: i64, hidden_size: i64, num_layers: i64) -> Self {
        let embedding = nn::embedding(vs, vocab_size, embedding_dim, Default::default());
        let lstm = nn::lstm(vs, embedding_dim, hidden_size, Default::default());
        let fc = nn::linear(vs, hidden_size, vocab_size, Default::default());

        LSTMModel { embedding, lstm, fc }
    }

    fn forward(&self, input: &Tensor, h: Tensor, c: Tensor) -> (Tensor, LSTMState) {
        let embedded = self.embedding.forward(input);
        // Initialize LSTM state
        let state: LSTMState = LSTMState((h,c));

        let (output, state_new) = self.lstm.seq_init(&embedded, &state);
        let logits = self.fc.forward(&output);
        (logits, state_new)
    }
}

// 5️⃣ Training Function
pub fn  train_model(num_epochs: usize, batch_size: i64, hidden_size:i64, char_to_idx: &HashMap<char, i64>, idx_to_char:&HashMap<i64, char>, commands: Vec<String>)->LSTMModel {
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);


    let vocab_size = char_to_idx.len() as i64;

    let (train_data, target_data) = create_sequences(&commands, &char_to_idx);
    println!("{:?} sequences creaed", train_data.size());
    let train_data = train_data.to_device(device);
    let target_data = target_data.to_device(device);

    let model = LSTMModel::new(&vs.root(), vocab_size, 16, hidden_size, 1);
    let mut opt = nn::Adam::default().build(&vs, 0.01).unwrap();
    // let criterion = nn::CrossEntropyLoss::new();

    let num_batches = train_data.size()[0] / batch_size;

    for epoch in 1..=num_epochs {
        let mut total_loss = 0.0;

        for i in 0..num_batches {
            let start = i * batch_size;
            let end = start + batch_size;

            let input_batch = train_data.narrow(0, start, batch_size).to_device(device);
            let target_batch = target_data.narrow(0, start, batch_size).to_device(device);

            let h = Tensor::zeros(&[1, batch_size, hidden_size], (tch::Kind::Float, device));
            let c = Tensor::zeros(&[1, batch_size, hidden_size], (tch::Kind::Float, device));

            opt.zero_grad();
            let (logits, _) = model.forward(&input_batch, h, c);
            let loss = logits.view([-1, vocab_size]).cross_entropy_for_logits(&target_batch.view([-1]));
            loss.backward();
            opt.step();

            total_loss += loss.double_value(&[]);
        }
        
        if epoch % 10 == 0 {
            println!("Epoch {}/{}, Loss: {:.4}", epoch, num_epochs, (total_loss / num_batches as f64) / batch_size as f64);
            let start_seq = "is not her";
            let predicted_text = generate_text(&model, &start_seq, char_to_idx, idx_to_char, 1 as usize, 128);
            println!("{:?}", predicted_text);
        
        }
    }
    model

}

// 6️⃣ Text Generation Function
pub fn generate_text(model: &LSTMModel, start_seq: &str, char_to_idx: &HashMap<char, i64>, idx_to_char: &HashMap<i64, char>, length: usize, hidden_size:i64) -> String {
    let device = Device::cuda_if_available();
    let mut generated = start_seq.to_string();

    let mut input_tensor = Tensor::from_slice(
        &start_seq.chars().filter_map(|c| char_to_idx.get(&c).copied()).collect::<Vec<i64>>()
    ).view([1, SEQ_LEN as i64]).to_device(device);
    let mut h = Tensor::zeros(&[1, 1, hidden_size], (tch::Kind::Float, device));
    let mut c = Tensor::zeros(&[1, 1, hidden_size], (tch::Kind::Float, device));

    for _ in 0..length {
        let (logits, state_new) = model.forward(&input_tensor, h, c);
        h = state_new.h();
        c = state_new.c();
        let next_idx = logits
            .softmax(-1, tch::Kind::Float)
            .argmax(-1, false).get(0).get(2).int64_value(&[]);

        if let Some(&ch) = idx_to_char.get(&next_idx) {
            generated.push(ch);
        } else {
            break;
        }

        input_tensor = Tensor::from_slice(&[next_idx]).view([1, 1]).to_device(device);
    }

    generated
}