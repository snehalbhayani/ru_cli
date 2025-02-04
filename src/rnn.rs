// use tch::nn::{self, LSTMState, Module, OptimizerConfig, LSTM, RNN};
// use tch::{Device, Kind, Tensor};


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

// pub fn retrain_model (device, seq_len, history,) {

//         let mut training_data: Tensor = Tensor::empty(&[seq_len, 1, 36], (Kind::Float, device));
//         let mut target_data: Tensor = Tensor::empty(&[seq_len, 1], (Kind::Int64, device));
    
//         for (_, commands) in history.iter() {
//             for cmd in commands {
//                 let chars: Vec<char> = cmd.chars().collect();
//                 if chars.len() > seq_len2 {
//                     for i in 0..chars.len() - seq_len2 {
//                         let mut training_data_2: Tensor = Tensor::empty(&[1, 1, 36], (Kind::Float, device));
//                         let mut target_data_2: Tensor = Tensor::empty(&[1, 1], (Kind::Int64, device));
//                         let mut skip_seq = false;
//                         for j in 0..seq_len2 {
//                             let tn: Tensor = one_hot_encode_char(chars[i+j], device).reshape(&[1, 1, 36]);
//                             if let Some(idx) = char_to_index(chars[i+j+1]) {
//                                 let tn2: Tensor = (idx as i64)*Tensor::ones(&[1, 1], (Kind::Int64, device));                        
//                                 target_data_2 = Tensor::cat(&[target_data_2, tn2], 0);
//                                 training_data_2 = Tensor::cat(&[training_data_2, tn], 0);
//                             } else {
//                                 skip_seq = true;
//                             }
//                         }
//                         if !skip_seq {
//                             training_data = Tensor::cat(&[training_data, training_data_2.slice(0, 1, 3, 1)], 1);
//                             target_data   = Tensor::cat(&[target_data, target_data_2.slice(0, 1, 3, 1)], 1);
//                         }
//                     }
//                 }
//             }
//         }
//         // target_data.get(1).print();
//         println!("{:?}", training_data.size());
    
//         let train_inputs: Tensor = training_data.slice(1, 1, training_data.size()[1], 1);
//         let train_targets: Tensor = target_data.slice(1, 1, target_data.size()[1], 1);
//         // println!("{:?}--{:?}", train_inputs.size(),train_targets.size());
        
    
//         for epoch in 1..=epochs {
//         //     // Zero gradients
//             optimizer.zero_grad();
    
//             // Initialize LSTM state
//             let state = LSTMState((
//                 Tensor::zeros(&[1, batch_size, hidden_size], (Kind::Float, device)), // h
//                 Tensor::zeros(&[1, batch_size, hidden_size], (Kind::Float, device)), // c
//             ));
    
//             // Forward pass
//             let (output, _) = model.forward(&train_inputs, &state);
//             // println!("{:?}--{:?}", train_targets.reshape(&[2*494]), output.view([-1, output_size]).size() );        
//             // // Reshape output and compute loss
//             let loss = output.view([-1, output_size]) // Flatten for softmax
//                 .cross_entropy_for_logits(&train_targets.reshape(&[2*train_targets.size()[1]]));
    
//         // //     // Backpropagation
//             loss.backward();
//             optimizer.step();
    
//             // Print loss
//             if epoch % 100 == 0 {
//                 println!("Epoch {}: Loss = {:?}", epoch, loss.double_value(&[]));
//             }

    
//         println!("Training complete!");    
//     }
// }