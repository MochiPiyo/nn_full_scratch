use std::thread::panicking;

use rand::{Rng, rngs::ThreadRng, seq::SliceRandom};

use crate::model::TwoLayerNet;


mod affine_layer;
mod const_layers;

mod load_mnist;
mod layer_trait;
mod model;

mod tensor;

const NUMBER_OF_ROWS: usize = 28;
const NUMBER_OF_COLUMNS: usize = 28;

fn main() {
    //load mnist and serialize
    println!("prepare data");
    let image_path = "./train-images-idx3-ubyte";
    let label_path = "./train-labels-idx1-ubyte";
    let (image_datas, label_datas) = load_mnist::load_minst(image_path, label_path);

    let test_size: usize = 10000;
    let train_size = image_datas.len() - test_size;
    println!("train size: {}, test size: {}", train_size, test_size);

    let x_test: Vec<[f32; 784]> = load_mnist::selialize_minst(&image_datas[0..test_size]);
    let t_test: Vec<usize> = label_datas[0..test_size].iter().map(|i| *i as usize).collect();
    let x_train: Vec<[f32; 784]> = load_mnist::selialize_minst(&image_datas[test_size..]);
    //println!("{:?}", x_train[0][0..50].to_vec());\
    let t_train: Vec<usize> = label_datas[test_size..].iter().map(|i| *i as usize).collect();
    
    
    //model setting
    const INPUT_SIZE: usize = 784;//28*28
    const HIDDEN_SIZE: usize = 50;
    const OUTPUT_SIZE: usize = 10;
    let weight_init_std: f32 = 0.01;
    let mut rng = rand::thread_rng();
    println!("create network: input_size: {}, hidden_size: {}, output_size: {}", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    let mut network = TwoLayerNet::new(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, weight_init_std, &mut rng);
    
    
    //train setting
    const EPOCH_NUM: usize = 100;
    const PRINT_INTERVAL_OF_EPOCH: usize = 1;
    const BATCH_SIZE: usize = 1000;
    const LEARNING_RATE: f32 = 0.1;
    println!("iter num: {}, batch size: {}, learning rate: {}", EPOCH_NUM, BATCH_SIZE, LEARNING_RATE);
    
    let mut train_loss_list = Vec::new();
    let mut train_acc_list = Vec::new();
    let mut test_acc_list = Vec::new();


    println!("---train start---");
    for epoch in 0..EPOCH_NUM {
        //println!("start epoch {}", epoch);
        //BATCH_SIZE
        let batch_mask: Vec<u32> = (0..train_size as u32).collect::<Vec<u32>>().choose_multiple(&mut rng, BATCH_SIZE).cloned().collect();

        //勾配は一つずつ求めて、それを加算する方式にした。こうすると、layer内部の順方向のデータキャッシュがforwardでバッチ対応せずに済む。
        let mut grad: Vec<Option<(Vec<Vec<f32>>, Vec<f32>)>> = Vec::with_capacity(network.layers.len());
        //init grad with 0.0
        for layer in network.layers.iter() {
            match layer.seek_gradient_size() {
                None => grad.push(None),
                Some(((size_out, size_in), bias_size)) => {
                    let init = (vec![vec![0.0; size_in]; size_out], vec![0.0; bias_size]);
                    grad.push(Some(init));
                }
            }
        }
        for (n, &i) in batch_mask.iter().enumerate() {
            //println!("{} image[{}]", n, i);
            let mut input_data = x_train[i as usize].clone().to_vec();
            
            let mut has_non_zero = false;
            for i in input_data.iter() {
                if *i != 0.0 {
                    has_non_zero = true;
                }
            }
            if has_non_zero == false {
                panic!("input is zero");
            }


            let teacher_label = t_train[i as usize];
            
            //println!("{:?}", x_train[i as usize].to_vec());
            //println!("{}: {:?} - {}",i,  input_data[0..100].to_vec(), teacher_label);
            //データ一つあたりのgrad
            let this_grad: Vec<Option<(Vec<Vec<f32>>, Vec<f32>)>> = network.gradient(&mut input_data, teacher_label);

            //
            for (option_layer_grad, option_layer_grad_this) in grad.iter_mut().zip(this_grad.iter()) {
                if let Some(layer_grad) = option_layer_grad {
                    
                    //add
                    if let Some((w_this, b_this)) = option_layer_grad_this {
                        
                        //weight
                        //println!("{} {}", layer_grad.0.len(), layer_grad.0[0].len());
                        for i in 0..layer_grad.0.len() {
                            for j in 0..layer_grad.0[0].len() {
                                
                                layer_grad.0[i][j] += w_this[i][j];
                                //println!("epoch{} after :  {} {}", epoch, layer_grad.0[i][j], w_this[i][j]);
                            }
                        }
                        
                        //bias
                        for (b_g, b_g_this) in layer_grad.1.iter_mut().zip(b_this.iter()) {
                            *b_g += b_g_this;
                        }
                    }
                    
                }
            }
        }

        let seek = false;
            if seek {
                for (i, layer) in grad.iter().enumerate() {
                    if let Some((weights, biases)) = layer {
                        if i == 0 {
                            //わかった。問題は第一epochは更新されるが、それ以降は何も起こらない、ということだね。[0..10]は端っこすぎて分からなかった。
                            println!("layer{} grad: {:?}... , ", i, weights[0][300..350].to_vec());
                        //println!("layer{} bias  : {:?}...", i, biases[0..10].to_vec());
                        }
                    }
                }
            }

        
        if network.layers.len() != grad.len() {
            println!("leyers:{}, grad:{}", network.layers.len(), grad.len());
            panic!("size err");
        }
        for (layer, grad_to_update) in network.layers.iter_mut().zip(grad.iter()) {
            //println!("check");
            if let Some(grad) = grad_to_update {
                //println!("epoch:{} grad:{:?}", i, grad.0.get(0));
                match layer.update_gradient(grad, LEARNING_RATE) {
                    Ok(_s) => {
                        //println!("{}", s);
                    },
                    Err(s) => println!("{}", s),
                }
            }
        }

        
        let mut loss_sum = 0.0;
        for &i in batch_mask.iter() {
            //println!("Loss");
            loss_sum += network.loss(&mut x_train[i as usize].clone().to_vec(), t_train[i as usize]);
        }
        let loss = loss_sum / BATCH_SIZE as f32;
        train_loss_list.push(loss);

        //---end of epoch---
        if epoch % PRINT_INTERVAL_OF_EPOCH == 0 {
            
            

            let select_train: Vec<u32> = (0..train_size as u32).collect::<Vec<u32>>().choose_multiple(&mut rng, BATCH_SIZE).cloned().collect();
            let select_test: Vec<u32> = (0..test_size as u32).collect::<Vec<u32>>().choose_multiple(&mut rng, BATCH_SIZE).cloned().collect();
    
            let mut train_acc_sum: f64 = 0.0;
            for &i in select_train.iter() {
                //println!("accuracy");
                train_acc_sum += network.accuracy(&mut x_train[i as usize].clone().to_vec(), t_train[i as usize]) as f64;
            }
            let mut test_acc_sum: f64 = 0.0;
            for &i in select_test.iter() {
                test_acc_sum += network.accuracy(&mut x_test[i as usize].clone().to_vec(), t_test[i as usize]) as f64;
            }
    
            let train_acc = (train_acc_sum / BATCH_SIZE as f64) as f32;
            let test_acc = (test_acc_sum / BATCH_SIZE as f64) as f32;
    
            train_acc_list.push(train_acc);
            test_acc_list.push(test_acc);
    
            println!("epoch{:6}: train_acc = {:.1}%, test_acc = {:.1}%", epoch, train_acc * 100.0, test_acc * 100.0);
        }
    }

    println!("train_acc_list: {:?}", train_acc_list);
    println!("test_acc_list : {:?}", test_acc_list);
}