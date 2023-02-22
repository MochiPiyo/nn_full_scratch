

use rand::Rng;
use rand::rngs::ThreadRng;

use crate::layer_trait::LayerTrait;
use crate::affine_layer::AffineLayer;
use crate::const_layers::{ReluLayer, SoftMaxWithLoss};

pub struct TwoLayerNet {
    _input_size: usize,
    _hidden_size: usize,
    _output_size: usize,

    //define layers
    pub layers: Vec<Box<dyn LayerTrait>>,
    pub last_layer: SoftMaxWithLoss,
}
impl TwoLayerNet {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, weight_init_std: f32, rng: &mut ThreadRng) -> Self {
        /*
        layer1
            input_size -> hidden_size
            matrix.vertical: input_size
            matrix.horizontal: hidden_size
        layer2
            hidden_size -> output_size
            matrix.vertical: hidden_size
            matrix.horizontal: output_size
        */
        //こっちの方がキャッシュ効率良さげなので縦を１塊で格納する
        //out[0] = sum(data[eatch] * weight[0][eatch])を前から繰り返せるから。
        //weight[col][row] (MAX: weight[horizontal.len() - 1][vertical.len() - 1])

        //define weights (these vecs are going to given for layer structs, and owned by them. thus, no copy.)
        let weight1 = vec![vec![weight_init_std * rng.gen::<f32>(); input_size]; hidden_size];
        let bias1 = vec![0.0; hidden_size];

        let weight2 = vec![vec![weight_init_std * rng.gen::<f32>(); hidden_size]; output_size];
        let bias2 = vec![0.0; output_size];

        //make layers
        let affine1 = AffineLayer::new(input_size, hidden_size, weight1, bias1);
        let relu = ReluLayer::new(hidden_size);
        let affine2 = AffineLayer::new(hidden_size, output_size, weight2, bias2);

        //define layers
        let layers: Vec<Box<dyn LayerTrait>> = vec![
            //Box::newが面倒だからマクロでも書くか。
            Box::new(affine1),
            Box::new(relu),
            Box::new(affine2),
        ];
        
        /*enum使うのはやめました */
        //use Layer::*;でLayer::Affine(affine1)とかを簡潔にできるが
        //use宣言を{}で囲わないと上のAffine::new()とかも反応しちゃうらしい。
        //だがlayersは{}でドロップして欲しくないので外で宣言して一度だけ代入
        //->という話もあったがAffineLayerに名前を変えることで解決
        
        
        let last_layer = SoftMaxWithLoss::new(output_size);

        Self {
            _input_size: input_size,
            _hidden_size: hidden_size,
            _output_size: output_size,

            layers,
            last_layer,
        }
    }

    //data -> predict
    fn predict<'input>(&mut self, mut input_data: &'input mut Vec<f32>) -> &'input mut Vec<f32> {
        //println!("model predict: layer lengt: {}", self.layers.len());
        for (n, layer) in self.layers.iter_mut().enumerate() {
            //layer gets temp_data(n) and returns temp_data(n+1)
            input_data = layer.forward(input_data);
            //println!("->{}", input_data.len());
        }
        return input_data;
    }

    pub fn loss(&mut self, input_data: &mut Vec<f32>, teacher_label: usize) -> f32 {
        let predict = &*self.predict(input_data);
        //println!("{}", predict.len());

        let loss = self.last_layer.forward(&mut predict.to_vec(), teacher_label);

        return loss;
    }

    pub fn accuracy(&mut self, input: &mut Vec<f32>, teacher_label: usize) -> f32 {
        let predict = self.predict(input);

        let (mut largest, mut largest_index): (f32, usize) = (0.0, 0);
        for (index, &value) in predict.iter().enumerate() {
            if value > largest {
                largest = value;
                largest_index = index;
            }
        }

        if largest_index == teacher_label {
            return 1.0;
        }else {
            return 0.0;
        }

    }

    //return layers[option<(weight_grad, bias_grad)>]
    pub fn gradient(&mut self, input_data: &mut Vec<f32>, teacher_label: usize) -> Vec<Option<(Vec<Vec<f32>>, Vec<f32>)>> {
        //clear gradient buffer
        for layer in self.layers.iter_mut() {
            layer.clear_gradient_buffer();
        }
        //forward
        let _loss = self.loss(input_data, teacher_label);

        //backward
        let dout = 1.0;
        let mut dout = self.last_layer.backward(dout);
        //println!("dout.len(): {}", dout.len());

        //backward layers in reverse order
        let mut temp_grad: &mut Vec<f32> = &mut dout;
        for layer in self.layers.iter_mut().rev() {
            //println!("temp_grad.len(): {}", temp_grad.len());
            temp_grad = layer.backward(temp_grad);
        }

        //get grads from layers
        let mut grads: Vec<Option<(Vec<Vec<f32>>, Vec<f32>)>> = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter_mut() {
            //println!("{:?}", layer.get_gradient_buffer());
            grads.push(layer.get_gradient_buffer());
        }
        return grads;
    }
}
