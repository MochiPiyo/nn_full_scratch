

use crate::layer_trait::LayerTrait;

pub struct ReluLayer {
    size: usize,
    mask_cach: Vec<bool>,
}
impl ReluLayer {
    pub fn new(size: usize) -> Self {
        //println!("init relu, size = {}", size);
        let mask_cach = vec![false; size];
        //println!("{}", mask_cach.len());
        Self { 
            size,
            mask_cach,
        }
    }
}
impl LayerTrait for ReluLayer {
    fn forward<'input>(&mut self, input: &'input mut Vec<f32>) -> &'input mut Vec<f32> {
        if self.size != input.len() {
            panic!("Relu-forkward: input size is incorrect. self.size = {}, input.len() = {}", self.size, input.len());
        }
        self.mask_cach.clear();
        for (i, f) in input.iter_mut().enumerate() {
            if *f <= 0.0 {
                *f = 0.0;
                self.mask_cach.push(true);
            }else {
                self.mask_cach.push(false);
            }
        }
        //println!("{}", self.mask_cach.len());
        return input;
    }

    fn backward<'dout>(&mut self, dout: &'dout mut Vec<f32>) -> &'dout mut Vec<f32> {
        if self.size != self.mask_cach.len() {
            panic!("Relu-backward: err, size is not same. self.mask.len() = {}, self.size = {}", self.mask_cach.len(), self.size);
        }
        if self.size != dout.len() {
            panic!("Relu-backward: input size is incorrect. self.size = {}, dout.len() = {}", self.size, dout.len());
        }
        for (d, &is_masked) in dout.iter_mut().zip(self.mask_cach.iter()) {
            if is_masked {
                *d = 0.0;
            }
        }


        return dout;
    }
}

pub struct SigmoidLayer {
    size: usize,
    //順方向出力の値を保持してbackwardで使う
    out: Vec<f32>,
}
impl SigmoidLayer {
    fn new(size: usize) -> Self {
        Self {
            size,
            out: Vec::with_capacity(size),
        }
    }
}
impl LayerTrait for SigmoidLayer {
    fn forward<'input>(&mut self, input: &'input mut Vec<f32>) -> &'input mut Vec<f32> {
        if self.size != input.len() {
            panic!("Sigmoid-forward: input size is incorrect. self.size = {}, input.len() = {}", self.size, input.len());
        }
        for i in input.iter_mut() {
            *i = 1.0 / (1.0 + (- *i).exp());
            
        }
        //save output and use it in backward()
        self.out.clone_from_slice(*&input);

        return input;
    }
    fn backward<'dout>(&mut self, dout: &'dout mut Vec<f32>) -> &'dout mut Vec<f32> {
        if self.size != dout.len() {
            panic!("Sigmoid-backward: dout size is incorrect. self.size = {}, dout.len() = {}", self.size, dout.len());
        }
        for (d, &out) in dout.iter_mut().zip(self.out.iter()) {
            //y (1 - y), y is saved as self.out
            *d = (*d) * (1.0 - out) * out;
        }
        return dout;
    }
}



//受け取ったVecに結果を詰めて返す
fn softmax(input: &mut Vec<f32>) -> &mut Vec<f32> {
    //for anti overflow
    let mut largest = 0.0;
    for i in input.iter() {
        if largest < *i {
            largest = *i;
        }
    }
    //exp
    let mut sum_exp = 0.0;
    for i in input.iter_mut() {
        //anti overflow
        *i = ((*i) - largest).exp();
        sum_exp += *i;
    }
    //exp / sum_exp
    for i in input.iter_mut() {
        *i = *i / sum_exp
    }

    return input;
}


fn cross_entropy_error(output: &Vec<f32>, teacher_label: usize) -> f32 {
    let error = (output[teacher_label] + f32::MIN).ln();
    return -error;
}

//batched
//Vec of output: Vec<f32>, teacher_data is Vec<label_id>
fn _batched_cross_entropy_error(output_batch: &Vec<Vec<f32>>, teacher_labels: &Vec<usize>) -> f32 {
    let batch_size = output_batch.len();

    let mut error_sum = 0.0;
    for (output, &teacher_label) in output_batch.iter().zip(teacher_labels.iter()) {
        /*
        E = - sum(t[k] * log(y[k]))
        だから、t[k]が0か1しかないとき、0なら消失、1ならlog(y[k])そのまま、であるから
        t[k]において他が0でteacher_labelだけが1なら、その場所だけ計算すればよい。
        */
        //ln()はeで対数とる
        error_sum += (output[teacher_label] + f32::MIN).ln();
    }
    return - error_sum / batch_size as f32;
}

//batched
pub struct SoftMaxWithLoss {
    //size_in == size_out
    _size: usize,
    output_cach: Vec<f32>,
    teacher_label_cach: usize,
}
impl SoftMaxWithLoss {
    pub fn new(size: usize) -> Self {
        Self {
            _size: size,
            //cachs are initialized by temp value
            output_cach: Vec::with_capacity(0),
            teacher_label_cach: 100,
        }
    }
    pub fn forward(&mut self, input: &mut Vec<f32>, teacher_label: usize) -> f32 {
        self.teacher_label_cach = teacher_label;

        self.output_cach = softmax(input).to_vec();
        let loss = cross_entropy_error(&self.output_cach, self.teacher_label_cach);
        return loss;
    }

    pub fn backward(&mut self, _dout: f32) -> Vec<f32> {
        let mut dx = self.output_cach.clone();

        dx[self.teacher_label_cach] -= 1.0;
        //println!("{}, {}, {}", self._size, self.output_cach.len(), dx.len());

        return dx;
    }
}