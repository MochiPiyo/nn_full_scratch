use rand::{Rng, rngs::ThreadRng, seq::SliceRandom};


trait LayerTrait {
    //返り値のライフタイムは受け取ったものと同じ
    fn forward<'input>(&mut self, input: &'input mut Vec<f32>) -> &'input mut Vec<f32>;
    fn backward<'dout>(&mut self, dout: &'dout mut Vec<f32>) -> &'dout mut Vec<f32>;
    //backwardで貯めた更新量を取り出す。無いLayerにはデフォルトのNone
    fn get_gradient_buffer(&mut self) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
        None
    }
    fn update_gradient(&mut self, _gradient: &(Vec<Vec<f32>>, Vec<f32>), _learning_rate: f32) -> Result<&str, &str> {
        Err("this layer has nothing to update")
    }
}
/*拡張性が低いのでライブラリにできないからやめました。速度なんて些細なもんだろ
enum Layer {
    Affine(AffineLayer),
    Relu(ReluLayer),
}
//冗長だけどこれをやるとdyn、動的ディスパッチじゃなくなって速くなるらしい。
//呼び出し回数はたかがしれているけど、Vec<Box<dyn Trait>>って書くのあんまり。
impl Layer {
    fn forward(&self, input: &mut Vec<f32>) -> Vec<f32> {
        match self {
            Layer::Affine(layer) => layer.forward(input),
            Layer::Relu(layer) => layer.forward(input),
        }
    }
    fn backward(&self, dout: &mut Vec<f32>) -> Vec<f32> {
        match self {
            Layer::Affine(layer) => layer.backward(dout),
            Layer::Relu(layer) => layer.backward(dout),
        }
    }
}
*/

struct ReluLayer {
    size: usize,
    mask: Vec<bool>,
}
impl ReluLayer {
    fn new(size: usize) -> Self {
        Self { 
            size,
            mask: Vec::with_capacity(size),
        }
    }
}
impl LayerTrait for ReluLayer {
    fn forward<'input>(&mut self, input: &'input mut Vec<f32>) -> &'input mut Vec<f32> {
        if self.size != input.len() {
            panic!("Relu-forkward: input size is incorrect. self.size = {}, input.len() = {}", self.size, input.len());
        }
        for i in input.iter_mut() {
            if *i <= 0.0 {
                *i = 0.0;
                self.mask.push(true);
            }else {
                self.mask.push(false);
            }
        }
        return input;
    }

    fn backward<'dout>(&mut self, dout: &'dout mut Vec<f32>) -> &'dout mut Vec<f32> {
        if self.size != self.mask.len() {
            panic!("Relu-backward: self.mask is not initialized. It seems forward() has not excuted yet. ");
        }
        if self.size != dout.len() {
            panic!("Relu-backward: input size is incorrect. self.size = {}, dout.len() = {}", self.size, dout.len());
        }
        for (d, &is_masked) in dout.iter_mut().zip(self.mask.iter()) {
            if is_masked {
                *d = 0.0;
            }
        }
        return dout;
    }
}

struct SigmoidLayer {
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

struct AffineLayer {
    size_in: usize,
    size_out: usize,
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
    input_cach: Vec<f32>,

    //gradient buffer
    gradient_buffer_weight: Vec<Vec<f32>>,
    gradient_buffer_bias: Vec<f32>,
}
impl AffineLayer {
    //weitht[size_out][size_in]
    fn new(size_in: usize, size_out: usize, weight: Vec<Vec<f32>>, bias: Vec<f32>) -> Self{
        Self {
            size_in,
            size_out,
            weight,
            bias,
            input_cach: Vec::with_capacity(0),

            //gradient buffers are init by 0.0
            //推論モードでは外すべき
            gradient_buffer_weight: vec![vec![0.0; size_in]; size_out],
            gradient_buffer_bias: vec![0.0; size_out],
        }
    }
}
impl LayerTrait for AffineLayer {
    fn forward<'input>(&mut self, input: &'input mut Vec<f32>) -> &'input mut Vec<f32> {
        if input.len() != self.size_in {
            panic!("Affing-forward: input size is incorrect. self.size_in = {}, input.len() = {}", self.size_in, input.len());
        }
        //save input to use it in backward
        self.input_cach.clone_from_slice(input);

        /*
        matrix
            size_in -> size_out
            matrix.vertical: size_in
            matrix.horizontal: size_out
        */
        //こっちの方がキャッシュ効率良さげなので縦を１塊で格納する
        //out[0] = sum(data[eatch] * weight[0][eatch])を前から繰り返せるから。
        //weight[col][row] (MAX: weight[horizontal.len() - 1][vertical.len() - 1])
        //affine

        //local valueを外に出せないので仕方なくinputを流用する。letで名前変えるのもだめらしい。
        //絶対他にやりかたあるよな
        let input_clone = input.clone();
        input.clear();
        input.try_reserve_exact(self.size_out).unwrap();
        for col in self.weight.iter() {
            if input_clone.len() != col.len() {
                panic!("matrix size err at layer 1");
            }

            let mut col_sum = 0.0;
            for (weight, data_value) in col.iter().zip(input_clone.iter()) {
                col_sum += weight * data_value
            }
            input.push(col_sum);
        }
        //add bias
        for (i, bias_i) in input.iter_mut().zip(self.bias.iter()) {
            *i += bias_i;
        }
        return input;
    }

    fn backward<'dout>(&mut self, dout: &'dout mut Vec<f32>) -> &'dout mut Vec<f32> {

        //--diff for next layer--
        if self.size_out != dout.len() {
            panic!("Affine-backward: dout.len() is incorrect. self.size_out = {}, dout.len() = {}", self.size_out, dout.len());
        }
        //転地の高速化は今はしない
        let mut dx = Vec::with_capacity(self.size_in);
        for row in 0..self.size_in {
            let mut sum_of_dx_i = 0.0;
            for (col, d) in dout.iter().enumerate() {
                //縦切りで入っている
                sum_of_dx_i += (*d) * self.weight[col][row];
            }
            dx.push(sum_of_dx_i);
        }

        //-- diff to learn --
        //dot(input_cach[size_in].Tench(), dout[size_out]) -> d_weight[size_in][size_out]
        for (row, &i) in self.input_cach.iter().enumerate() {
            for (col, &d) in dout.iter().enumerate() {
                self.gradient_buffer_weight[col][row] = i * d;
            }
        }

        //bias は加算なので微分値が1。上から降ってきたsize_outの誤差をそのままでよい
        for (d_bias_this, dout_this) in self.gradient_buffer_bias.iter_mut().zip(dout.iter()) {
            *d_bias_this += *dout_this;
        }

        //reset cach
        self.input_cach.clear();

        //local variableは実体がdorpするので返却できない。中身の所有権を移動できないのか？
        dout.clone_from_slice(&dx);
        return dout;
    }

    //(weight, bias)
    fn get_gradient_buffer(&mut self) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
        //コピー入っちゃうけどこの関数はbatchでしか呼ばれないからまあ。。
        //古いのをクローン
        let gb_weight = self.gradient_buffer_weight.clone();
        let gb_bias = self.gradient_buffer_bias.clone();
        //selfの方は新しいのを割り当て
        self.gradient_buffer_weight = vec![vec![0.0; self.size_in]; self.size_out];
        self.gradient_buffer_bias = vec![0.0; self.size_out];
        //古いのをrerutn
        Some(
            (gb_weight, gb_bias)
        )
    }

    fn update_gradient(&mut self, gradient: &(Vec<Vec<f32>>, Vec<f32>), learnig_rate: f32) -> Result<&str, &str> {
        //weight update
        for (w_col, g_col) in self.weight.iter_mut().zip(gradient.0.iter()) {
            for (w, g) in w_col.iter_mut().zip(g_col.iter()) {
                *w = *w - g * learnig_rate;
            }
        }

        //bias update
        for (b, g) in self.bias.iter_mut().zip(gradient.1.iter()) {
            *b = *b - g * learnig_rate;
        }
        Ok("Affine-update_gradient(): weight updated")
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

//batched
//Vec of output: Vec<f32>, teacher_data is Vec<label_id>
fn batched_cross_entropy_error(output_batch: &Vec<Vec<f32>>, teacher_labels: &Vec<usize>) -> f32 {
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
struct SoftMaxWithLoss {
    //size_in == size_out
    size: usize,
    batch_size_cach: usize,
    output_batch_cach: Vec<Vec<f32>>,
    teacher_labels_cach: Vec<usize>,
}
impl SoftMaxWithLoss {
    fn new(size: usize) -> Self {
        Self {
            size,
            //cachs are initialized by temp value
            batch_size_cach: 0,
            output_batch_cach: Vec::new(),
            teacher_labels_cach: Vec::with_capacity(0),
        }
    }

    fn batched_forward(&mut self, input_batches: &mut Vec<Vec<f32>>, teacher_labels: Vec<usize>) -> f32 {
        if input_batches.len() != teacher_labels.len() {
            panic!("SoftMaxWithLoss-forward: batch size err. input_batches.len() = {}, teacher_label.len() = {}", input_batches.len(), teacher_labels.len());
        }
        //cach values to use in backward()
        self.batch_size_cach = input_batches.len();
        self.teacher_labels_cach = teacher_labels;

        //softmax
        for input in input_batches.iter_mut() {
            self.output_batch_cach.push(softmax(input).to_vec())
        }

        //cross entropy err
        let loss = batched_cross_entropy_error(&self.output_batch_cach, &self.teacher_labels_cach);

        return loss;
    }

    //return value is sum of batch
    //default dout as last_layer is 1.0
    fn batched_backward(&mut self, _dout: f32) -> Vec<f32> {
        if self.batch_size_cach == 0 {
            panic!("SoftMaxWithLoss-backward(): err, batch_size_cach == 0. you may have not excuted forward() yet.")
        }
        //(y[k] - t[k])
        //t[] is one hot
        let mut dx_sum = vec![0.0; self.size];
        for (output, &teacher_label) in self.output_batch_cach.iter_mut().zip(self.teacher_labels_cach.iter()) {
            //t[k] == 1である正解ラベルのとこだけ(y[k] - 1.0)するが、他は(y[k] - 0.0)なので省略できる
            output[teacher_label] -= 1.0;
            
            //add to get sum of dx.
            for (dx_i, output_i) in dx_sum.iter_mut().zip(output.iter()) {
                *dx_i += *output_i;
            }
        }
        
        //get average of batch
        for i in dx_sum.iter_mut() {
            *i = *i / self.batch_size_cach as f32;
        }
        let dx_average = dx_sum;

        //clear cach
        self.batch_size_cach = 0;
        self.output_batch_cach.clear();
        self.teacher_labels_cach.clear();

        return dx_average;

    }
}


/*
現状これはpredictを個別データごとにやってる
バッチ化方がweightを使いまわせるからいいのかもしれないが、途中データのメモリ消費が発生するな。
途中データがキャッシュに入るかってこと？


 */
struct TwoLayerNet {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,

    //define layers
    layers: Vec<Box<dyn LayerTrait>>,
    last_layer: SoftMaxWithLoss,
}
impl TwoLayerNet {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, weight_init_std: f32, rng: &mut ThreadRng) -> Self {
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
        let weight1 = vec![vec![weight_init_std * rng.gen::<f32>(); hidden_size]; input_size];
        let bias1 = vec![0.0; hidden_size];

        let weight2 = vec![vec![weight_init_std * rng.gen::<f32>(); output_size]; hidden_size];
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
            input_size,
            hidden_size,
            output_size,

            layers,
            last_layer,
        }
    }

    //data -> predict
    fn predict<'input>(&mut self, mut input_data: &'input mut Vec<f32>) -> &'input mut Vec<f32> {
        for layer in self.layers.iter_mut() {
            //layer gets temp_data(n) and returns temp_data(n+1)
            input_data = layer.forward(input_data);
        }
        return input_data;
    }

    fn loss(&mut self, input_data: &mut Vec<f32>, teacher_label: usize) -> f32 {
        let predict = &*self.predict(input_data);

        //this fn is for single data loss, but last_layer have only batched one, so dummy vec.
        let mut dummy_input_batch = Vec::with_capacity(1);
        dummy_input_batch.push(predict.to_vec());
        let mut dummy_teacher_label_batch = Vec::with_capacity(1);
        dummy_teacher_label_batch.push(teacher_label);
        let loss = self.last_layer.batched_forward(&mut dummy_input_batch, dummy_teacher_label_batch);

        return loss;
    }

    fn batched_loss(&mut self, batched_input_data: &mut Vec<Vec<f32>>, batched_teacher_label: &Vec<usize>) -> f32 {
        let mut loss_sum: f64 = 0.0;
        for (input_data, &teacher_label) in batched_input_data.iter_mut().zip(batched_teacher_label.iter()) {
            loss_sum += self.loss(input_data, teacher_label) as f64;
        }
        return (loss_sum / batched_teacher_label.len() as f64) as f32;
    }

    fn accuracy(&mut self, batched_input_data: &mut Vec<Vec<f32>>, batched_teacher_label: &Vec<usize>) -> f32 {
        if batched_input_data.len() != batched_teacher_label.len() {
            panic!("TwoLayerNet-accuracy: size err. batched_input_data.len() = {}, batched_teacher_label.len() = {}", batched_input_data.len(), batched_teacher_label.len());
        }
        let mut correct_num = 0;
        for (input_data, &teacher_label) in batched_input_data.iter_mut().zip(batched_teacher_label.iter()) {
            let predict = self.predict(input_data);

            let (mut largest, mut largest_index): (f32, usize) = (0.0, 0);
            for (index, &value) in predict.iter().enumerate() {
                if value > largest {
                    largest = value;
                    largest_index = index;
                }
            }

            if largest_index == teacher_label {
                correct_num += 1;
            }
        }
        
        return correct_num as f32 / batched_input_data.len() as f32; 
    }

    //return gradient diff of 
    fn gradient(&mut self, batched_input_data: &mut Vec<Vec<f32>>, batched_teacher_label: &Vec<usize>) -> Vec<Option<(Vec<Vec<f32>>, Vec<f32>)>> {
        //forward
        let mut loss_sum = 0.0;
        for (input_data, &teacher_label) in batched_input_data.iter_mut().zip(batched_teacher_label.iter()) {
            loss_sum += self.loss(input_data, teacher_label);
        }
        let loss_average = loss_sum / batched_input_data.len() as f32;
        println!("loss average of this training_batch is '{}'", loss_average);

        //backward
        let dout = 1.0;
        let mut dout = self.last_layer.batched_backward(dout);

        //backward layers in reverse order
        let mut temp_grad: &mut Vec<f32> = &mut dout;
        for layer in self.layers.iter_mut().rev() {
            temp_grad = layer.backward(temp_grad);
        }

        //get grads from layers
        let mut grads: Vec<Option<(Vec<Vec<f32>>, Vec<f32>)>> = Vec::with_capacity(self.layers.len());
        for layer in self.layers.iter_mut() {
            grads.push(layer.get_gradient_buffer());
        }
        return grads;
    }
}




//(data, label)
const NUMBER_OF_ROWS: usize = 28;
const NUMBER_OF_COLUMNS: usize = 28;
fn load_minst(image_path: &str, label_path: &str) -> (Vec<[[i32; NUMBER_OF_COLUMNS]; NUMBER_OF_ROWS]>, Vec<u8>) {
    println!("load_mnist");
    use std::fs::File;
    use std::io::Read;

    //load label
    let mut file = File::open(label_path).unwrap();
    let mut buf = Vec::new();
    let _ = file.read_to_end(&mut buf).unwrap();

    let mut labels = Vec::new();

    let mut offset: usize = 0;
    let _magic_number = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;
    let number_of_items = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;

    //label is array of u8
    for _i in 0..number_of_items {
        labels.push(buf[offset] as u8);
        offset += 1;
    }

    //load images
    let mut file = File::open(image_path).unwrap();
    let mut buf = Vec::new();
    let _ = file.read_to_end(&mut buf).unwrap();

    let mut offset: usize = 0;
    let _magic_number = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;
    let number_of_images = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;

    //28
    let number_of_rows = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;
    //28
    let number_of_columns = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
    offset += 4;

    if number_of_rows as usize != NUMBER_OF_ROWS || number_of_columns as usize != NUMBER_OF_COLUMNS {
        panic!("load_minst: size err. readed data from file is : number_of_rows = {}, number_of_columns = {}", number_of_rows, number_of_columns);
    }

    //read pixel data
    let mut image_datas: Vec<[[i32; NUMBER_OF_COLUMNS]; NUMBER_OF_ROWS]> = vec![
        [[0; NUMBER_OF_COLUMNS]; NUMBER_OF_ROWS]; number_of_images as usize
    ];
    for image_id in 0..number_of_images {
        let image: &mut [[i32; NUMBER_OF_COLUMNS]; NUMBER_OF_ROWS] = &mut image_datas[image_id as usize];
        for row_id in 0..number_of_rows {
            for column_id in 0..number_of_columns {
                let this_pixel = i32::from_be_bytes(buf[offset..offset+4].try_into().unwrap());
                offset += 4;

                image[row_id as usize][column_id as usize] = this_pixel;
            }
        }
    }

    return (image_datas, labels);
} 

fn main() {

    let image_path = "./train-images-idx3-ubyte";
    let label_path = "./train-labels-idx1-ubyte";
    let (image_datas, label_datas) = load_minst(image_path, label_path);

    if image_datas.len() == label_datas.len() {
        println!("num of data is '{}'", image_datas.len());
    }else {
        println!("err: image and label count is not same. image: {}, label: {}", image_datas.len(), label_datas.len());
        panic!();
    }

    

    let test_size: usize = 10000;
    let mut x_test: Vec<Vec<f32>> = Vec::with_capacity(test_size);
    for image in image_datas[0..test_size].iter() {
        //28*28を784にする
        let mut selialize = Vec::with_capacity(INPUT_SIZE);
        for j in 0..NUMBER_OF_ROWS {
            let mut row = image[j].iter().map(|i| *i as f32).collect();
            selialize.append(&mut row);
        }
        x_test.push(selialize);
    }
    let t_test: Vec<usize> = label_datas[0..test_size].iter().map(|i| *i as usize).collect();

    let train_size = image_datas.len() - test_size;
    let mut x_train: Vec<Vec<f32>> = Vec::with_capacity(train_size);
    for image in image_datas[test_size..].iter() {
        //28*28を784にする
        let mut selialize = Vec::with_capacity(INPUT_SIZE);
        for j in 0..NUMBER_OF_ROWS {
            let mut row = image[j].iter().map(|i| *i as f32).collect();
            selialize.append(&mut row);
        }
        x_test.push(selialize);
    }
    let t_train: Vec<usize> = label_datas[test_size..].iter().map(|i| *i as usize).collect();
    
    let train_size = x_train.len();
    println!("train size: {}, test size: {}", train_size, test_size);

    const ITER_NUM: usize = 10000;
    const BATCH_SIZE: usize = 100;
    const LEARNING_RATE: f32 = 0.1;
    println!("iter num: {}, batch size: {}, learning rate: {}", ITER_NUM, BATCH_SIZE, LEARNING_RATE);
    


    let mut train_loss_list = Vec::new();
    let mut train_acc_list = Vec::new();
    let mut test_acc_list = Vec::new();

    let iter_per_epoch = if train_size / BATCH_SIZE > 1 {
        train_size / BATCH_SIZE
    }else {
        1
    };


    
    let mut rng = rand::thread_rng();

    const INPUT_SIZE: usize = 784;//28*28
    const HIDDEN_SIZE: usize = 50;
    const OUTPUT_SIZE: usize = 10;
    let weight_init_std: f32 = 0.01;
    let mut network = TwoLayerNet::new(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, weight_init_std, &mut rng);

    for i in 0..ITER_NUM {
        let batch_mask: Vec<u32> = (0..train_size as u32).collect::<Vec<u32>>().choose_multiple(&mut rng, BATCH_SIZE).cloned().collect();

        //コピーなしでやりたいけどつかれた
        let mut batched_input_data: Vec<Vec<f32>> = Vec::with_capacity(BATCH_SIZE);
        for &i in batch_mask.iter() {
            batched_input_data.push(x_train[i as usize].clone());
        }
        let mut batched_teacher_label: Vec<usize> = Vec::with_capacity(BATCH_SIZE);
        for &i in batch_mask.iter() {
            batched_teacher_label.push(t_train[i as usize] as usize);
        }

        let grad = network.gradient(&mut batched_input_data, &batched_teacher_label);

        for (layer, grad_to_update) in network.layers.iter_mut().zip(grad.iter()) {
            if let Some(grad) = grad_to_update {
                match layer.update_gradient(grad, LEARNING_RATE) {
                    Ok(s) => println!("{}", s),
                    Err(s) => println!("{}", s),
                }
            }
        }

        let loss = network.batched_loss(&mut batched_input_data, &batched_teacher_label);
        train_loss_list.push(loss);

        //end of epoch
        if i % iter_per_epoch == 0 {
            let train_acc = network.accuracy(&mut x_train, &t_train);
            let test_acc = network.accuracy(&mut x_test, &t_test);

            train_acc_list.push(train_acc);
            test_acc_list.push(test_acc);

            println!("train_acc = {}, test_acc = {}", train_acc, test_acc);
        }
    }

    println!("train_acc_list: {:?}", train_acc_list);
    println!("test_acc_list : {:?}", test_acc_list);
}