use crate::layer_trait::LayerTrait;

pub struct AffineLayer {
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
    pub fn new(size_in: usize, size_out: usize, weight: Vec<Vec<f32>>, bias: Vec<f32>) -> Self{
        Self {
            size_in,
            size_out,
            weight,
            bias,
            input_cach: Vec::with_capacity(size_in),

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
        //reset cach
        self.input_cach.clear();
        self.input_cach = input.clone();
        //println!("{} {}: {:?} {:?}", self.size_in, self.size_out ,self.input_cach[0..10].to_vec(), input[0..10].to_vec());

        

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
        //println!("{}", input.len());
        for col in self.weight.iter() {
            if input_clone.len() != col.len() {
                panic!("Affine-forward(): matrix size err. input_colne: {}, weight.col: {}", input_clone.len(), col.len());
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
        //println!("{}", input.len());
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
        
        //println!("{:?}", self.input_cach[0..10].to_vec());
        
        //dot(self.input_cach.T, dout)
        //let old = self.gradient_buffer_weight.clone();
        for (col, &dout_i) in dout.iter().enumerate() {
            for (row, &input_i) in self.input_cach.iter().enumerate() {
                //ここの+=が=になってた。それは更新量ほぼ０になるわな。
                //println!("{} {} -> {}", input_i, dout_i, input_i * dout_i);
                self.gradient_buffer_weight[col][row] += input_i * dout_i;
            }
        }
        /* changeしてない
        let mut changed = false;
        for (old, new) in old.iter().zip(self.gradient_buffer_weight.iter()) {
            for (o, n) in old.iter().zip(new.iter()) {
                if o != n {
                    changed = true;
                }
            }
        }
        if changed == false {
            panic!();
        }*/
              

        //bias は加算なので微分値が1。上から降ってきたsize_outの誤差をそのままでよい
        for (d_bias_this, dout_this) in self.gradient_buffer_bias.iter_mut().zip(dout.iter()) {
            *d_bias_this += *dout_this;
        }

        

        //local variableは実体がdorpするので返却できない。中身の所有権を移動できないのか？
        dout.clear();
        dout.append(&mut dx);
        return dout;
    }

    fn clear_gradient_buffer(&mut self) {
        self.gradient_buffer_weight = vec![vec![0.0; self.size_in]; self.size_out];
        self.gradient_buffer_bias = vec![0.0; self.size_out];
    }
    //(weight, bias)
    fn get_gradient_buffer(&mut self) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
        //コピー入っちゃうけどこの関数はbatchでしか呼ばれないからまあ。。
        //古いのをクローン
        //println!("{:?}", self.gradient_buffer_weight[0][0..10].to_vec());
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

    fn seek_gradient_size(&self) -> Option<((usize, usize), usize)> {
        Some(((self.size_out, self.size_in), self.size_out))
    }

    fn update_gradient(&mut self, gradient: &(Vec<Vec<f32>>, Vec<f32>), learnig_rate: f32) -> Result<&str, &str> {
        //println!("Affine: update_gradient()");
        //weight update
        if gradient.0.len() != self.weight.len() {
            panic!("Affine-update_gradient: size err. gradient.0.len(): {}, self.weight.len(): {}", gradient.0.len(), self.weight.len());
        }
        //println!("{} {}", self.weight.len(), self.weight[0].len());
        //println!("before: {:?}", self.weight[4]);
        //println!("gradient: {:?}", gradient.0[4]);
        for i in 0..self.weight.len() {
            for j in 0..self.weight[0].len() {
                //println!("{} {} {}", self.weight[i][j], gradient.0[i][j], learnig_rate);
                self.weight[i][j] -= gradient.0[i][j] * learnig_rate;
                //println!("{}", self.weight[i][j]);
                
            }
            //panic!();
        }
        //println!("after: {:?}", self.weight[4]);

        //println!("bias grad: {:?}", gradient.1);
        //bias update
        for (b, g) in self.bias.iter_mut().zip(gradient.1.iter()) {
            *b -= g * learnig_rate;
        }
        Ok("Affine-update_gradient(): weight updated")
    }

    fn seek_weights(&self) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
        return Some((self.weight.clone(), self.bias.clone()));
    }
}