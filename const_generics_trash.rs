struct TwoLayerNet<
    const INPUT_SIZE: usize,
    const HIDDEN_SIZE: usize,
    const OUTPUT_SIZE: usize,
    > {
    //constでサイズを事前検証できるのめっちゃいいな
    weight1: [[f32; INPUT_SIZE]; HIDDEN_SIZE],
    bias1: [f32; HIDDEN_SIZE],
    weight2: [[f32; HIDDEN_SIZE]; OUTPUT_SIZE],
    bias2: [f32; OUTPUT_SIZE],
}
impl<
    const INPUT_SIZE: usize,
    const HIDDEN_SIZE: usize,
    const OUTPUT_SIZE: usize
    > TwoLayerNet<INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE> {
    fn new(weight_init_std: f32, rng: ThreadRng) -> Self {
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
        let weight1 = [[weight_init_std * rng.gen::<f32>(); INPUT_SIZE]; HIDDEN_SIZE];
        let bias1 = [0.0; HIDDEN_SIZE];
        let weight2 = [[weight_init_std * rng.gen::<f32>(); HIDDEN_SIZE]; OUTPUT_SIZE];
        let bias2 = [0.0; OUTPUT_SIZE];
        Self {
            weight1,
            bias1,
            weight2,
            bias2,
        }
    }
}