pub trait LayerTrait {
    //返り値のライフタイムは受け取ったものと同じ
    fn forward<'input>(&mut self, input: &'input mut Vec<f32>) -> &'input mut Vec<f32>;
    fn backward<'dout>(&mut self, dout: &'dout mut Vec<f32>) -> &'dout mut Vec<f32>;
    //backwardで貯めた更新量を取り出す。無いLayerにはデフォルトのNone
    fn get_gradient_buffer(&mut self) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
        None
    }
    fn clear_gradient_buffer(&mut self) {
        
    }
    fn seek_gradient_size(&self) -> Option<((usize, usize), usize)> {
        None
    }
    fn update_gradient(&mut self, _gradient: &(Vec<Vec<f32>>, Vec<f32>), _learning_rate: f32) -> Result<&str, &str> {
        Err("this layer has nothing to update")
    }

    //(weight, bias)
    fn seek_weights(&self) -> Option<(Vec<Vec<f32>>, Vec<f32>)> {
        None
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

