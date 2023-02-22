use super::{Num, tensor1d::Tensor1d, tensor2d::{Tensor2d, self}, TensorTrait};


pub fn dot_1d1d_2d<T: Num, const N: usize, const M: usize>
    (left: Tensor1d<T, N>, right: Tensor1d<T, M>, zero: T) -> Result<Tensor2d<T, N, M>, &'static str>
{
    if left.axis == 1 && right.axis == 0 {
        let mut returns: Tensor2d<T, N, M> = Tensor2d::new_fill_with(zero);
        for (returns_row, &left_i) in returns.body.iter_mut().zip(left.body.iter()) {
            for (returns_i, &right_i) in returns_row.iter_mut().zip(right.body.iter()) {
                *returns_i = left_i * right_i;
            }
        }
        return Ok(returns);
    }else {
        return Err("dot_1d1d_2d: axis err");
    }
}

pub fn dot_1d1d_scalar<T: Num, const N: usize, const M: usize>
    (left: Tensor1d<T, N>, right: Tensor1d<T, M>, zero: T) -> Result<T, &'static str>
{
    if left.axis == 0 && right.axis == 1 {
        let mut sum = zero;
        for &l in left.body.iter() {
            for &r in left.body.iter() {
                sum += l * r;
            }
        }
        return Ok(sum);
    }
    Err("dot_1d1d_scalar: size err")
}

pub fn dot_2d2d_2d<T: Num, const N: usize, const M: usize, const S: usize>
    (left: Tensor2d<T, N, M>, right: Tensor2d<T, M, S>, zero: T) -> Tensor2d<T, N, S>
{
    let result = tensor2d<T, N, M> = Tensor2d::new();
    if left.axis == [0, 1] && right.axis == [0, 1] {
        
    }

}