use super::{Num, TensorTrait};

#[derive(Clone)]
pub struct Tensor2d<T, const V: usize, const H: usize> {
    //vertical len: V, horizontal len: H
    pub body: [[T; H]; V],
    pub length: [usize; 2],
    //for transpose. default is length[axis[0]] == length[0]
    pub axis: [usize; 2],
}
impl<T: Num,  const V: usize, const H: usize> Tensor2d<T, V, H> {
    pub fn new_fill_with(init: T) -> Self {
        Self {
            body: [[init; H]; V],
            length: [V, H],
            axis: [0, 1],
        }
    }
    
    pub fn new_from_array(array: [[T; H]; V]) -> Self {
        Self {
            body: array,
            length: [V, H],
            axis: [0, 1],
        }
    }

    pub fn new_from_vec(vec: Vec<Vec<T>>) -> Self {
        let mut body = [[T::default(); H]; V];
        for (self_row, vec_row) in body.iter_mut().zip(vec.iter()) {
            for (self_i, vec_i) in self_row.iter_mut().zip(vec_row.iter()) {
                *self_i = *vec_i;
            }
        }
        Self {
            body,
            length: [V, H],
            axis: [0, 1],
        }
    }

    pub fn new_from_slices(slices: &[&[T]]) -> Self {
        let mut body = [[T::default(); H]; V];
        for (self_row, slice_row) in body.iter_mut().zip(slices.iter()) {
            for (self_i, slice_i) in self_row.iter_mut().zip(slice_row.iter()) {
                *self_i = *slice_i;
            }
        }
        Self {
            body,
            length: [V, H],
            axis: [0, 1],
        }
    }

    pub fn transpose_as_2d(&mut self) -> Result<[usize; 2], [usize; 2]> {
        if self.axis == [0, 1] {
            self.axis = [1, 0];
            return Ok(self.axis);
        }else if self.axis == [1, 0] {
            self.axis = [0, 1];
            return Ok(self.axis);
        }else {
            return Err(self.axis);
        }
    }
}

impl<T: Num, const V: usize, const H: usize> TensorTrait for Tensor2d<T, V, H> {
    fn new() -> Self {
        Self {
            body: [[T::default(); H]; V],
            length: [V, H],
            axis: [0, 1],
        }
    }

    fn add(&mut self, other: Self) {
        for (self_col, other_col) in self.body.iter_mut().zip(other.body.iter()) {
            for (self_i, other_i) in self_col.iter_mut().zip(other_col.iter()) {
                *self_i += *other_i;
            }
        }
    }

    fn sub(&mut self, other: Self) {
        for (self_col, other_col) in self.body.iter_mut().zip(other.body.iter()) {
            for (self_i, other_i) in self_col.iter_mut().zip(other_col.iter()) {
                *self_i -= *other_i;
            }
        }
    }

    fn mul(&mut self, other: Self) {
        for (self_col, other_col) in self.body.iter_mut().zip(other.body.iter()) {
            for (self_i, other_i) in self_col.iter_mut().zip(other_col.iter()) {
                *self_i *= *other_i;
            }
        }
    }

    fn div(&mut self, other: Self) {
        for (self_col, other_col) in self.body.iter_mut().zip(other.body.iter()) {
            for (self_i, other_i) in self_col.iter_mut().zip(other_col.iter()) {
                *self_i /= *other_i;
            }
        }
    }

    fn rem(&mut self, other: Self) {
        for (self_col, other_col) in self.body.iter_mut().zip(other.body.iter()) {
            for (self_i, other_i) in self_col.iter_mut().zip(other_col.iter()) {
                *self_i %= *other_i;
            }
        }
    }
}



//---impl Ops --------------------------------------------------------------
impl<T: Num, const V: usize, const H: usize> std::ops::Add for Tensor2d<T, V, H> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_row, other_row) in returns.body.iter_mut().zip(rhs.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i += *other_i;
            }
        }
        returns
    }
}
impl<T: Num, const V: usize, const H: usize> std::ops::Sub for Tensor2d<T, V, H> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_row, other_row) in returns.body.iter_mut().zip(rhs.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i -= *other_i;
            }
        }
        returns
    }
}
impl<T: Num, const V: usize, const H: usize> std::ops::Mul for Tensor2d<T, V, H> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_row, other_row) in returns.body.iter_mut().zip(rhs.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i *= *other_i;
            }
        }
        returns
    }
}
impl<T: Num, const V: usize, const H: usize> std::ops::Div for Tensor2d<T, V, H> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_row, other_row) in returns.body.iter_mut().zip(rhs.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i /= *other_i;
            }
        }
        returns
    }
}
impl<T: Num, const V: usize, const H: usize> std::ops::Rem for Tensor2d<T, V, H> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_row, other_row) in returns.body.iter_mut().zip(rhs.body.iter()) {
            for (self_i, other_i) in self_row.iter_mut().zip(other_row.iter()) {
                *self_i %= *other_i;
            }
        }
        returns
    }
}



//---impl OpsAssign --------------------------------------------------------------
impl<T: Num, const V: usize, const H: usize> std::ops::AddAssign for Tensor2d<T, V, H> {
    fn add_assign(&mut self, rhs: Self) {
        self.add(rhs)
    }
}
impl<T: Num, const V: usize, const H: usize> std::ops::SubAssign for Tensor2d<T, V, H> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub(rhs)
    }
}
impl<T: Num, const V: usize, const H: usize> std::ops::MulAssign for Tensor2d<T, V, H> {
    fn mul_assign(&mut self, rhs: Self) {
        self.mul(rhs)
    }
}
impl<T: Num, const V: usize, const H: usize> std::ops::DivAssign for Tensor2d<T, V, H> {
    fn div_assign(&mut self, rhs: Self) {
        self.div(rhs)
    }
}
impl<T: Num, const V: usize, const H: usize> std::ops::RemAssign for Tensor2d<T, V, H> {
    fn rem_assign(&mut self, rhs: Self) {
        self.rem(rhs)
    }
}
