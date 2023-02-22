use super::{Num, TensorTrait};

#[derive(Clone)]
pub struct Tensor1d<T, const N: usize> {
    pub body: [T; N],

    pub length: usize,
    pub axis: usize,
}
impl<T: Num, const N: usize> Tensor1d<T, N> {
    fn new_fill_with(init: T) -> Self {
        Self {
            body: [init; N],
            length: N,
            axis: 0,
        }
    }
    fn new_from_array(array: [T; N]) -> Self {
        Self {
            body: array,
            length: N,
            axis: 0,
        }
    }

    fn new_from_vec(vec: &Vec<T>) -> Result<Self, String> {
        if vec.len() < N {
            return Err(format!("vec is to short. vec.len() = {}, while expect length is {}", vec.len(), N));
        }else if vec.len() > N {
            return Err(format!("vec is to long. vec.len() = {}, while expect length is {}", vec.len(), N));
        }
        let mut body = [T::default(); N];
        for i in 0..N {
            body[i] = vec[i];
        }
        
        Ok(Self {
            body,
            length: N,
            axis: 0,
        })
    }
    
    //return the converted self.axis
    fn transpose_as_2d(&mut self) -> Result<usize, usize> {
        if self.axis == 0 {
            self.axis = 1;
            return Ok(self.axis);
        }else if self.axis == 1 {
            self.axis = 0;
            return Ok(self.axis);
        }else {
            return Err(self.axis);
        }
    }
}

impl<T: Num, const N: usize> TensorTrait for Tensor1d<T, N> {
    fn new() -> Self {
        Self {
            body: [T::default(); N],
            length: N,
            axis: 0,
        }
    }

    fn add(&mut self, other: Self) {
        for (self_i, other_i) in self.body.iter_mut().zip(other.body.iter()) {
            *self_i += *other_i;
        }
    }

    fn sub(&mut self, other: Self) {
        for (self_i, other_i) in self.body.iter_mut().zip(other.body.iter()) {
            *self_i -= *other_i;
        }
    }

    fn mul(&mut self, other: Self) {
        for (self_i, other_i) in self.body.iter_mut().zip(other.body.iter()) {
            *self_i *= *other_i;
        }
    }

    fn div(&mut self, other: Self) {
        for (self_i, other_i) in self.body.iter_mut().zip(other.body.iter()) {
            *self_i /= *other_i;
        }
    }

    fn rem(&mut self, other: Self) {
        for (self_i, other_i) in self.body.iter_mut().zip(other.body.iter()) {
            *self_i %= *other_i;
        }
    }
}


//---impl Ops --------------------------------------------------------------
impl<T: Num, const N: usize> std::ops::Add for Tensor1d<T, N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_i, other_i) in returns.body.iter_mut().zip(rhs.body.iter()) {
            *self_i += *other_i;
        }
        returns
    }
}
impl<T: Num, const N: usize> std::ops::Sub for Tensor1d<T, N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_i, other_i) in returns.body.iter_mut().zip(rhs.body.iter()) {
            *self_i -= *other_i;
        }
        returns
    }
}
impl<T: Num, const N: usize> std::ops::Mul for Tensor1d<T, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_i, other_i) in returns.body.iter_mut().zip(rhs.body.iter()) {
            *self_i *= *other_i;
        }
        returns
    }
}
impl<T: Num, const N: usize> std::ops::Div for Tensor1d<T, N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_i, other_i) in returns.body.iter_mut().zip(rhs.body.iter()) {
            *self_i /= *other_i;
        }
        returns
    }
}
impl<T: Num, const N: usize> std::ops::Rem for Tensor1d<T, N> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        let mut returns = self.clone();
        for (self_i, other_i) in returns.body.iter_mut().zip(rhs.body.iter()) {
            *self_i %= *other_i;
        }
        returns
    }
}

//---impl OpsAssign --------------------------------------------------------------
impl<T: Num, const N: usize> std::ops::AddAssign for Tensor1d<T, N> {
    fn add_assign(&mut self, rhs: Self) {
        self.add(rhs)
    }
}
impl<T: Num, const N: usize> std::ops::SubAssign for Tensor1d<T, N> {
    fn sub_assign(&mut self, rhs: Self) {
        self.sub(rhs)
    }
}
impl<T: Num, const N: usize> std::ops::MulAssign for Tensor1d<T, N> {
    fn mul_assign(&mut self, rhs: Self) {
        self.mul(rhs)
    }
}
impl<T: Num, const N: usize> std::ops::DivAssign for Tensor1d<T, N> {
    fn div_assign(&mut self, rhs: Self) {
        self.div(rhs)
    }
}
impl<T: Num, const N: usize> std::ops::RemAssign for Tensor1d<T, N> {
    fn rem_assign(&mut self, rhs: Self) {
        self.rem(rhs)
    }
}