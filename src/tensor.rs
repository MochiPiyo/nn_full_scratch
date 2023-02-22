

//mod dot;
//mod tensor1d;
//mod tensor2d;


pub trait Num<Rhs = Self, Output = Self>:
    //+, -, *, /, %
    std::ops::Add<Rhs, Output = Output>
    + std::ops::Sub<Rhs, Output = Output>
    + std::ops::Mul<Rhs, Output = Output>
    + std::ops::Div<Rhs, Output = Output>
    + std::ops::Rem<Rhs, Output = Output>// % operator

    //+=, -=, *=, /=, %=
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::RemAssign


    + Default
    + Copy
{}

pub trait TensorTrait {
    //init by Default::default() value
    fn new() -> Self;
    
    //+, -, *, /
    fn add(&mut self, other: Self);
    fn sub(&mut self, other: Self);
    fn mul(&mut self, other: Self);
    fn div(&mut self, other: Self);
    fn rem(&mut self, otehr: Self);

    
}



//for easy usage
//別々にimplしたtraitが全部ついてるかのチェックにもなる
pub trait Tensor<Rhs = Self, Output = Self>:
    TensorTrait

    //+, -, *, /, %
    + std::ops::Add<Rhs, Output = Output>
    + std::ops::Sub<Rhs, Output = Output>
    + std::ops::Mul<Rhs, Output = Output>
    + std::ops::Div<Rhs, Output = Output>
    + std::ops::Rem<Rhs, Output = Output>

    //+=, -=, *=, /=, %=
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::ops::RemAssign


    + Clone
{}


/*
TensorTrait defines operation to change self value
    example: add(&mut self, other: Self); //no return
and then,
impl operator(+, -, *,...) trait, which creates new object, using TensorTrait fnctions.
example:
---------------------------------------------
    fn add(self, rhs: Self) -> Self::Output {
        let out = self.clone(); //create new.
        return out.add(rhs); //add operation in TensorTrait
    }
----------------------------------------------
 */





