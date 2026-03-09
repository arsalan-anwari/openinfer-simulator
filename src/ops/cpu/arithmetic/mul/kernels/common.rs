use crate::tensor::{BF16, F16, F8};

pub trait MulElement: Copy {
    fn mul(self, rhs: Self) -> Self;
}

impl MulElement for f32 {
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl MulElement for f64 {
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl MulElement for i8 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for i16 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for i32 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for i64 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for u8 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for u16 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for u32 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for u64 {
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MulElement for bool {
    fn mul(self, rhs: Self) -> Self {
        self & rhs
    }
}

impl MulElement for F16 {
    fn mul(self, rhs: Self) -> Self {
        F16::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MulElement for BF16 {
    fn mul(self, rhs: Self) -> Self {
        BF16::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MulElement for F8 {
    fn mul(self, rhs: Self) -> Self {
        F8::from_f32(self.to_f32() * rhs.to_f32())
    }
}
