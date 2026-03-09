use crate::tensor::{BF16, F16, F8};

pub trait AbsElement: Copy {
    fn abs_value(self) -> Self;
}

impl AbsElement for f32 {
    fn abs_value(self) -> Self {
        self.abs()
    }
}

impl AbsElement for f64 {
    fn abs_value(self) -> Self {
        self.abs()
    }
}

impl AbsElement for i8 {
    fn abs_value(self) -> Self {
        self.wrapping_abs()
    }
}

impl AbsElement for i16 {
    fn abs_value(self) -> Self {
        self.wrapping_abs()
    }
}

impl AbsElement for i32 {
    fn abs_value(self) -> Self {
        self.wrapping_abs()
    }
}

impl AbsElement for i64 {
    fn abs_value(self) -> Self {
        self.wrapping_abs()
    }
}

impl AbsElement for F16 {
    fn abs_value(self) -> Self {
        F16::from_f32(self.to_f32().abs())
    }
}

impl AbsElement for BF16 {
    fn abs_value(self) -> Self {
        BF16::from_f32(self.to_f32().abs())
    }
}

impl AbsElement for F8 {
    fn abs_value(self) -> Self {
        F8::from_f32(self.to_f32().abs())
    }
}
