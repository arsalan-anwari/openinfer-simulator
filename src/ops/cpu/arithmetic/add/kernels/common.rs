use crate::tensor::{BF16, F16, F8};

pub trait AddElement: Copy {
    fn add(self, rhs: Self) -> Self;
}

impl AddElement for f32 {
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl AddElement for f64 {
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl AddElement for i8 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for i16 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for i32 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for i64 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for u8 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for u16 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for u32 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for u64 {
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl AddElement for bool {
    fn add(self, rhs: Self) -> Self {
        let sum = (self as u8).wrapping_add(rhs as u8);
        sum != 0
    }
}

impl AddElement for F16 {
    fn add(self, rhs: Self) -> Self {
        F16::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddElement for BF16 {
    fn add(self, rhs: Self) -> Self {
        BF16::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl AddElement for F8 {
    fn add(self, rhs: Self) -> Self {
        F8::from_f32(self.to_f32() + rhs.to_f32())
    }
}
