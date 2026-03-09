use crate::tensor::{BF16, F16, F8};

pub trait MatmulElement: Copy {
    fn zero() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
}

impl MatmulElement for f32 {
    fn zero() -> Self {
        0.0
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl MatmulElement for f64 {
    fn zero() -> Self {
        0.0
    }

    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl MatmulElement for i8 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for i16 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for i32 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for i64 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for u8 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for u16 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for u32 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for u64 {
    fn zero() -> Self {
        0
    }

    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }

    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl MatmulElement for bool {
    fn zero() -> Self {
        false
    }

    fn add(self, rhs: Self) -> Self {
        let sum = (self as u8).wrapping_add(rhs as u8);
        sum != 0
    }

    fn mul(self, rhs: Self) -> Self {
        self & rhs
    }
}

impl MatmulElement for F16 {
    fn zero() -> Self {
        F16::from_f32(0.0)
    }

    fn add(self, rhs: Self) -> Self {
        F16::from_f32(self.to_f32() + rhs.to_f32())
    }

    fn mul(self, rhs: Self) -> Self {
        F16::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MatmulElement for BF16 {
    fn zero() -> Self {
        BF16::from_f32(0.0)
    }

    fn add(self, rhs: Self) -> Self {
        BF16::from_f32(self.to_f32() + rhs.to_f32())
    }

    fn mul(self, rhs: Self) -> Self {
        BF16::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl MatmulElement for F8 {
    fn zero() -> Self {
        F8::from_f32(0.0)
    }

    fn add(self, rhs: Self) -> Self {
        F8::from_f32(self.to_f32() + rhs.to_f32())
    }

    fn mul(self, rhs: Self) -> Self {
        F8::from_f32(self.to_f32() * rhs.to_f32())
    }
}
