use super::{point::Point, polygon::Polygon};

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub center: Point,
    pub half_width: f32,
    pub half_height: f32,
}

impl AABB {
    pub fn new(center: Point, half_width: f32, half_height: f32) -> Self {
        AABB {
            center,
            half_width,
            half_height,
        }
    }

    pub fn to_polygon(&self) -> Polygon {
        Polygon::new_rect(
            self.center.x - self.half_width,
            self.center.y - self.half_height,
            self.half_width * 2.0,
            self.half_height * 2.0,
        )
    }

    pub fn top_left(&self) -> Point {
        Point {
            x: self.center.x - self.half_width,
            y: self.center.y - self.half_height,
        }
    }

    pub fn top_right(&self) -> Point {
        Point {
            x: self.center.x + self.half_width,
            y: self.center.y - self.half_height,
        }
    }

    pub fn bottom_left(&self) -> Point {
        Point {
            x: self.center.x - self.half_width,
            y: self.center.y + self.half_height,
        }
    }

    pub fn bottom_right(&self) -> Point {
        Point {
            x: self.center.x + self.half_width,
            y: self.center.y + self.half_height,
        }
    }

    pub fn contains(&self, point: Point) -> bool {
        point.x >= self.top_left().x
            && point.x <= self.top_right().x
            && point.y >= self.top_left().y
            && point.y <= self.bottom_left().y
    }

    pub fn intersects(&self, other: &AABB) -> bool {
        self.top_right().x >= other.top_left().x
            && self.top_left().x <= other.top_right().x
            && self.top_left().y <= other.bottom_left().y
            && self.bottom_left().y >= other.top_left().y
    }
}