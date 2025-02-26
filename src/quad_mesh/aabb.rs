use crate::utils::intersection::{do_intersect, line_segment_intersection};
use super::{point::Point, polygon::Polygon};

#[derive(Debug, Clone, Copy)]
pub struct AABB {
    pub center: Point,
    pub half_width: f64,
    pub half_height: f64,
}

impl AABB {
    pub fn new(center: Point, half_width: f64, half_height: f64) -> Self {
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

    pub fn width(&self) -> f64 {
        2.0 * self.half_width
    }

    pub fn height(&self) -> f64 {
        2.0 * self.half_height
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

    pub fn intersects_segment(&self, a: &Point, b: &Point) -> bool {
        let tl = &self.top_left();
        let tr = &self.top_right();
        let bl = &self.bottom_left();
        let br = &self.bottom_right();

        do_intersect(a, b, tl, tr)
            || do_intersect(a, b, tr, br)
            || do_intersect(a, b, br, bl)
            || do_intersect(a, b, bl, tl)
    }
}
