use super::point::Point;

/// Represents a quadrilateral defined by its four vertices.
pub struct Quad {
    pub bottom_left: Point,
    pub bottom_right: Point,
    pub top_right: Point,
    pub top_left: Point,
}

impl Quad {
    /// Constructs a new `Quad` from four arbitrary vertices.
    pub fn new(bottom_left: Point, bottom_right: Point, top_right: Point, top_left: Point) -> Self {
        Quad {
            bottom_left,
            bottom_right,
            top_right,
            top_left,
        }
    }

    /// Constructs a new `Quad` given the center and half dimensions.
    pub fn new_rect(center: &Point, half_width: f64, half_height: f64) -> Self {
        let left = center.x - half_width;
        let right = center.x + half_width;
        let bottom = center.y - half_height;
        let top = center.y + half_height;
        Quad {
            bottom_left: Point { x: left, y: bottom },
            bottom_right: Point { x: right, y: bottom },
            top_right: Point { x: right, y: top },
            top_left: Point { x: left, y: top },
        }
    }
    
    /// Returns an array of all vertices in counter-clockwise order starting from bottom-left.
    pub fn vertices(&self) -> [Point; 4] {
        [
            self.bottom_left,
            self.bottom_right,
            self.top_right,
            self.top_left,
        ]
    }
}
