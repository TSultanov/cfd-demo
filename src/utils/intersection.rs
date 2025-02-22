use crate::quad_mesh::point::Point;

// Helper: compute intersection point of segment p-q with segment a-b
pub fn line_segment_intersection(p: &Point, q: &Point, a: &Point, b: &Point) -> Option<Point> {
    let a1 = q.y - p.y;
    let b1 = p.x - q.x;
    let c1 = a1 * p.x + b1 * p.y;

    let a2 = b.y - a.y;
    let b2 = a.x - b.x;
    let c2 = a2 * a.x + b2 * a.y;

    let det = a1 * b2 - a2 * b1;
    if det.abs() < std::f32::EPSILON {
        return None; // lines are parallel
    }
    let x = (b2 * c1 - b1 * c2) / det;
    let y = (a1 * c2 - a2 * c1) / det;

    // Check if intersection lies on both segments.
    if x < p.x.min(q.x) - std::f32::EPSILON || x > p.x.max(q.x) + std::f32::EPSILON {
        return None;
    }
    if x < a.x.min(b.x) - std::f32::EPSILON || x > a.x.max(b.x) + std::f32::EPSILON {
        return None;
    }
    if y < p.y.min(q.y) - std::f32::EPSILON || y > p.y.max(q.y) + std::f32::EPSILON {
        return None;
    }
    if y < a.y.min(b.y) - std::f32::EPSILON || y > a.y.max(b.y) + std::f32::EPSILON {
        return None;
    }
    Some(Point { x, y })
}