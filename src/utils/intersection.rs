use crate::quad_mesh::point::Point;

fn orientation(p: &Point, q: &Point, r: &Point) -> i32 {
    // Calculate the determinant (cross product)
    let val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if val.abs() < std::f32::EPSILON {
        0 // collinear
    } else if val > 0.0 {
        1 // clockwise
    } else {
        2 // counterclockwise
    }
}

fn on_segment(p: &Point, q: &Point, r: &Point) -> bool {
    (q.x <= p.x.max(r.x) + std::f32::EPSILON && q.x >= p.x.min(r.x) - std::f32::EPSILON) &&
        (q.y <= p.y.max(r.y) + std::f32::EPSILON && q.y >= p.y.min(r.y) - std::f32::EPSILON)
}

pub fn do_intersect(p: &Point, q: &Point, a: &Point, b: &Point) -> bool {
    // Find the four orientations needed for the general and special cases
    let o1 = orientation(p, q, a);
    let o2 = orientation(p, q, b);
    let o3 = orientation(a, b, p);
    let o4 = orientation(a, b, q);

    // General case: segments intersect if they straddle each other.
    if o1 != o2 && o3 != o4 {
        return true;
    }

    // Special cases:
    if o1 == 0 && on_segment(p, a, q) { return true; }   // a is collinear with p-q and lies on segment p-q
    if o2 == 0 && on_segment(p, b, q) { return true; }   // b is collinear with p-q and lies on segment p-q
    if o3 == 0 && on_segment(a, p, b) { return true; }   // p is collinear with a-b and lies on segment a-b
    if o4 == 0 && on_segment(a, q, b) { return true; }   // q is collinear with a-b and lies on segment a-b

    false
}

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

#[cfg(test)]
mod test {
    use crate::quad_mesh::point::Point;
    use crate::utils::intersection::{do_intersect, line_segment_intersection};

    #[test]
    fn test_line_intersection_intersecting() {
        // segments that clearly intersect: (0,0)-(1,1) and (0,1)-(1,0)
        let p1 = Point { x: 0.0, y: 0.0 };
        let q1 = Point { x: 1.0, y: 1.0 };
        let p2 = Point { x: 0.0, y: 1.0 };
        let q2 = Point { x: 1.0, y: 0.0 };
        assert!(line_segment_intersection(&p1, &q1, &p2, &q2).is_some());
    }

    #[test]
    fn test_line_intersection_non_intersecting_but_lines_do() {
        // segments whose infinite lines intersect at (1,1) but segments don't reach that point.
        let p1 = Point { x: 0.0, y: 0.0 };
        let q1 = Point { x: 0.5, y: 0.5 };
        let p2 = Point { x: 2.0, y: 0.0 };
        let q2 = Point { x: 3.0, y: -1.0 };
        assert!(line_segment_intersection(&p1, &q1, &p2, &q2).is_none());
    }

    #[test]
    fn test_line_intersection_parallel() {
        // parallel segments: (0,0)-(1,0) and (0,1)-(1,1)
        let p1 = Point { x: 0.0, y: 0.0 };
        let q1 = Point { x: 1.0, y: 0.0 };
        let p2 = Point { x: 0.0, y: 1.0 };
        let q2 = Point { x: 1.0, y: 1.0 };
        assert!(line_segment_intersection(&p1, &q1, &p2, &q2).is_none());
    }

    #[test]
    fn test_line_intersection_collinear() {
        // collinear segments with no overlap: (0,0)-(1,1) and (2,2)-(3,3)
        let p1 = Point { x: 0.0, y: 0.0 };
        let q1 = Point { x: 1.0, y: 1.0 };
        let p2 = Point { x: 2.0, y: 2.0 };
        let q2 = Point { x: 3.0, y: 3.0 };
        assert!(line_segment_intersection(&p1, &q1, &p2, &q2).is_none());
    }

    #[test]
    fn test_line_intersection_endpoint() {
        // Segments sharing an endpoint should return that endpoint.
        let p1 = Point { x: 0.0, y: 0.0 };
        let q1 = Point { x: 1.0, y: 1.0 };
        let p2 = Point { x: 1.0, y: 1.0 };
        let q2 = Point { x: 2.0, y: 0.0 };
        if let Some(ip) = line_segment_intersection(&p1, &q1, &p2, &q2) {
            assert!((ip.x - 1.0).abs() < std::f32::EPSILON);
            assert!((ip.y - 1.0).abs() < std::f32::EPSILON);
        } else {
            panic!("Expected intersection at endpoint (1.0, 1.0)");
        }
    }

    #[test]
    fn test_line_intersection_overlapping_collinear() {
        // Collinear overlapping segments should return None as per our implementation
        let p1 = Point { x: 0.0, y: 0.0 };
        let q1 = Point { x: 2.0, y: 2.0 };
        let p2 = Point { x: 1.0, y: 1.0 };
        let q2 = Point { x: 3.0, y: 3.0 };
        // Even though they overlap, our line_intersection returns None since they are collinear.
        assert!(line_segment_intersection(&p1, &q1, &p2, &q2).is_none());
    }

    #[test]
    fn test_line_intersection_nearly_parallel() {
        // Two segments nearly parallel resulting in no intersection.
        let p1 = Point { x: 0.0, y: 0.0 };
        let q1 = Point { x: 10.0, y: 0.0001 };
        let p2 = Point { x: 0.0, y: 1.0 };
        let q2 = Point { x: 10.0, y: 1.0001 };
        assert!(line_segment_intersection(&p1, &q1, &p2, &q2).is_none());
    }

    #[test]
    fn test_line_intersection_exact_intersection() {
        // Two segments with a clear intersection not at endpoints.
        let p1 = Point { x: 0.0, y: 0.0 };
        let q1 = Point { x: 2.0, y: 2.0 };
        let p2 = Point { x: 0.0, y: 2.0 };
        let q2 = Point { x: 2.0, y: 0.0 };
        if let Some(ip) = line_segment_intersection(&p1, &q1, &p2, &q2) {
            assert!((ip.x - 1.0).abs() < std::f32::EPSILON);
            assert!((ip.y - 1.0).abs() < std::f32::EPSILON);
        } else {
            panic!("Expected intersection at (1.0, 1.0)");
        }
    }

    #[test]
    fn test_intersecting_segments() {
        // (0,0)-(1,1) and (0,1)-(1,0) intersect.
        let p = Point { x: 0.0, y: 0.0 };
        let q = Point { x: 1.0, y: 1.0 };
        let a = Point { x: 0.0, y: 1.0 };
        let b = Point { x: 1.0, y: 0.0 };
        assert!(do_intersect(&p, &q, &a, &b));
    }

    #[test]
    fn test_non_intersecting_segments() {
        // (0,0)-(0.5,0.5) and (2,0)-(3,-1) do not intersect.
        let p = Point { x: 0.0, y: 0.0 };
        let q = Point { x: 0.5, y: 0.5 };
        let a = Point { x: 2.0, y: 0.0 };
        let b = Point { x: 3.0, y: -1.0 };
        assert!(!do_intersect(&p, &q, &a, &b));
    }

    #[test]
    fn test_collinear_but_disjoint() {
        // Collinear segments without overlapping.
        let p = Point { x: 0.0, y: 0.0 };
        let q = Point { x: 1.0, y: 1.0 };
        let a = Point { x: 2.0, y: 2.0 };
        let b = Point { x: 3.0, y: 3.0 };
        assert!(!do_intersect(&p, &q, &a, &b));
    }

    #[test]
    fn test_sharing_endpoint() {
        // Segments sharing the endpoint (1.0, 1.0)
        let p = Point { x: 0.0, y: 0.0 };
        let q = Point { x: 1.0, y: 1.0 };
        let a = Point { x: 1.0, y: 1.0 };
        let b = Point { x: 2.0, y: 0.0 };
        assert!(do_intersect(&p, &q, &a, &b));
    }
}