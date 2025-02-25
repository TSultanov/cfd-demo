use crate::quad_mesh::{point::Point, quad::Quad};

fn orientation(p: &Point, q: &Point, r: &Point) -> i32 {
    // Calculate the determinant (cross product)
    let val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if val.abs() < std::f64::EPSILON {
        0 // collinear
    } else if val > 0.0 {
        1 // clockwise
    } else {
        2 // counterclockwise
    }
}

fn on_segment(p: &Point, q: &Point, r: &Point) -> bool {
    (q.x <= p.x.max(r.x) + std::f64::EPSILON && q.x >= p.x.min(r.x) - std::f64::EPSILON) &&
        (q.y <= p.y.max(r.y) + std::f64::EPSILON && q.y >= p.y.min(r.y) - std::f64::EPSILON)
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
    // Check if the segments actually intersect by using the helper
    if !do_intersect(p, q, a, b) {
        return None;
    }
    
    let a1 = q.y - p.y;
    let b1 = p.x - q.x;
    let c1 = a1 * p.x + b1 * p.y;
    
    let a2 = b.y - a.y;
    let b2 = a.x - b.x;
    let c2 = a2 * a.x + b2 * a.y;
    
    let det = a1 * b2 - a2 * b1;
    if det.abs() < std::f64::EPSILON {
        return None; // Lines are parallel (or collinear) so no unique intersection point.
    }
    
    let x = (b2 * c1 - b1 * c2) / det;
    let y = (a1 * c2 - a2 * c1) / det;
    
    Some(Point { x, y })
}


/// Computes intersection points between an edge (defined by two points) and this quad.
/// Returns a vector of intersection points, which may contain 0, 1, or 2 points.
pub fn intersect_quad_edge(quad: &Quad, p1: &Point, p2: &Point) -> Vec<Point> {        
    let vertices = quad.vertices();
    let mut intersections = Vec::new();
    
    // Check intersection with each edge of the quad
    for i in 0..4 {
        let j = (i + 1) % 4; // Next vertex (wrapping around)
        let v1 = &vertices[i];
        let v2 = &vertices[j];
        
        // Special case: if the quad edge and the input edge are collinear,
        // compute the overlapping segment.
        if orientation(p1, p2, v1) == 0 && orientation(p1, p2, v2) == 0 {
            let d_x = p2.x - p1.x;
            let d_y = p2.y - p1.y;
            let norm = d_x * d_x + d_y * d_y;
            // If p1-p2 is degenerate, skip the collinear logic.
            if norm.abs() < std::f64::EPSILON {
                continue;
            }
            // Compute the projection parameters of v1 and v2 on the line p1->p2.
            let t_v1 = ((v1.x - p1.x) * d_x + (v1.y - p1.y) * d_y) / norm;
            let t_v2 = ((v2.x - p1.x) * d_x + (v2.y - p1.y) * d_y) / norm;
            // The overlapping interval on p1-p2 is from max(0.0, min(t_v1,t_v2))
            // to min(1.0, max(t_v1, t_v2))
            let t_start = t_v1.min(t_v2).max(0.0);
            let t_end = t_v1.max(t_v2).min(1.0);
            if t_start <= t_end + std::f64::EPSILON {
                let overlap_start = Point { x: p1.x + t_start * d_x, y: p1.y + t_start * d_y };
                let overlap_end   = Point { x: p1.x + t_end   * d_x, y: p1.y + t_end   * d_y };
                // Avoid duplicates (e.g. at corners)
                if !intersections.iter().any(|p: &Point| 
                    (p.x - overlap_start.x).abs() < std::f64::EPSILON && 
                    (p.y - overlap_start.y).abs() < std::f64::EPSILON
                ) {
                    intersections.push(overlap_start);
                }
                if !intersections.iter().any(|p: &Point| 
                    (p.x - overlap_end.x).abs() < std::f64::EPSILON && 
                    (p.y - overlap_end.y).abs() < std::f64::EPSILON
                ) {
                    intersections.push(overlap_end);
                }
                // Continue to the next quad edge.
                continue;
            }
        }
        
        if let Some(intersection) = line_segment_intersection(p1, p2, v1, v2) {
            // Check if this intersection is already in our list (avoid duplicates at corners)
            if !intersections.iter().any(|p: &Point| 
                (p.x - intersection.x).abs() < std::f64::EPSILON && 
                (p.y - intersection.y).abs() < std::f64::EPSILON
            ) {
                intersections.push(intersection);
            }
        }
    }
    
    intersections
}

#[cfg(test)]
mod test {
    use crate::quad_mesh::{point::Point, quad::Quad};
    use crate::utils::intersection::{do_intersect, line_segment_intersection, intersect_quad_edge};

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
            assert!((ip.x - 1.0).abs() < std::f64::EPSILON);
            assert!((ip.y - 1.0).abs() < std::f64::EPSILON);
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
            assert!((ip.x - 1.0).abs() < std::f64::EPSILON);
            assert!((ip.y - 1.0).abs() < std::f64::EPSILON);
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

    #[test]
    fn test_intersect_quad_edge_no_intersection() {
        // Edge completely outside the quad
        let quad = Quad::new_rect(&Point { x: 0.0, y: 0.0 }, 1.0, 1.0);
        let p1 = Point { x: -3.0, y: -3.0 };
        let p2 = Point { x: -2.0, y: -2.0 };
        
        let intersections = intersect_quad_edge(&quad, &p1, &p2);
        assert_eq!(intersections.len(), 0);
    }

    #[test]
    fn test_intersect_quad_edge_one_intersection() {
        // Edge intersects one edge of the quad
        let quad = Quad::new_rect(&Point { x: 0.0, y: 0.0 }, 1.0, 1.0);
        let p1 = Point { x: -2.0, y: 0.0 };
        let p2 = Point { x: 0.0, y: 0.0 };
        
        let intersections = intersect_quad_edge(&quad, &p1, &p2);
        assert_eq!(intersections.len(), 1);
        assert!((intersections[0].x - (-1.0)).abs() < std::f64::EPSILON);
        assert!((intersections[0].y - 0.0).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_intersect_quad_edge_two_intersections() {
        // Edge passes through the quad
        let quad = Quad::new_rect(&Point { x: 0.0, y: 0.0 }, 1.0, 1.0);
        let p1 = Point { x: -2.0, y: 0.0 };
        let p2 = Point { x: 2.0, y: 0.0 };
        
        let intersections = intersect_quad_edge(&quad, &p1, &p2);
        assert_eq!(intersections.len(), 2);
        
        // Sort intersections by x-coordinate for consistent testing
        let mut sorted = intersections.clone();
        sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        
        assert!((sorted[0].x - (-1.0)).abs() < std::f64::EPSILON);
        assert!((sorted[0].y - 0.0).abs() < std::f64::EPSILON);
        assert!((sorted[1].x - 1.0).abs() < std::f64::EPSILON);
        assert!((sorted[1].y - 0.0).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_intersect_quad_edge_through_vertex() {
        // Edge passes through a vertex of the quad
        let quad = Quad::new_rect(&Point { x: 0.0, y: 0.0 }, 1.0, 1.0);
        let p1 = Point { x: -2.0, y: -2.0 };
        let p2 = Point { x: 2.0, y: 2.0 };
        
        let intersections = intersect_quad_edge(&quad, &p1, &p2);
        assert_eq!(intersections.len(), 2);
        
        // Should intersect at bottom-left and top-right vertices
        let expected_points = [
            Point { x: -1.0, y: -1.0 },
            Point { x: 1.0, y: 1.0 }
        ];
        
        // Check that each expected point is found in the intersections
        for expected in &expected_points {
            assert!(intersections.iter().any(|p| 
                (p.x - expected.x).abs() < std::f64::EPSILON && 
                (p.y - expected.y).abs() < std::f64::EPSILON
            ));
        }
    }

    #[test]
    fn test_intersect_quad_edge_along_edge() {
        // Edge coincides with one edge of the quad
        let quad = Quad::new_rect(&Point { x: 0.0, y: 0.0 }, 1.0, 1.0);
        let p1 = Point { x: -1.0, y: 1.0 }; // top-left
        let p2 = Point { x: 1.0, y: 1.0 };  // top-right
        
        let intersections = intersect_quad_edge(&quad, &p1, &p2);
        assert_eq!(intersections.len(), 2);
        
        // Sort by x-coordinate
        let mut sorted = intersections.clone();
        sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        
        assert!((sorted[0].x - (-1.0)).abs() < std::f64::EPSILON);
        assert!((sorted[0].y - 1.0).abs() < std::f64::EPSILON);
        assert!((sorted[1].x - 1.0).abs() < std::f64::EPSILON);
        assert!((sorted[1].y - 1.0).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_intersect_quad_edge_inside_quad() {
        // Edge completely inside the quad should return no intersections
        let quad = Quad::new_rect(&Point { x: 0.0, y: 0.0 }, 1.0, 1.0);
        let p1 = Point { x: -0.5, y: -0.5 };
        let p2 = Point { x: 0.5, y: 0.5 };
        
        let intersections = intersect_quad_edge(&quad, &p1, &p2);
        assert_eq!(intersections.len(), 0);
    }

    #[test]
    fn test_intersect_quad_edge_diagonal() {
        // Edge passing diagonally through the quad
        let quad = Quad::new_rect(&Point { x: 0.0, y: 0.0 }, 1.0, 1.0);
        let p1 = Point { x: -2.0, y: -1.0 };
        let p2 = Point { x: 0.0, y: 1.0 };  // Changed to ensure it passes through the quad
        
        let intersections = intersect_quad_edge(&quad, &p1, &p2);
        assert_eq!(intersections.len(), 2);
        
        // The intersection points would be at (-1,1) and (0.5,2.5)
        let expected_points = [
            Point { x: -1.0, y: 0.0 },
            Point { x: 0.0, y: 1.0 }
        ];
        
        // Check that each expected point is found in the intersections
        for expected in &expected_points {
            assert!(intersections.iter().any(|p| 
                (p.x - expected.x).abs() < std::f64::EPSILON && 
                (p.y - expected.y).abs() < std::f64::EPSILON
            ));
        }
    }
}