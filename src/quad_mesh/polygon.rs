use crate::utils::intersection::line_segment_intersection;
use super::{aabb::AABB, point::Point};

#[derive(Debug, Clone)]
pub struct Polygon {
    // Vertex buffer containing all points
    pub vertex_buffer: Vec<Point>,
    pub vertices: Vec<usize>,
    pub holes: Vec<Polygon>,
}

#[derive(Debug)]
pub enum PolygonError {
    NotEnoughVertices,
    SelfIntersecting,
    InvalidHole, // new variant for hole validation
}

impl Polygon {
    // Modified to return a Result for error handling
    pub fn new(vertex_buffer: Vec<Point>, vertices: Vec<usize>) -> Result<Self, PolygonError> {
        // Ensure the polygon has at least 3 vertices
        if vertices.len() < 3 {
            return Err(PolygonError::NotEnoughVertices);
        }

        // Extract polygon vertices based on the indices
        let pts: Vec<Point> = vertices.iter().map(|&i| vertex_buffer[i].clone()).collect();

        // Check for self-intersections
        if polygon_is_self_intersecting(&pts) {
            return Err(PolygonError::SelfIntersecting);
        }

        Ok(Self {
            vertex_buffer,
            vertices,
            holes: Vec::new(),
        })
    }

    pub fn new_rect(x: f32, y: f32, w: f32, h: f32) -> Self {
        let vertex_buffer = vec![
            Point { x, y },
            Point { x: x + w, y },
            Point { x: x + w, y: y + h },
            Point { x, y: y + h },
        ];
        let vertices = vec![0, 1, 2, 3];
        // Unwrap since rectangle always provides 4 vertices and is non-self-intersecting
        Self::new(vertex_buffer, vertices).unwrap()
    }

    pub fn new_polygon(center: Point, radius: f32, n: usize, start_angle: f32) -> Self {
        let mut vertex_buffer = Vec::new();
        let mut vertices = Vec::new();
        for i in 0..n {
            let theta = (i as f32) * std::f32::consts::TAU / (n as f32) + start_angle;
            let x = center.x + radius * theta.cos();
            let y = center.y + radius * theta.sin();
            vertex_buffer.push(Point { x, y });
            vertices.push(i);
        }
        // Unwrap since polygon always provides n vertices and is non-self-intersecting
        Self::new(vertex_buffer, vertices).unwrap()
    }

    pub fn add_hole(&mut self, hole: Polygon) -> Result<(), PolygonError> {
        // Validate that every vertex of the hole is inside the parent polygon.
        for &idx in &hole.vertices {
            let pt = &hole.vertex_buffer[idx];
            if !self.contains_point(pt) {
                return Err(PolygonError::InvalidHole);
            }
        }
        self.holes.push(hole);
        Ok(())
    }

    pub fn contains_point(&self, p: &Point) -> bool {
        // Standard ray-casting algorithm.
        let mut count = 0;
        for i in 0..self.vertices.len() {
            let j = (i + 1) % self.vertices.len();
            let a = &self.vertex_buffer[self.vertices[i]];
            let b = &self.vertex_buffer[self.vertices[j]];
            if (a.y > p.y) != (b.y > p.y) {
                let x_intersect = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
                if p.x < x_intersect {
                    count += 1;
                }
            }
        }
        let inside = count % 2 == 1;
        if !inside {
            return false;
        }
        // If point lies in the outer, ensure it is not inside any hole.
        for hole in &self.holes {
            if hole.contains_point(p) {
                return false;
            }
        }
        true
    }

    pub fn intersects_aabb(&self, other: &AABB) -> bool {
        // Check if any point of the AABB is inside the polygon
        if self.contains_point(&other.top_left())
            || self.contains_point(&other.top_right())
            || self.contains_point(&other.bottom_left())
            || self.contains_point(&other.bottom_right())
            || self.contains_point(&other.center)
        {
            return true;
        }

        false
    }

    pub fn edges_intersect_aabb(&self, other: &AABB) -> bool {
        for (a, b) in self.edges() {
            if other.contains(a) || other.contains(b) {
                return true;
            }
            if other.intersects_segment(&a, &b) {
                return true;
            }
        }

        for hole in &self.holes {
            if hole.edges_intersect_aabb(other) {
                return true;
            }
        }

        false
    }

    fn is_point_on_segment(p: &Point, a: &Point, b: &Point) -> bool {
        let cross = (b.y - a.y) * (p.x - a.x) - (b.x - a.x) * (p.y - a.y);
        if cross.abs() > std::f32::EPSILON {
            return false;
        }
        let min_x = a.x.min(b.x);
        let max_x = a.x.max(b.x);
        let min_y = a.y.min(b.y);
        let max_y = a.y.max(b.y);
        p.x >= min_x - std::f32::EPSILON
            && p.x <= max_x + std::f32::EPSILON
            && p.y >= min_y - std::f32::EPSILON
            && p.y <= max_y + std::f32::EPSILON
    }

    pub fn bounding_box(&self) -> AABB {
        let min_x = self
            .vertex_buffer
            .iter()
            .map(|p| p.x)
            .fold(std::f32::INFINITY, f32::min);
        let max_x = self
            .vertex_buffer
            .iter()
            .map(|p| p.x)
            .fold(std::f32::NEG_INFINITY, f32::max);
        let min_y = self
            .vertex_buffer
            .iter()
            .map(|p| p.y)
            .fold(std::f32::INFINITY, f32::min);
        let max_y = self
            .vertex_buffer
            .iter()
            .map(|p| p.y)
            .fold(std::f32::NEG_INFINITY, f32::max);
        let center = Point {
            x: (min_x + max_x) / 2.0,
            y: (min_y + max_y) / 2.0,
        };
        let half_width = (max_x - min_x) / 2.0;
        let half_height = (max_y - min_y) / 2.0;
        AABB::new(center, half_width, half_height)
    }

    pub fn bounding_square(&self) -> AABB {
        let bbox = self.bounding_box();
        let max_dim = bbox.width().max(bbox.height());
        AABB::new(bbox.center, max_dim / 2.0, max_dim / 2.0)
    }

    pub fn edges(&self) -> Vec<(Point, Point)> {
        self.vertices
            .iter()
            .map(|&i| {
                let a = self.vertex_buffer[i];
                let b = self.vertex_buffer[(i + 1) % self.vertices.len()];
                (a, b)
            })
            .collect()
    }
}

// Helper closure used within contains_point() to check holes for points on edge.
fn false_if_in_hole(polygon: &Polygon, p: &Point) -> bool {
    for hole in &polygon.holes {
        if hole.contains_point(p) {
            return false;
        }
    }
    true
}

// Check if a polygon (given ordered points) is self-intersecting
fn polygon_is_self_intersecting(pts: &[Point]) -> bool {
    let n = pts.len();
    if n < 4 {
        return false;
    } // triangle cannot intersect

    for i in 0..n {
        let p1 = &pts[i];
        let q1 = &pts[(i + 1) % n];
        for j in (i + 1)..n {
            // Skip adjacent edges and same edge
            if j == i || (j + 1) % n == i || (i + 1) % n == j {
                continue;
            }
            let p2 = &pts[j];
            let q2 = &pts[(j + 1) % n];
            if line_segment_intersection(p1, q1, p2, q2).is_some() {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line() {
        // A line is not a polygon
        let vertex_buffer = vec![Point { x: 0.0, y: 0.0 }, Point { x: 1.0, y: 1.0 }];
        let vertices = vec![0, 1];
        let polygon = Polygon::new(vertex_buffer, vertices);
        assert!(matches!(polygon, Err(PolygonError::NotEnoughVertices)));
    }

    #[test]
    fn test_non_intersecting_polygon() {
        // A simple convex polygon (square)
        let vertex_buffer = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 0.0, y: 1.0 },
        ];
        let vertices = vec![0, 1, 2, 3];
        let polygon = Polygon::new(vertex_buffer, vertices);
        assert!(polygon.is_ok());
    }

    #[test]
    fn test_self_intersecting_polygon() {
        // A bowtie polygon, which is self-intersecting
        let vertex_buffer = vec![
            Point { x: 0.0, y: 0.0 }, // 0
            Point { x: 1.0, y: 1.0 }, // 1
            Point { x: 0.0, y: 1.0 }, // 2
            Point { x: 1.0, y: 0.0 }, // 3
        ];
        let vertices = vec![0, 1, 2, 3];
        let polygon = Polygon::new(vertex_buffer, vertices);
        assert!(matches!(polygon, Err(PolygonError::SelfIntersecting)));
    }

    #[test]
    fn test_triangle() {
        // Minimal valid triangle
        let vertex_buffer = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 0.0 },
            Point { x: 0.0, y: 1.0 },
        ];
        let vertices = vec![0, 1, 2];
        let polygon = Polygon::new(vertex_buffer, vertices);
        assert!(polygon.is_ok());
    }

    #[test]
    fn test_concave_polygon() {
        // A concave, non-self-intersecting polygon
        // Points are: (0,0), (4,0), (4,3), (2,1), (0,3)
        let vertex_buffer = vec![
            Point { x: 0.0, y: 0.0 }, // 0
            Point { x: 4.0, y: 0.0 }, // 1
            Point { x: 4.0, y: 3.0 }, // 2
            Point { x: 2.0, y: 1.0 }, // 3 (concave vertex)
            Point { x: 0.0, y: 3.0 }, // 4
        ];
        let vertices = vec![0, 1, 2, 3, 4];
        let polygon = Polygon::new(vertex_buffer, vertices);
        assert!(polygon.is_ok());
    }

    #[test]
    fn test_complex_self_intersecting_polygon() {
        // A self-intersecting polygon (a five-pointed star shape)
        // The pattern: 0 -> 2 -> 4 -> 1 -> 3 -> 0 forms a star and is self-intersecting.
        let vertex_buffer = vec![
            Point { x: -1.0, y: -1.0 }, // 0
            Point { x: 1.0, y: -1.0 },  // 1
            Point { x: -1.0, y: 0.0 },  // 2
            Point { x: 1.0, y: 0.0 },   // 3
            Point { x: 0.0, y: 1.0 },   // 4
        ];
        // Order that produces self-intersection
        let vertices = vec![0, 3, 2, 1, 4];
        let polygon = Polygon::new(vertex_buffer, vertices);
        assert!(matches!(polygon, Err(PolygonError::SelfIntersecting)));
    }

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
    fn test_point_in_polygon_inside() {
        let vertex_buffer = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 4.0, y: 0.0 },
            Point { x: 4.0, y: 4.0 },
            Point { x: 0.0, y: 4.0 },
        ];
        let vertices = vec![0, 1, 2, 3];
        let polygon = Polygon::new(vertex_buffer, vertices).unwrap();
        let p_inside = Point { x: 2.0, y: 2.0 };
        assert!(polygon.contains_point(&p_inside));
    }

    #[test]
    fn test_point_in_polygon_outside() {
        let vertex_buffer = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 4.0, y: 0.0 },
            Point { x: 4.0, y: 4.0 },
            Point { x: 0.0, y: 4.0 },
        ];
        let vertices = vec![0, 1, 2, 3];
        let polygon = Polygon::new(vertex_buffer, vertices).unwrap();
        let p_outside = Point { x: 5.0, y: 5.0 };
        assert!(!polygon.contains_point(&p_outside));
    }

    #[test]
    fn test_contains_point_with_hole() {
        // Outer polygon: square from (0,0) to (10,10)
        let outer_vb = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 10.0, y: 0.0 },
            Point { x: 10.0, y: 10.0 },
            Point { x: 0.0, y: 10.0 },
        ];
        let outer_vertices = vec![0, 1, 2, 3];
        let mut outer = Polygon::new(outer_vb, outer_vertices).unwrap();

        // Hole: square from (3,3) to (7,7)
        let hole_vb = vec![
            Point { x: 3.0, y: 3.0 },
            Point { x: 7.0, y: 3.0 },
            Point { x: 7.0, y: 7.0 },
            Point { x: 3.0, y: 7.0 },
        ];
        let hole_vertices = vec![0, 1, 2, 3];
        let hole = Polygon::new(hole_vb, hole_vertices).unwrap();
        outer.add_hole(hole).unwrap();

        // Point inside the hole: should be rejected.
        let p_inside_hole = Point { x: 5.0, y: 5.0 };
        assert!(
            !outer.contains_point(&p_inside_hole),
            "Point inside the hole must be considered outside overall"
        );

        // Point in outer polygon but outside the hole: should be accepted.
        let p_in_outer = Point { x: 2.0, y: 2.0 };
        assert!(
            outer.contains_point(&p_in_outer),
            "Point in outer polygon and outside the hole must be considered inside overall"
        );

        // Point on the boundary of the hole: treated as inside the hole, thus overall false.
        let p_on_hole_edge = Point { x: 3.0, y: 5.0 };
        assert!(
            !outer.contains_point(&p_on_hole_edge),
            "Point on the hole's edge must be considered outside overall"
        );
    }

    #[test]
    fn test_add_valid_hole() {
        // Outer polygon: square from (0,0) to (10,10)
        let outer_vb = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 10.0, y: 0.0 },
            Point { x: 10.0, y: 10.0 },
            Point { x: 0.0, y: 10.0 },
        ];
        let outer_vertices = vec![0, 1, 2, 3];
        let mut outer = Polygon::new(outer_vb, outer_vertices).unwrap();

        // Hole: square from (3,3) to (7,7)
        let hole_vb = vec![
            Point { x: 3.0, y: 3.0 },
            Point { x: 7.0, y: 3.0 },
            Point { x: 7.0, y: 7.0 },
            Point { x: 3.0, y: 7.0 },
        ];
        let hole_vertices = vec![0, 1, 2, 3];
        let hole = Polygon::new(hole_vb, hole_vertices).unwrap();

        // Valid hole should be added successfully.
        assert!(outer.add_hole(hole).is_ok());
    }

    #[test]
    fn test_add_valid_hole2() {
        // Outer polygon: square from (0,0) to (10,10)
        let mut outer = Polygon::new_rect(0.0, 0.0, 10.0, 10.0);

        // Hole: square from (3,3) to (7,7)
        let hole = Polygon::new_rect(3.0, 3.0, 4.0, 4.0);

        // Valid hole should be added successfully.
        assert!(outer.add_hole(hole).is_ok());
    }

    #[test]
    fn test_add_invalid_hole() {
        // Outer polygon: square from (0,0) to (10,10)
        let outer_vb = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 10.0, y: 0.0 },
            Point { x: 10.0, y: 10.0 },
            Point { x: 0.0, y: 10.0 },
        ];
        let outer_vertices = vec![0, 1, 2, 3];
        let mut outer = Polygon::new(outer_vb, outer_vertices).unwrap();

        // Hole: square with one vertex outside the outer polygon.
        let hole_vb = vec![
            Point { x: 3.0, y: 3.0 },
            Point { x: 11.0, y: 3.0 }, // outside the outer polygon
            Point { x: 11.0, y: 7.0 },
            Point { x: 3.0, y: 7.0 },
        ];
        let hole_vertices = vec![0, 1, 2, 3];
        let hole = Polygon::new(hole_vb, hole_vertices).unwrap();

        // Adding invalid hole should return an error.
        assert!(matches!(
            outer.add_hole(hole),
            Err(PolygonError::InvalidHole)
        ));
    }
}
