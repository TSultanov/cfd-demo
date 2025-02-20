// Quad-tree mesh data structure

use super::{aabb::AABB, point::Point, polygon::Polygon};

pub struct Cell {
    pub boundary: AABB,
    pub children: Option<Box<[Cell; 4]>>,
}

pub fn tesselate(polygon: &Polygon, depth: usize, feature_size: f32) -> Cell {
    tesselate_impl(polygon, &polygon.bounding_box(), depth, feature_size)
}

// New function: adaptive quadtree meshing based on polygon edges.
fn tesselate_impl(polygon: &Polygon, boundary: &AABB, depth: usize, feature_size: f32) -> Cell {
    let cell_width = boundary.half_width * 2.0;
    // Get cell corner points.
    let tl = boundary.top_left();
    let tr = boundary.top_right();
    let bl = boundary.bottom_left();
    let br = boundary.bottom_right();
    let all_inside = polygon.contains_point(&tl)
        && polygon.contains_point(&tr)
        && polygon.contains_point(&bl)
        && polygon.contains_point(&br);

    let all_outside = !polygon.intersects(&boundary.to_polygon());

    // Stop subdividing if cell is homogeneous, too small, or no remaining depth.
    if depth == 0 || cell_width <= feature_size || all_inside || all_outside {
        return Cell { boundary: *boundary, children: None };
    }

    let new_half_width = boundary.half_width / 2.0;
    let new_half_height = boundary.half_height / 2.0;
    let cx = boundary.center.x;
    let cy = boundary.center.y;

    // Subdivide into 4 quadrants.
    let cells = [
        tesselate_impl(
            polygon,
            &AABB::new(Point { x: cx - new_half_width, y: cy - new_half_height }, new_half_width, new_half_height),
            depth - 1,
            feature_size,
        ),
        tesselate_impl(
            polygon,
            &AABB::new(Point { x: cx + new_half_width, y: cy - new_half_height }, new_half_width, new_half_height),
            depth - 1,
            feature_size,
        ),
        tesselate_impl(
            polygon,
            &AABB::new(Point { x: cx - new_half_width, y: cy + new_half_height }, new_half_width, new_half_height),
            depth - 1,
            feature_size,
        ),
        tesselate_impl(
            polygon,
            &AABB::new(Point { x: cx + new_half_width, y: cy + new_half_height }, new_half_width, new_half_height),
            depth - 1,
            feature_size,
        ),
    ];

    Cell { boundary: *boundary, children: Some(Box::new(cells)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quad_mesh::point::Point;
    use crate::quad_mesh::polygon::Polygon;

    #[test]
    fn test_tesselate_rect_leaf() {
        // Use the rectangle polygon: its bounding box matches the polygon.
        let polygon = Polygon::new_rect(0.0, 0.0, 10.0, 10.0);
        let cell = tesselate(&polygon, 0, 0.5);
        // Should be homogeneous, so no subdivision.
        assert!(cell.children.is_none());
    }

    #[test]
    fn test_tesselate_rect_one_sub() {
        // Use the rectangle polygon: its bounding box matches the polygon.
        let polygon = Polygon::new_rect(0.0, 0.0, 10.0, 10.0);
        let cell = tesselate(&polygon, 1, 0.5);
        // Should be homogeneous, so no subdivision.
        assert!(cell.children.is_some_and(|c| c.iter().all(|c| c.children.is_none())));
    }

    #[test]
    fn test_tesselate_octagon_subdivision() {
        // Create an octagon centered at (5,5) with radius 4.
        let center = Point { x: 5.0, y: 5.0 };
        let mut vertex_buffer = Vec::new();
        let mut vertices = Vec::new();
        let n = 8;
        for i in 0..n {
            let theta = (i as f32) * std::f32::consts::TAU / (n as f32);
            let pt = Point { x: center.x + 4.0 * theta.cos(), y: center.y + 4.0 * theta.sin() };
            vertex_buffer.push(pt);
            vertices.push(i);
        }
        let polygon = Polygon::new(vertex_buffer, vertices).unwrap();
        let cell = tesselate(&polygon, 5, 0.5);
        // Expect subdivision because the bounding box is larger than the circle.
        assert!(cell.children.is_some());
    }
}

