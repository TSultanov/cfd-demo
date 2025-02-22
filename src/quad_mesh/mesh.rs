// Quad-tree mesh data structure

use super::{aabb::AABB, point::Point, polygon::Polygon};

#[derive(Clone)]
pub struct Cell {
    pub boundary: AABB,
    pub children: Option<Box<[Cell; 4]>>,
}

impl Cell {
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }
}

pub fn tesselate(polygon: &Polygon, feature_size: f32) -> Cell {
    tesselate_impl(polygon, &polygon.bounding_square(), feature_size)
}

// New function: adaptive quadtree meshing based on polygon edges.
fn tesselate_impl(polygon: &Polygon, boundary: &AABB, feature_size: f32) -> Cell {
    let cell_size = boundary.width().min(boundary.height());

    let all_outside = !polygon.intersects_aabb(boundary);
    let intersects_edges = polygon.edges_intersect_aabb(boundary);

    // Stop subdividing if cell is not in polygon, too small, or no remaining depth.
    if cell_size <= feature_size || all_outside || !intersects_edges {
        return Cell {
            boundary: *boundary,
            children: None,
        };
    }

    let new_half_width = boundary.half_width / 2.0;
    let new_half_height = boundary.half_height / 2.0;
    let cx = boundary.center.x;
    let cy = boundary.center.y;

    // Subdivide into 4 quadrants.
    let cells = [
        tesselate_impl(
            polygon,
            &AABB::new(
                Point {
                    x: cx - new_half_width,
                    y: cy - new_half_height,
                },
                new_half_width,
                new_half_height,
            ),
            feature_size,
        ),
        tesselate_impl(
            polygon,
            &AABB::new(
                Point {
                    x: cx + new_half_width,
                    y: cy - new_half_height,
                },
                new_half_width,
                new_half_height,
            ),
            feature_size,
        ),
        tesselate_impl(
            polygon,
            &AABB::new(
                Point {
                    x: cx - new_half_width,
                    y: cy + new_half_height,
                },
                new_half_width,
                new_half_height,
            ),
            feature_size,
        ),
        tesselate_impl(
            polygon,
            &AABB::new(
                Point {
                    x: cx + new_half_width,
                    y: cy + new_half_height,
                },
                new_half_width,
                new_half_height,
            ),
            feature_size,
        ),
    ];

    Cell {
        boundary: *boundary,
        children: Some(Box::new(cells)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quad_mesh::point::Point;
    use crate::quad_mesh::polygon::Polygon;

    #[test]
    fn test_tesselate_rect_one_sub() {
        // Use the rectangle polygon: its bounding box matches the polygon.
        let polygon = Polygon::new_rect(0.0, 0.0, 10.0, 10.0);
        let cell = tesselate(&polygon, 5.0);
        // Should be homogeneous, so no subdivision.
        assert!(cell
            .children.iter().all(|c| c.is_none()));
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
            let pt = Point {
                x: center.x + 4.0 * theta.cos(),
                y: center.y + 4.0 * theta.sin(),
            };
            vertex_buffer.push(pt);
            vertices.push(i);
        }
        let polygon = Polygon::new(vertex_buffer, vertices).unwrap();
        let cell = tesselate(&polygon, 0.5);
        // Expect subdivision because the bounding box is larger than the circle.
        assert!(!cell.children.is_empty());
    }
}
