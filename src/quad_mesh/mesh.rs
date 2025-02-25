use crate::quad_mesh::{quad_tree::QuadTree, point::Point};
use crate::quad_mesh::polygon::Polygon;

use super::aabb::AABB;

/// Represents a quadrilateral defined by its four vertices.
pub struct Quad {
    pub bottom_left: Point,
    pub bottom_right: Point,
    pub top_right: Point,
    pub top_left: Point,
}

impl Quad {
    /// Constructs a new `Quad` given the center and half dimensions.
    pub fn new_rect(center: &Point, half_width: f32, half_height: f32) -> Self {
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

/// Mesh data structure based on quad tree mesh in structure-of-arrays format.
pub struct Mesh {
    // Cell centers
    pub cell_centers_x: Vec<f32>,
    pub cell_centers_y: Vec<f32>,
    // Cell half sizes
    pub cell_half_width: Vec<f32>,
    pub cell_half_height: Vec<f32>,
    // Neighbors for east face: ranges and flat list of indices.
    pub neighbors_east_range: Vec<(usize, usize)>,
    pub neighbors_east_indexes: Vec<usize>,
    // Neighbors for west face.
    pub neighbors_west_range: Vec<(usize, usize)>,
    pub neighbors_west_indexes: Vec<usize>,
    // Neighbors for north face.
    pub neighbors_north_range: Vec<(usize, usize)>,
    pub neighbors_north_indexes: Vec<usize>,
    // Neighbors for south face.
    pub neighbors_south_range: Vec<(usize, usize)>,
    pub neighbors_south_indexes: Vec<usize>,
}

pub struct Cell<'a> {
    pub center: Point,
    pub quad: Quad,
    pub neighbors: Neighbors<'a>
}

pub struct Neighbors<'a> {
    pub east: &'a [usize],
    pub west: &'a [usize],
    pub north: &'a [usize],
    pub south: &'a [usize],
}

impl Mesh {
    /// Builds a Mesh from a provided quad tree that is filtered based on the computational domain
    /// defined by polygon.
    pub fn from_quad_tree(root: &QuadTree, polygon: &Polygon) -> Self {
        // First, gather all leaf cells from the quad tree.
        let mut cells = Vec::new();
        gather_leaves(root, &mut cells);
        // Filter out cells where neither the center nor any vertex is inside the polygon.
        let valid_cells: Vec<TmpCell> = cells
            .into_iter()
            .filter(|cell| {
                // Check the cell center.
                let center_inside = polygon.contains_point(&cell.center);
                
                // Compute cell vertices.
                let left = cell.center.x - cell.half_width;
                let right = cell.center.x + cell.half_width;
                let bottom = cell.center.y - cell.half_height;
                let top = cell.center.y + cell.half_height;
                
                // Check if any vertex is inside the polygon.
                let vertex_inside = polygon.contains_point(&Point { x: left, y: bottom })
                    || polygon.contains_point(&Point { x: left, y: top })
                    || polygon.contains_point(&Point { x: right, y: bottom })
                    || polygon.contains_point(&Point { x: right, y: top });
                
                center_inside || vertex_inside
            })
            .collect();

        let num_cells = valid_cells.len();
        // Preallocate arrays.
        let mut cell_centers_x = Vec::with_capacity(num_cells);
        let mut cell_centers_y = Vec::with_capacity(num_cells);
        let mut cell_half_width = Vec::with_capacity(num_cells);
        let mut cell_half_height = Vec::with_capacity(num_cells);
        // Store boundaries for neighbor search: (xmin, xmax, ymin, ymax).
        let mut boundaries: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(num_cells);

        for cell in &valid_cells {
            cell_centers_x.push(cell.center.x);
            cell_centers_y.push(cell.center.y);
            cell_half_width.push(cell.half_width);
            cell_half_height.push(cell.half_height);
            let xmin = cell.center.x - cell.half_width;
            let xmax = cell.center.x + cell.half_width;
            let ymin = cell.center.y - cell.half_height;
            let ymax = cell.center.y + cell.half_height;
            boundaries.push((xmin, xmax, ymin, ymax));
        }

        // Temporary storage for neighbors as vectors-of-vectors.
        let mut temp_neighbors_east = vec![Vec::new(); num_cells];
        let mut temp_neighbors_west = vec![Vec::new(); num_cells];
        let mut temp_neighbors_north = vec![Vec::new(); num_cells];
        let mut temp_neighbors_south = vec![Vec::new(); num_cells];

        // Tolerance for boundary comparisons.
        let eps = 1e-6;
        // Use a simple nested loop to assign neighbors.
        for i in 0..num_cells {
            let (xmin_i, xmax_i, ymin_i, ymax_i) = boundaries[i];
            for j in 0..num_cells {
                if i == j {
                    continue;
                }
                let (xmin_j, xmax_j, ymin_j, ymax_j) = boundaries[j];
                // East: cell j touches east boundary of cell i.
                if (xmin_j - xmax_i).abs() < eps && (ymin_i < ymax_j && ymax_i > ymin_j) {
                    temp_neighbors_east[i].push(j);
                }
                // West: cell j touches west boundary of cell i.
                if (xmax_j - xmin_i).abs() < eps && (ymin_i < ymax_j && ymax_i > ymin_j) {
                    temp_neighbors_west[i].push(j);
                }
                // North: cell j touches top (north) boundary of cell i.
                if (ymin_j - ymax_i).abs() < eps && (xmin_i < xmax_j && xmax_i > xmin_j) {
                    temp_neighbors_north[i].push(j);
                }
                // South: cell j touches bottom (south) boundary of cell i.
                if (ymax_j - ymin_i).abs() < eps && (xmin_i < xmax_j && xmax_i > xmin_j) {
                    temp_neighbors_south[i].push(j);
                }
            }
        }

        // Flatten temporary neighbor vectors and compute ranges.
        let mut neighbors_east_indexes = Vec::new();
        let mut neighbors_east_range = Vec::with_capacity(num_cells);
        for neighs in &temp_neighbors_east {
            let start = neighbors_east_indexes.len();
            neighbors_east_indexes.extend(neighs.iter().cloned());
            let end = neighbors_east_indexes.len();
            neighbors_east_range.push((start, end));
        }

        let mut neighbors_west_indexes = Vec::new();
        let mut neighbors_west_range = Vec::with_capacity(num_cells);
        for neighs in &temp_neighbors_west {
            let start = neighbors_west_indexes.len();
            neighbors_west_indexes.extend(neighs.iter().cloned());
            let end = neighbors_west_indexes.len();
            neighbors_west_range.push((start, end));
        }

        let mut neighbors_north_indexes = Vec::new();
        let mut neighbors_north_range = Vec::with_capacity(num_cells);
        for neighs in &temp_neighbors_north {
            let start = neighbors_north_indexes.len();
            neighbors_north_indexes.extend(neighs.iter().cloned());
            let end = neighbors_north_indexes.len();
            neighbors_north_range.push((start, end));
        }

        let mut neighbors_south_indexes = Vec::new();
        let mut neighbors_south_range = Vec::with_capacity(num_cells);
        for neighs in &temp_neighbors_south {
            let start = neighbors_south_indexes.len();
            neighbors_south_indexes.extend(neighs.iter().cloned());
            let end = neighbors_south_indexes.len();
            neighbors_south_range.push((start, end));
        }

        Mesh {
            cell_centers_x,
            cell_centers_y,
            cell_half_width,
            cell_half_height,
            neighbors_east_range,
            neighbors_east_indexes,
            neighbors_west_range,
            neighbors_west_indexes,
            neighbors_north_range,
            neighbors_north_indexes,
            neighbors_south_range,
            neighbors_south_indexes,
        }
    }

    pub fn visit_cell<F>(&self, cell_index: usize, mut visit: F)
    where
        F: FnMut(&Cell),
    {
        // Create a Point from the stored cell center coordinates.
        let center = Point {
            x: self.cell_centers_x[cell_index],
            y: self.cell_centers_y[cell_index],
        };

        // Retrieve half dimensions (note: these are already stored per cell).
        let half_width = self.cell_half_width[cell_index];
        let half_height = self.cell_half_height[cell_index];
        // Construct the explicit quad from the cell center and half dimensions.
        let quad = Quad::new_rect(&center, half_width, half_height);

        // Retrieve neighbors for each face by slicing the neighbor indexes using the appropriate range.
        let (east_start, east_end) = self.neighbors_east_range[cell_index];
        let east_neighbors = &self.neighbors_east_indexes[east_start..east_end];

        let (west_start, west_end) = self.neighbors_west_range[cell_index];
        let west_neighbors = &self.neighbors_west_indexes[west_start..west_end];

        let (north_start, north_end) = self.neighbors_north_range[cell_index];
        let north_neighbors = &self.neighbors_north_indexes[north_start..north_end];

        let (south_start, south_end) = self.neighbors_south_range[cell_index];
        let south_neighbors = &self.neighbors_south_indexes[south_start..south_end];

        // Bundle neighbor slices into the Neighbors struct.
        let neighbors = Neighbors {
            east: east_neighbors,
            west: west_neighbors,
            north: north_neighbors,
            south: south_neighbors,
        };

        let cell = Cell {
            center,
            quad,
            neighbors,
        };

        // Call the closure with the information about the cell.
        visit(&cell);
    }

    pub fn visit_all_cells<F>(&self, mut visit: F)
    where
        F: FnMut(&Cell),
    {
        let num_cells = self.cell_centers_x.len();
        for i in 0..num_cells {
            // For each cell, use the existing visit_cell method.
            // Here, we pass a small closure that simply forwards the cell to `visit`.
            self.visit_cell(i, |cell| visit(cell));
        }
    }

    pub fn full_bounding_box(&self) -> AABB {
        // Early exit if there are no cells.
        if self.cell_centers_x.is_empty() {
            return AABB {
                center: Point { x: 0.0, y: 0.0 },
                half_width: 0.0,
                half_height: 0.0,
            };
        }

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        self.visit_all_cells(|cell| {
            // Compare all four vertices in the quad.
            let vertices = cell.quad.vertices();
            let cell_min_x = vertices.iter().fold(f32::INFINITY, |acc, v| acc.min(v.x));
            let cell_max_x = vertices.iter().fold(f32::NEG_INFINITY, |acc, v| acc.max(v.x));
            let cell_min_y = vertices.iter().fold(f32::INFINITY, |acc, v| acc.min(v.y));
            let cell_max_y = vertices.iter().fold(f32::NEG_INFINITY, |acc, v| acc.max(v.y));

            if cell_min_x < min_x {
                min_x = cell_min_x;
            }
            if cell_max_x > max_x {
                max_x = cell_max_x;
            }
            if cell_min_y < min_y {
                min_y = cell_min_y;
            }
            if cell_max_y > max_y {
                max_y = cell_max_y;
            }
        });

        // Compute overall center and half dimensions of the bounding box.
        let center_x = 0.5 * (min_x + max_x);
        let center_y = 0.5 * (min_y + max_y);
        let half_width = 0.5 * (max_x - min_x);
        let half_height = 0.5 * (max_y - min_y);

        AABB {
            center: Point { x: center_x, y: center_y },
            half_width,
            half_height,
        }
    }
}

// Helper cell information extracted from a quad tree leaf.
struct TmpCell {
    center: Point,
    half_width: f32,
    half_height: f32,
}

/// Recursively gather quad tree leaves.
fn gather_leaves(node: &QuadTree, cells: &mut Vec<TmpCell>) {
    if node.children.is_none() {
        cells.push(TmpCell {
            center: node.boundary.center,
            half_width: node.boundary.half_width,
            half_height: node.boundary.half_height,
        });
        return;
    }
    if let Some(ref children) = node.children {
        for child in children.iter() {
            gather_leaves(child, cells);
        }
    }
}