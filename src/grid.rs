//! The spatial substrate: a typed 2-D cell grid that replaces the original
//! `Vec<u128>` occupancy bitmask.
//!
//! The old world held a single occupancy bit per cell in 128 column words,
//! which hard-capped the world at 128×128 and could not carry any per-cell
//! state beyond "occupied". This grid stores a real `Cell` per position, so the
//! width/height are parameters (the 128 cap is lifted) and later ecosystem
//! rungs can grow `Cell` with nutrient/biomass fields without touching the
//! agents. Row-major: cell `(x, y)` lives at `cells[y * width + x]`.
//!
//! A cell's occupant is the hecs [`Entity`] standing on it (if any), so later
//! interaction mechanics (predation, grazing) can ask *who* is in a cell — not
//! just whether it is full. "Blocked" for collision is `obstacle || occupant`.

use hecs::Entity;

/// One grid cell: a static obstacle flag plus the creature currently on it (if
/// any). "Blocked" for collision purposes means either — a wall or a creature
/// both block movement onto the cell.
#[derive(Clone, Default)]
pub struct Cell {
    pub obstacle: bool,
    pub occupant: Option<Entity>,
}

impl Cell {
    /// Whether movement onto this cell is blocked.
    #[inline]
    pub fn blocked(&self) -> bool {
        self.obstacle || self.occupant.is_some()
    }
}

/// A dense grid of [`Cell`]s. The occupancy/obstacle helpers replace the old
/// bit-twiddling world helpers one-for-one.
pub struct Grid {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<Cell>,
}

impl Grid {
    pub fn new(width: usize, height: usize) -> Grid {
        Grid {
            width,
            height,
            cells: vec![Cell::default(); width * height],
        }
    }

    #[inline]
    fn idx(&self, x: u32, y: u32) -> usize {
        y as usize * self.width + x as usize
    }

    /// Is the cell blocked (obstacle or occupied)? This is the occupancy test
    /// the collision resolver and the directional sensors read.
    #[inline]
    pub fn blocked(&self, (x, y): (u32, u32)) -> bool {
        self.cells[self.idx(x, y)].blocked()
    }

    /// Mark `pos` as occupied by `entity` (a creature moved in / spawned there).
    pub fn set_occupant(&mut self, (x, y): (u32, u32), entity: Entity) {
        let i = self.idx(x, y);
        self.cells[i].occupant = Some(entity);
    }

    /// Clear the occupant of `pos` (a creature moved out / was culled).
    pub fn clear_occupant(&mut self, (x, y): (u32, u32)) {
        let i = self.idx(x, y);
        self.cells[i].occupant = None;
    }

    /// Stamp a static obstacle at `pos`.
    pub fn set_obstacle(&mut self, (x, y): (u32, u32)) {
        let i = self.idx(x, y);
        self.cells[i].obstacle = true;
    }

    /// Clear every cell (both occupancy and obstacles). Obstacles are re-stamped
    /// afterward from the challenge, which may vary the layout by generation.
    pub fn reset(&mut self) {
        for c in &mut self.cells {
            c.obstacle = false;
            c.occupant = None;
        }
    }

    /// Largest valid x / y coordinate (`width - 1` / `height - 1`). Used for
    /// clamping moves to the edge and for the `/(size-1)` sensor normalizations
    /// (127.0 at the default 128×128).
    #[inline]
    pub fn max_x(&self) -> i32 {
        self.width as i32 - 1
    }

    #[inline]
    pub fn max_y(&self) -> i32 {
        self.height as i32 - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexing_round_trips_and_occupancy_reflects_state() {
        // Round-trip coordinates through set/clear and confirm the row-major
        // indexing maps (x, y) to the right cell — a non-square grid so a
        // transposed index would be caught.
        let mut g = Grid::new(10, 6);
        assert!(!g.blocked((3, 4)));
        // Any Entity works; the grid only stores the id and reads is_some().
        let mut w = hecs::World::new();
        let e = w.spawn((1u8,));
        g.set_occupant((3, 4), e);
        assert!(g.blocked((3, 4)), "cell should be occupied after set");
        assert!(!g.blocked((4, 3)), "the transposed cell must be independent");
        g.clear_occupant((3, 4));
        assert!(!g.blocked((3, 4)), "cell should be free after clear");

        g.set_obstacle((0, 0));
        assert!(g.blocked((0, 0)), "obstacle cell is blocked");
        g.set_obstacle((9, 5));
        assert!(g.blocked((9, 5)), "far corner obstacle is blocked");
        g.reset();
        assert!(!g.blocked((0, 0)) && !g.blocked((9, 5)), "reset clears all cells");
    }
}
