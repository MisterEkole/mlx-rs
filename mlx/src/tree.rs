use crate::Array;

pub trait TreeFlatten {
    /// Extracts all arrays from this state into a flat vector.
    fn flatten_state(&self) -> Vec<Array>;

    /// Updates the internal arrays by consuming from an iterator of flat arrays.
    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>);
}