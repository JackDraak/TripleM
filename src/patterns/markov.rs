use std::collections::HashMap;
use std::hash::Hash;
use rand::Rng;
// use crate::error::Result;

/// A generic Markov chain implementation for pattern generation
#[derive(Debug, Clone)]
pub struct MarkovChain<T> {
    transitions: HashMap<Vec<T>, HashMap<T, f32>>,
    order: usize,
    current_sequence: Vec<T>,
    total_sequences: usize,
}

impl<T> MarkovChain<T>
where
    T: Clone + Eq + Hash,
{
    /// Create a new Markov chain with the specified order
    pub fn new(order: usize) -> Self {
        if order == 0 {
            panic!("Markov chain order must be at least 1");
        }

        Self {
            transitions: HashMap::new(),
            order,
            current_sequence: Vec::new(),
            total_sequences: 0,
        }
    }

    /// Add a training sequence to the Markov chain
    pub fn add_sequence(&mut self, sequence: &[T]) {
        if sequence.len() < self.order + 1 {
            return; // Sequence too short
        }

        for window in sequence.windows(self.order + 1) {
            let state = window[..self.order].to_vec();
            let next_element = window[self.order].clone();

            let transitions = self.transitions.entry(state).or_insert_with(HashMap::new);
            let count = transitions.entry(next_element).or_insert(0.0);
            *count += 1.0;
        }

        self.total_sequences += 1;
    }

    /// Add multiple training sequences
    pub fn add_sequences(&mut self, sequences: &[&[T]]) {
        for sequence in sequences {
            self.add_sequence(sequence);
        }
    }

    /// Normalize transition probabilities
    pub fn normalize(&mut self) {
        for transitions in self.transitions.values_mut() {
            let total: f32 = transitions.values().sum();
            if total > 0.0 {
                for count in transitions.values_mut() {
                    *count /= total;
                }
            }
        }
    }

    /// Generate the next element based on current state
    pub fn next<R: Rng>(&mut self, rng: &mut R) -> Option<T> {
        if self.current_sequence.len() < self.order {
            return None; // Not enough history
        }

        let state = self.current_sequence[self.current_sequence.len() - self.order..].to_vec();

        if let Some(transitions) = self.transitions.get(&state) {
            let next_element = self.weighted_random_choice(rng, transitions)?;
            self.current_sequence.push(next_element.clone());

            // Keep only the last `order` elements
            if self.current_sequence.len() > self.order {
                self.current_sequence.remove(0);
            }

            Some(next_element)
        } else {
            None
        }
    }

    /// Set the current sequence state
    pub fn set_state(&mut self, state: Vec<T>) {
        if state.len() >= self.order {
            self.current_sequence = state[state.len() - self.order..].to_vec();
        } else {
            self.current_sequence = state;
        }
    }

    /// Get the current state
    pub fn current_state(&self) -> &[T] {
        &self.current_sequence
    }

    /// Reset the chain to empty state
    pub fn reset(&mut self) {
        self.current_sequence.clear();
    }

    /// Get the order of this Markov chain
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the number of states in the transition table
    pub fn state_count(&self) -> usize {
        self.transitions.len()
    }

    /// Get the number of training sequences added
    pub fn sequence_count(&self) -> usize {
        self.total_sequences
    }

    /// Generate a sequence of specified length
    pub fn generate_sequence<R: Rng>(&mut self, rng: &mut R, length: usize) -> Vec<T> {
        let mut sequence = Vec::with_capacity(length);

        for _ in 0..length {
            if let Some(element) = self.next(rng) {
                sequence.push(element);
            } else {
                break;
            }
        }

        sequence
    }

    /// Seed the chain with a random state from the training data
    pub fn seed_random<R: Rng>(&mut self, rng: &mut R) -> bool {
        if self.transitions.is_empty() {
            return false;
        }

        let states: Vec<_> = self.transitions.keys().collect();
        let random_state = states[rng.gen_range(0..states.len())].clone();
        self.current_sequence = random_state;
        true
    }

    /// Get transition probability for a specific state and next element
    pub fn get_probability(&self, state: &[T], next_element: &T) -> f32 {
        if state.len() != self.order {
            return 0.0;
        }

        self.transitions
            .get(state)
            .and_then(|transitions| transitions.get(next_element))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get all possible next elements for a given state
    pub fn get_next_elements(&self, state: &[T]) -> Vec<(T, f32)> {
        if state.len() != self.order {
            return Vec::new();
        }

        self.transitions
            .get(state)
            .map(|transitions| {
                transitions
                    .iter()
                    .map(|(element, &prob)| (element.clone(), prob))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Perform weighted random selection from transition probabilities
    fn weighted_random_choice<R: Rng>(
        &self,
        rng: &mut R,
        transitions: &HashMap<T, f32>,
    ) -> Option<T> {
        let total_weight: f32 = transitions.values().sum();
        if total_weight <= 0.0 {
            return None;
        }

        let mut random_value = rng.gen_range(0.0..total_weight);

        for (element, &weight) in transitions {
            random_value -= weight;
            if random_value <= 0.0 {
                return Some(element.clone());
            }
        }

        // Fallback to first element (shouldn't happen with correct weights)
        transitions.keys().next().cloned()
    }
}

/// Specialized Markov chain for musical notes
pub type NoteMarkovChain = MarkovChain<u8>;

/// Specialized Markov chain for rhythm patterns
pub type RhythmMarkovChain = MarkovChain<bool>;

/// Specialized Markov chain for chord progressions
pub type ChordMarkovChain = MarkovChain<u8>;

/// Builder for creating pre-trained Markov chains
pub struct MarkovBuilder<T> {
    chain: MarkovChain<T>,
}

impl<T> MarkovBuilder<T>
where
    T: Clone + Eq + Hash,
{
    /// Create a new builder with specified order
    pub fn new(order: usize) -> Self {
        Self {
            chain: MarkovChain::new(order),
        }
    }

    /// Add training data
    pub fn add_training_data(mut self, sequences: &[&[T]]) -> Self {
        self.chain.add_sequences(sequences);
        self
    }

    /// Build and normalize the Markov chain
    pub fn build(mut self) -> MarkovChain<T> {
        self.chain.normalize();
        self.chain
    }
}

/// Pre-built Markov chains for common musical patterns
pub mod presets {
    use super::*;

    /// Create a Markov chain trained on common major scale progressions
    pub fn major_scale_melody() -> NoteMarkovChain {
        let sequences: &[&[u8]] = &[
            &[60, 62, 64, 65, 67, 69, 71, 72], // C major scale up
            &[72, 71, 69, 67, 65, 64, 62, 60], // C major scale down
            &[60, 64, 67, 72, 67, 64, 60],     // C major arpeggio
            &[60, 65, 67, 72, 69, 65, 60],     // Common melody pattern
            &[67, 69, 71, 72, 71, 69, 67, 65], // G to E pattern
        ];

        MarkovBuilder::new(2)
            .add_training_data(sequences)
            .build()
    }

    /// Create a Markov chain for simple rhythm patterns
    pub fn simple_rhythm() -> RhythmMarkovChain {
        let sequences: &[&[bool]] = &[
            &[true, false, true, false, true, false, true, false], // Basic beat
            &[true, false, false, true, false, false, true, false], // Syncopated
            &[true, true, false, true, false, true, false, false], // Complex
            &[true, false, true, true, false, true, false, false], // Variation
        ];

        MarkovBuilder::new(2)
            .add_training_data(sequences)
            .build()
    }

    /// Create a Markov chain for chord progressions (Roman numeral analysis)
    pub fn chord_progressions() -> ChordMarkovChain {
        // Using scale degrees: 0=I, 1=ii, 2=iii, 3=IV, 4=V, 5=vi, 6=vii°
        let sequences: &[&[u8]] = &[
            &[0, 3, 4, 0], // I-IV-V-I
            &[0, 5, 3, 4], // I-vi-IV-V
            &[0, 1, 4, 0], // I-ii-V-I
            &[5, 3, 0, 4], // vi-IV-I-V
            &[0, 4, 5, 3], // I-V-vi-IV
        ];

        MarkovBuilder::new(2)
            .add_training_data(sequences)
            .build()
    }

    /// Create a Markov chain for ambient environmental patterns
    pub fn ambient_patterns() -> NoteMarkovChain {
        let sequences: &[&[u8]] = &[
            &[60, 60, 64, 64, 67, 67, 64, 60], // Gentle repetition
            &[55, 60, 64, 67, 64, 60, 55],     // Lower ambient
            &[72, 69, 67, 64, 67, 69, 72],     // Higher ambient
            &[60, 67, 60, 64, 60, 67, 60],     // Simple pattern
        ];

        MarkovBuilder::new(1) // Lower order for more randomness
            .add_training_data(sequences)
            .build()
    }

    /// Create a Markov chain for EDM-style patterns
    pub fn edm_patterns() -> NoteMarkovChain {
        let sequences: &[&[u8]] = &[
            &[36, 36, 43, 36, 48, 43, 36, 48], // Bass-heavy pattern
            &[60, 67, 72, 79, 72, 67, 60, 55], // Lead pattern
            &[48, 48, 55, 55, 60, 60, 67, 67], // Stepped progression
            &[72, 84, 72, 79, 72, 84, 72, 76], // High energy
        ];

        MarkovBuilder::new(3) // Higher order for more structure
            .add_training_data(sequences)
            .build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_markov_chain_creation() {
        let chain: MarkovChain<u8> = MarkovChain::new(2);
        assert_eq!(chain.order(), 2);
        assert_eq!(chain.state_count(), 0);
    }

    #[test]
    fn test_add_sequence() {
        let mut chain = MarkovChain::new(2);
        let sequence = [1, 2, 3, 4, 5];
        chain.add_sequence(&sequence);
        assert_eq!(chain.sequence_count(), 1);
        assert!(chain.state_count() > 0);
    }

    #[test]
    fn test_sequence_generation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut chain = MarkovChain::new(2);

        // Add training data
        let sequences = [
            [1, 2, 3, 4].as_slice(),
            [2, 3, 4, 5].as_slice(),
            [3, 4, 5, 1].as_slice(),
        ];
        chain.add_sequences(&sequences);
        chain.normalize();

        // Set initial state
        chain.set_state(vec![1, 2]);

        // Generate next elements
        let next1 = chain.next(&mut rng);
        assert!(next1.is_some());

        let next2 = chain.next(&mut rng);
        assert!(next2.is_some());
    }

    #[test]
    fn test_probability_calculation() {
        let mut chain = MarkovChain::new(1);
        let sequence = [1, 2, 1, 2, 1, 3];
        chain.add_sequence(&sequence);
        chain.normalize();

        // After state [1], we see [2] twice and [3] once
        // So probability of 2 after 1 should be 2/3 ≈ 0.67
        let prob_2_after_1 = chain.get_probability(&[1], &2);
        assert!((prob_2_after_1 - 0.6667).abs() < 0.01);

        let prob_3_after_1 = chain.get_probability(&[1], &3);
        assert!((prob_3_after_1 - 0.3333).abs() < 0.01);
    }

    #[test]
    fn test_next_elements() {
        let mut chain = MarkovChain::new(1);
        let sequence = [1, 2, 1, 3];
        chain.add_sequence(&sequence);
        chain.normalize();

        let next_elements = chain.get_next_elements(&[1]);
        assert_eq!(next_elements.len(), 2); // Should have transitions to 2 and 3

        let total_prob: f32 = next_elements.iter().map(|(_, prob)| prob).sum();
        assert!((total_prob - 1.0).abs() < 0.01); // Should sum to 1.0
    }

    #[test]
    fn test_sequence_generation_with_length() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut chain = MarkovChain::new(1);

        // Simple training: 1->2->3->1 cycle
        let sequence = [1, 2, 3, 1, 2, 3, 1, 2, 3];
        chain.add_sequence(&sequence);
        chain.normalize();

        chain.set_state(vec![1]);
        let generated = chain.generate_sequence(&mut rng, 10);

        assert_eq!(generated.len(), 10);
        // Should contain valid values from our training set
        for &val in &generated {
            assert!([1, 2, 3].contains(&val));
        }
    }

    #[test]
    fn test_seed_random() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut chain = MarkovChain::new(2);

        // Empty chain should fail to seed
        assert!(!chain.seed_random(&mut rng));

        // Add some data
        chain.add_sequence(&[1, 2, 3, 4, 5]);
        chain.normalize();

        // Now should be able to seed
        assert!(chain.seed_random(&mut rng));
        assert!(!chain.current_state().is_empty());
    }

    #[test]
    fn test_preset_chains() {
        use presets::*;

        let mut rng = StdRng::seed_from_u64(42);

        // Test major scale melody chain
        let mut melody_chain = major_scale_melody();
        assert!(melody_chain.seed_random(&mut rng));
        let melody = melody_chain.generate_sequence(&mut rng, 8);
        assert_eq!(melody.len(), 8);

        // Test rhythm chain
        let mut rhythm_chain = simple_rhythm();
        assert!(rhythm_chain.seed_random(&mut rng));
        let rhythm = rhythm_chain.generate_sequence(&mut rng, 16);
        assert_eq!(rhythm.len(), 16);

        // Test chord progression chain
        let mut chord_chain = chord_progressions();
        assert!(chord_chain.seed_random(&mut rng));
        let chords = chord_chain.generate_sequence(&mut rng, 4);
        assert_eq!(chords.len(), 4);
        // All chord values should be valid scale degrees (0-6)
        for &chord in &chords {
            assert!(chord <= 6);
        }
    }

    #[test]
    #[should_panic]
    fn test_zero_order_panic() {
        MarkovChain::<u8>::new(0);
    }

    #[test]
    fn test_builder_pattern() {
        let sequences = [
            [1, 2, 3].as_slice(),
            [2, 3, 4].as_slice(),
            [3, 4, 1].as_slice(),
        ];

        let chain = MarkovBuilder::new(2)
            .add_training_data(&sequences)
            .build();

        assert_eq!(chain.order(), 2);
        assert!(chain.state_count() > 0);
    }
}