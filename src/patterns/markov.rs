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

    /// Create a Markov chain trained on sophisticated melodic patterns for gentle music
    pub fn gentle_melodies() -> NoteMarkovChain {
        let sequences: &[&[u8]] = &[
            // Pentatonic-based gentle melodies (C pentatonic: C D E G A)
            &[60, 62, 64, 67, 69, 67, 64, 62, 60], // Pentatonic scale flow
            &[69, 67, 64, 62, 60, 62, 64, 67, 69], // Mirror pattern
            &[60, 64, 67, 69, 72, 69, 67, 64, 60], // Octave arpeggios

            // Gentle interval patterns
            &[60, 65, 64, 69, 67, 72, 67, 64, 60], // Perfect fourths and fifths
            &[72, 67, 69, 64, 65, 60, 64, 67, 72], // Descending then ascending

            // Spa-like flowing patterns
            &[60, 62, 67, 65, 69, 67, 64, 62, 60], // Gentle waves
            &[67, 64, 69, 65, 72, 67, 69, 64, 67], // Floating melodies
            &[60, 64, 60, 67, 64, 69, 67, 64, 60], // Repetitive meditation

            // Minor pentatonic for warmth (A minor pentatonic: A C D E G)
            &[57, 60, 62, 64, 67, 64, 62, 60, 57], // Am pentatonic
            &[67, 64, 62, 60, 57, 60, 62, 64, 67], // Reverse flow
        ];

        MarkovBuilder::new(2)
            .add_training_data(sequences)
            .build()
    }

    /// Create sophisticated chord progression patterns for ambient music
    pub fn sophisticated_chord_progressions() -> ChordMarkovChain {
        // Using scale degrees: 0=I, 1=ii, 2=iii, 3=IV, 4=V, 5=vi, 6=vii°
        let sequences: &[&[u8]] = &[
            // Neo-soul and jazz-influenced progressions
            &[0, 5, 3, 4, 0], // I-vi-IV-V-I (classic)
            &[0, 2, 5, 3, 4, 0], // I-iii-vi-IV-V-I (extended)
            &[5, 3, 0, 4, 5], // vi-IV-I-V-vi (pop progression)

            // Modal progressions for ambient feel
            &[0, 6, 3, 0], // I-vii°-IV-I (modal flavor)
            &[3, 0, 4, 5], // IV-I-V-vi (Plagal motion)
            &[0, 1, 3, 1, 0], // I-ii-IV-ii-I (sus feel)

            // Complex jazz-inspired movements
            &[0, 2, 1, 4, 0], // I-iii-ii-V-I (jazz turnaround)
            &[5, 1, 4, 0], // vi-ii-V-I (jazz standard)
            &[0, 6, 3, 6, 0], // I-vii°-IV-vii°-I (chromatic)

            // Ambient/ethereal progressions
            &[0, 4, 3, 5, 0], // I-V-IV-vi-I (open voicing feel)
            &[3, 5, 0, 1, 3], // IV-vi-I-ii-IV (circular)
        ];

        MarkovBuilder::new(2)
            .add_training_data(sequences)
            .build()
    }

    /// Create sophisticated rhythm patterns for productivity music
    pub fn productivity_rhythms() -> RhythmMarkovChain {
        let sequences: &[&[bool]] = &[
            // Steady but engaging 4/4 patterns
            &[true, false, true, false, true, false, true, false], // Basic steady
            &[true, false, false, true, false, true, false, false], // Slight syncopation
            &[true, true, false, true, false, false, true, false], // Double hit start

            // Polyrhythmic-inspired patterns (3 against 4 feel)
            &[true, false, false, true, false, true, false, true], // 3+3+2 feeling
            &[true, false, true, false, false, true, false, true], // Offset pattern

            // Focus-enhancing patterns (research shows certain rhythms aid concentration)
            &[true, false, true, true, false, false, true, false], // Strong-weak-strong-strong pattern
            &[true, false, false, false, true, false, true, false], // Spaced for thinking

            // Subtle complexity for long listening
            &[true, false, true, false, true, true, false, false], // Varied ending
            &[true, true, false, false, true, false, true, true], // Clustered beats
        ];

        MarkovBuilder::new(3) // Higher order for more structured patterns
            .add_training_data(sequences)
            .build()
    }

    /// Create complex EDM rhythm patterns with evolution potential
    pub fn edm_complex_rhythms() -> RhythmMarkovChain {
        let sequences: &[&[bool]] = &[
            // Standard EDM kick patterns
            &[true, false, false, false, true, false, false, false], // Four-on-floor
            &[true, false, false, true, false, false, true, false], // Broken kick

            // Hi-hat patterns
            &[false, false, true, false, false, false, true, false], // Upbeat hats
            &[false, true, false, true, false, true, false, true], // Rapid hats
            &[true, true, false, true, true, false, true, false], // Complex hats

            // Snare patterns
            &[false, false, false, false, true, false, false, false], // Standard snare
            &[false, false, true, false, false, false, true, false], // Double snare

            // Complex polyrhythmic patterns for builds
            &[true, false, true, true, false, true, false, true], // Dense pattern
            &[true, true, false, false, true, false, true, true], // Clustered
            &[true, false, false, true, true, false, false, true], // Syncopated

            // Breakdown patterns
            &[true, false, false, false, false, false, false, false], // Minimal
            &[true, false, true, false, false, false, false, false], // Sparse
        ];

        MarkovBuilder::new(4) // Very high order for structured EDM feel
            .add_training_data(sequences)
            .build()
    }

    /// Create bass line patterns specifically for EDM
    pub fn edm_bass_patterns() -> NoteMarkovChain {
        let sequences: &[&[u8]] = &[
            // Root-fifth patterns in different keys
            &[36, 36, 43, 36, 36, 43, 36, 43], // C1-G1 pattern
            &[38, 38, 45, 38, 38, 45, 38, 45], // D1-A1 pattern
            &[41, 41, 48, 41, 41, 48, 41, 48], // F1-C2 pattern
            &[43, 43, 50, 43, 43, 50, 43, 50], // G1-D2 pattern

            // More complex bass patterns with passing tones
            &[36, 41, 43, 48, 43, 41, 36, 31], // Chromatic movement
            &[36, 38, 36, 43, 41, 38, 36, 34], // Scalar movement
            &[36, 48, 43, 55, 48, 43, 36, 24], // Octave jumps

            // Progressive house style patterns
            &[36, 43, 48, 43, 41, 48, 43, 36], // Complex progression
            &[36, 36, 41, 43, 43, 48, 43, 41], // Building pattern
            &[24, 36, 43, 48, 55, 48, 43, 36], // Wide range pattern

            // Wobble bass note patterns
            &[36, 36, 36, 43, 36, 36, 36, 43], // Minimal wobble
            &[36, 38, 36, 41, 36, 43, 36, 41], // Chromatic wobble
        ];

        MarkovBuilder::new(3)
            .add_training_data(sequences)
            .build()
    }

    /// Create arpeggio patterns for different moods
    pub fn arpeggio_patterns(mood_type: ArpeggioMood) -> NoteMarkovChain {
        let sequences: &[&[u8]] = match mood_type {
            ArpeggioMood::Gentle => &[
                // Gentle, flowing arpeggios
                &[60, 64, 67, 72, 67, 64, 60, 55], // C major triad flow
                &[57, 60, 64, 69, 64, 60, 57, 52], // A minor triad flow
                &[65, 69, 72, 77, 72, 69, 65, 60], // F major triad flow
                &[60, 65, 69, 72, 69, 65, 60, 57], // Mixed intervals
                &[67, 71, 74, 79, 74, 71, 67, 64], // G major extended
            ],
            ArpeggioMood::Active => &[
                // More energetic, productivity-focused patterns
                &[72, 76, 79, 84, 79, 76, 72, 67], // Higher register active
                &[60, 64, 67, 72, 76, 72, 67, 64], // Ascending energy
                &[84, 79, 76, 72, 76, 79, 84, 88], // High energy pattern
                &[48, 60, 64, 67, 72, 67, 64, 60], // Wide range flow
                &[72, 67, 76, 72, 79, 76, 72, 67], // Interlocking pattern
            ],
            ArpeggioMood::Electronic => &[
                // Sharp, digital-feeling patterns
                &[60, 72, 84, 96, 84, 72, 60, 48], // Octave jumps
                &[36, 48, 60, 72, 60, 48, 36, 24], // Bass to treble
                &[72, 84, 72, 96, 84, 72, 96, 84], // Electronic bounce
                &[48, 60, 72, 84, 96, 84, 72, 60], // Rising energy
                &[60, 67, 72, 79, 84, 79, 72, 67], // Fifths-based pattern
            ],
        };

        let order = match mood_type {
            ArpeggioMood::Gentle => 2,      // More predictable
            ArpeggioMood::Active => 2,      // Balanced
            ArpeggioMood::Electronic => 3,  // More structured
        };

        MarkovBuilder::new(order)
            .add_training_data(sequences)
            .build()
    }

    /// Enhanced melody patterns with more musical sophistication
    pub fn enhanced_melodies(style: MelodyStyle) -> NoteMarkovChain {
        let sequences: &[&[u8]] = match style {
            MelodyStyle::Modal => &[
                // Dorian mode patterns (D dorian: D E F G A B C D)
                &[62, 64, 65, 67, 69, 71, 72, 74], // D dorian ascending
                &[74, 72, 71, 69, 67, 65, 64, 62], // D dorian descending
                &[62, 67, 69, 74, 72, 67, 65, 62], // Dorian leap patterns

                // Mixolydian patterns (G mixolydian: G A B C D E F G)
                &[67, 69, 71, 72, 74, 76, 77, 79], // G mixolydian
                &[67, 72, 74, 77, 76, 72, 69, 67], // Mixolydian arpeggios
            ],
            MelodyStyle::Pentatonic => &[
                // Major pentatonic variations
                &[60, 62, 64, 67, 69, 72, 69, 67, 64, 62, 60], // Extended pentatonic
                &[69, 72, 74, 77, 81, 77, 74, 72, 69], // High register pentatonic
                &[48, 50, 52, 55, 57, 60, 57, 55, 52, 50, 48], // Low register

                // Minor pentatonic patterns
                &[57, 60, 62, 64, 67, 69, 67, 64, 62, 60, 57], // A minor pentatonic
                &[69, 72, 74, 76, 79, 81, 79, 76, 74, 72, 69], // High minor pentatonic
            ],
            MelodyStyle::Chromatic => &[
                // Sophisticated chromatic passages
                &[60, 61, 62, 63, 64, 65, 66, 67], // Chromatic ascent
                &[72, 71, 70, 69, 68, 67, 66, 65], // Chromatic descent
                &[60, 63, 61, 64, 62, 65, 63, 66], // Chromatic weaving
                &[67, 66, 68, 67, 69, 68, 70, 69], // Chromatic oscillation
            ],
        };

        MarkovBuilder::new(2)
            .add_training_data(sequences)
            .build()
    }
}

/// Mood types for arpeggio patterns
#[derive(Debug, Clone, Copy)]
pub enum ArpeggioMood {
    Gentle,
    Active,
    Electronic,
}

/// Melody style categories
#[derive(Debug, Clone, Copy)]
pub enum MelodyStyle {
    Modal,
    Pentatonic,
    Chromatic,
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