//! Genetic algorithm for evolving rhythm patterns
//!
//! This module implements genetic algorithms for procedural music generation,
//! allowing rhythm patterns to evolve and adapt over time to create more
//! engaging and dynamic musical experiences.

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Genetic algorithm for evolving rhythm patterns
#[derive(Debug, Clone)]
pub struct GeneticRhythm {
    population: Vec<RhythmChromosome>,
    population_size: usize,
    generation: usize,

    // Genetic algorithm parameters
    mutation_rate: f32,
    crossover_rate: f32,
    elite_percentage: f32,

    // Evolution targets
    target_complexity: f32,
    target_density: f32,
    target_syncopation: f32,

    // Random number generator
    rng: StdRng,

    // Current best pattern
    current_best: RhythmChromosome,

    // Evolution tracking
    evolution_timer: f32,
    evolution_interval: f32, // Time between generations
}

/// Rhythm chromosome representing a pattern in genetic form
#[derive(Debug, Clone)]
pub struct RhythmChromosome {
    genes: Vec<bool>,           // Rhythm pattern (true = hit, false = rest)
    fitness: f32,               // Fitness score for selection
    age: usize,                 // Generation when created
    complexity_score: f32,      // Measure of pattern complexity
    density_score: f32,         // Measure of hit density
    syncopation_score: f32,     // Measure of syncopation
}

/// Fitness criteria for evaluating rhythm patterns
#[derive(Debug, Clone)]
pub struct FitnessCriteria {
    pub target_complexity: f32,      // 0.0 = simple, 1.0 = complex
    pub target_density: f32,         // 0.0 = sparse, 1.0 = dense
    pub target_syncopation: f32,     // 0.0 = on-beat, 1.0 = syncopated
    pub target_groove: f32,          // 0.0 = mechanical, 1.0 = groovy
    pub pattern_length: usize,       // Length of rhythm pattern
}

impl GeneticRhythm {
    /// Create a new genetic rhythm evolution system
    pub fn new(pattern_length: usize, fitness_criteria: FitnessCriteria) -> Self {
        let population_size = 32;
        let mut rng = StdRng::from_entropy();

        // Initialize population with random patterns
        let mut population = Vec::with_capacity(population_size);
        for _ in 0..population_size {
            let chromosome = RhythmChromosome::random(pattern_length, &mut rng);
            population.push(chromosome);
        }

        // Start with the first chromosome as current best
        let current_best = population[0].clone();

        Self {
            population,
            population_size,
            generation: 0,

            mutation_rate: 0.15,        // 15% mutation rate
            crossover_rate: 0.7,        // 70% crossover rate
            elite_percentage: 0.2,      // Keep top 20%

            target_complexity: fitness_criteria.target_complexity,
            target_density: fitness_criteria.target_density,
            target_syncopation: fitness_criteria.target_syncopation,

            rng,
            current_best,

            evolution_timer: 0.0,
            evolution_interval: 8.0,    // Evolve every 8 seconds
        }
    }

    /// Update the genetic algorithm (call once per audio sample or frame)
    pub fn update(&mut self, dt: f32) {
        self.evolution_timer += dt;

        if self.evolution_timer >= self.evolution_interval {
            self.evolution_timer = 0.0;
            self.evolve_generation();
        }
    }

    /// Get the current best rhythm pattern
    pub fn get_current_pattern(&self) -> &[bool] {
        &self.current_best.genes
    }

    /// Get the current generation number
    pub fn get_generation(&self) -> usize {
        self.generation
    }

    /// Get the fitness of the current best pattern
    pub fn get_best_fitness(&self) -> f32 {
        self.current_best.fitness
    }

    /// Set evolution targets (allows dynamic adaptation)
    pub fn set_targets(&mut self, complexity: f32, density: f32, syncopation: f32) {
        self.target_complexity = complexity.clamp(0.0, 1.0);
        self.target_density = density.clamp(0.0, 1.0);
        self.target_syncopation = syncopation.clamp(0.0, 1.0);
    }

    /// Force evolution of a new generation
    pub fn evolve_generation(&mut self) {
        // Calculate fitness for all chromosomes
        for chromosome in &mut self.population {
            chromosome.calculate_fitness(
                self.target_complexity,
                self.target_density,
                self.target_syncopation
            );
        }

        // Sort by fitness (descending)
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Update current best
        self.current_best = self.population[0].clone();

        // Create new generation
        let mut new_population = Vec::with_capacity(self.population_size);

        // Keep elite chromosomes
        let elite_count = (self.population_size as f32 * self.elite_percentage) as usize;
        for i in 0..elite_count {
            new_population.push(self.population[i].clone());
        }

        // Generate offspring through crossover and mutation
        while new_population.len() < self.population_size {
            let parent1 = self.tournament_selection();
            let parent2 = self.tournament_selection();

            let (mut child1, mut child2) = if self.rng.gen::<f32>() < self.crossover_rate {
                RhythmChromosome::crossover(&parent1, &parent2, &mut self.rng)
            } else {
                (parent1.clone(), parent2.clone())
            };

            // Apply mutation
            if self.rng.gen::<f32>() < self.mutation_rate {
                child1.mutate(&mut self.rng);
            }
            if self.rng.gen::<f32>() < self.mutation_rate {
                child2.mutate(&mut self.rng);
            }

            child1.age = self.generation + 1;
            child2.age = self.generation + 1;

            new_population.push(child1);
            if new_population.len() < self.population_size {
                new_population.push(child2);
            }
        }

        self.population = new_population;
        self.generation += 1;
    }

    /// Tournament selection for choosing parents
    fn tournament_selection(&mut self) -> RhythmChromosome {
        let tournament_size = 3;
        let mut best_fitness = -1.0;
        let mut best_index = 0;

        for _ in 0..tournament_size {
            let index = self.rng.gen_range(0..self.population.len());
            if self.population[index].fitness > best_fitness {
                best_fitness = self.population[index].fitness;
                best_index = index;
            }
        }

        self.population[best_index].clone()
    }

    /// Create a genetic rhythm system optimized for productivity music
    pub fn for_productivity(pattern_length: usize) -> Self {
        let criteria = FitnessCriteria {
            target_complexity: 0.4,     // Moderate complexity
            target_density: 0.6,        // Moderate density
            target_syncopation: 0.3,    // Light syncopation
            target_groove: 0.7,         // Good groove
            pattern_length,
        };

        Self::new(pattern_length, criteria)
    }

    /// Create a genetic rhythm system optimized for ambient music
    pub fn for_ambient(pattern_length: usize) -> Self {
        let criteria = FitnessCriteria {
            target_complexity: 0.2,     // Low complexity
            target_density: 0.3,        // Sparse
            target_syncopation: 0.1,    // Minimal syncopation
            target_groove: 0.4,         // Gentle groove
            pattern_length,
        };

        Self::new(pattern_length, criteria)
    }

    /// Create a genetic rhythm system optimized for electronic music
    pub fn for_electronic(pattern_length: usize) -> Self {
        let criteria = FitnessCriteria {
            target_complexity: 0.8,     // High complexity
            target_density: 0.8,        // Dense
            target_syncopation: 0.6,    // Heavy syncopation
            target_groove: 0.9,         // Strong groove
            pattern_length,
        };

        Self::new(pattern_length, criteria)
    }
}

impl RhythmChromosome {
    /// Create a random rhythm chromosome
    pub fn random(length: usize, rng: &mut StdRng) -> Self {
        let mut genes = Vec::with_capacity(length);

        // Generate pattern with some musical logic
        for i in 0..length {
            let probability = if i % 4 == 0 {
                0.8 // Strong beats more likely
            } else if i % 2 == 0 {
                0.4 // Weak beats moderate probability
            } else {
                0.2 // Off-beats low probability
            };

            genes.push(rng.gen::<f32>() < probability);
        }

        let mut chromosome = Self {
            genes,
            fitness: 0.0,
            age: 0,
            complexity_score: 0.0,
            density_score: 0.0,
            syncopation_score: 0.0,
        };

        chromosome.update_scores();
        chromosome
    }

    /// Perform crossover between two chromosomes
    pub fn crossover(parent1: &RhythmChromosome, parent2: &RhythmChromosome, rng: &mut StdRng) -> (RhythmChromosome, RhythmChromosome) {
        let length = parent1.genes.len();
        let crossover_point = rng.gen_range(1..length);

        let mut child1_genes = Vec::with_capacity(length);
        let mut child2_genes = Vec::with_capacity(length);

        // Single-point crossover
        for i in 0..length {
            if i < crossover_point {
                child1_genes.push(parent1.genes[i]);
                child2_genes.push(parent2.genes[i]);
            } else {
                child1_genes.push(parent2.genes[i]);
                child2_genes.push(parent1.genes[i]);
            }
        }

        let mut child1 = RhythmChromosome {
            genes: child1_genes,
            fitness: 0.0,
            age: 0,
            complexity_score: 0.0,
            density_score: 0.0,
            syncopation_score: 0.0,
        };

        let mut child2 = RhythmChromosome {
            genes: child2_genes,
            fitness: 0.0,
            age: 0,
            complexity_score: 0.0,
            density_score: 0.0,
            syncopation_score: 0.0,
        };

        child1.update_scores();
        child2.update_scores();

        (child1, child2)
    }

    /// Apply mutation to the chromosome
    pub fn mutate(&mut self, rng: &mut StdRng) {
        let mutation_strength = 0.2; // 20% of bits can flip
        let num_mutations = (self.genes.len() as f32 * mutation_strength) as usize;

        for _ in 0..num_mutations.max(1) {
            let index = rng.gen_range(0..self.genes.len());
            self.genes[index] = !self.genes[index];
        }

        self.update_scores();
    }

    /// Calculate fitness based on target criteria
    pub fn calculate_fitness(&mut self, target_complexity: f32, target_density: f32, target_syncopation: f32) {
        self.update_scores();

        // Calculate fitness as inverse of distance from targets
        let complexity_diff = (self.complexity_score - target_complexity).abs();
        let density_diff = (self.density_score - target_density).abs();
        let syncopation_diff = (self.syncopation_score - target_syncopation).abs();

        // Weighted fitness calculation
        let complexity_weight = 0.3;
        let density_weight = 0.4;
        let syncopation_weight = 0.3;

        let total_diff = complexity_diff * complexity_weight +
                        density_diff * density_weight +
                        syncopation_diff * syncopation_weight;

        // Convert to fitness (0.0 = worst, 1.0 = best)
        self.fitness = 1.0 - total_diff;

        // Bonus for musical patterns
        if self.has_good_downbeats() {
            self.fitness += 0.1;
        }

        if self.has_musical_phrase_structure() {
            self.fitness += 0.05;
        }

        self.fitness = self.fitness.clamp(0.0, 1.0);
    }

    /// Update internal scores
    fn update_scores(&mut self) {
        self.complexity_score = self.calculate_complexity();
        self.density_score = self.calculate_density();
        self.syncopation_score = self.calculate_syncopation();
    }

    /// Calculate pattern complexity
    fn calculate_complexity(&self) -> f32 {
        let mut transitions = 0;
        for i in 1..self.genes.len() {
            if self.genes[i] != self.genes[i - 1] {
                transitions += 1;
            }
        }

        // Normalize by maximum possible transitions
        transitions as f32 / (self.genes.len() - 1) as f32
    }

    /// Calculate hit density
    fn calculate_density(&self) -> f32 {
        let hits = self.genes.iter().filter(|&&x| x).count();
        hits as f32 / self.genes.len() as f32
    }

    /// Calculate syncopation level
    fn calculate_syncopation(&self) -> f32 {
        let mut syncopation = 0.0;
        let pattern_len = self.genes.len();

        for (i, &hit) in self.genes.iter().enumerate() {
            if hit {
                // Off-beat hits contribute to syncopation
                let beat_strength = if i % 4 == 0 {
                    0.0 // Strong beat - no syncopation
                } else if i % 2 == 0 {
                    0.3 // Weak beat - some syncopation
                } else {
                    1.0 // Off-beat - full syncopation
                };

                syncopation += beat_strength;
            }
        }

        // Normalize by pattern length
        syncopation / pattern_len as f32
    }

    /// Check if pattern has good downbeats
    fn has_good_downbeats(&self) -> bool {
        // Check if beat 1 of each measure has a hit
        let mut downbeat_hits = 0;
        let mut total_downbeats = 0;

        for i in (0..self.genes.len()).step_by(4) {
            total_downbeats += 1;
            if self.genes[i] {
                downbeat_hits += 1;
            }
        }

        // At least 75% of downbeats should have hits
        downbeat_hits as f32 / total_downbeats as f32 >= 0.75
    }

    /// Check if pattern has musical phrase structure
    fn has_musical_phrase_structure(&self) -> bool {
        // Simple check: pattern should have some repetition or variation
        let first_half = &self.genes[0..self.genes.len() / 2];
        let second_half = &self.genes[self.genes.len() / 2..];

        // Calculate similarity between halves
        let matches = first_half.iter()
            .zip(second_half.iter())
            .filter(|(a, b)| a == b)
            .count();

        let similarity = matches as f32 / first_half.len() as f32;

        // Good phrase structure: 30-70% similarity (some repetition, some variation)
        similarity >= 0.3 && similarity <= 0.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genetic_rhythm_creation() {
        let criteria = FitnessCriteria {
            target_complexity: 0.5,
            target_density: 0.5,
            target_syncopation: 0.3,
            target_groove: 0.7,
            pattern_length: 16,
        };

        let genetic_rhythm = GeneticRhythm::new(16, criteria);
        assert_eq!(genetic_rhythm.population.len(), 32);
        assert_eq!(genetic_rhythm.get_current_pattern().len(), 16);
    }

    #[test]
    fn test_chromosome_creation() {
        let mut rng = StdRng::seed_from_u64(42);
        let chromosome = RhythmChromosome::random(16, &mut rng);

        assert_eq!(chromosome.genes.len(), 16);
        assert!(chromosome.complexity_score >= 0.0 && chromosome.complexity_score <= 1.0);
        assert!(chromosome.density_score >= 0.0 && chromosome.density_score <= 1.0);
        assert!(chromosome.syncopation_score >= 0.0 && chromosome.syncopation_score <= 1.0);
    }

    #[test]
    fn test_crossover() {
        let mut rng = StdRng::seed_from_u64(42);
        let parent1 = RhythmChromosome::random(8, &mut rng);
        let parent2 = RhythmChromosome::random(8, &mut rng);

        let (child1, child2) = RhythmChromosome::crossover(&parent1, &parent2, &mut rng);

        assert_eq!(child1.genes.len(), 8);
        assert_eq!(child2.genes.len(), 8);
    }

    #[test]
    fn test_mutation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut chromosome = RhythmChromosome::random(16, &mut rng);
        let original_genes = chromosome.genes.clone();

        chromosome.mutate(&mut rng);

        // Should have some differences after mutation
        let differences = original_genes.iter()
            .zip(chromosome.genes.iter())
            .filter(|(a, b)| a != b)
            .count();

        assert!(differences > 0);
    }

    #[test]
    fn test_fitness_calculation() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut chromosome = RhythmChromosome::random(16, &mut rng);

        chromosome.calculate_fitness(0.5, 0.5, 0.3);
        assert!(chromosome.fitness >= 0.0 && chromosome.fitness <= 1.0);
    }

    #[test]
    fn test_preset_creation() {
        let productivity = GeneticRhythm::for_productivity(16);
        let ambient = GeneticRhythm::for_ambient(16);
        let electronic = GeneticRhythm::for_electronic(16);

        assert_eq!(productivity.target_complexity, 0.4);
        assert_eq!(ambient.target_density, 0.3);
        assert_eq!(electronic.target_syncopation, 0.6);
    }

    #[test]
    fn test_evolution() {
        let mut genetic_rhythm = GeneticRhythm::for_productivity(8);
        let initial_generation = genetic_rhythm.get_generation();

        genetic_rhythm.evolve_generation();

        assert_eq!(genetic_rhythm.get_generation(), initial_generation + 1);
        assert!(genetic_rhythm.get_best_fitness() >= 0.0);
    }
}