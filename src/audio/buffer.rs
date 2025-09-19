use std::sync::atomic::{AtomicUsize, Ordering};
use crate::error::{MoodMusicError, Result};

/// Lock-free circular buffer for audio data
#[derive(Debug)]
pub struct AudioBuffer {
    data: Vec<f32>,
    write_pos: AtomicUsize,
    read_pos: AtomicUsize,
    capacity: usize,
}

impl AudioBuffer {
    /// Create a new audio buffer with the specified capacity
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err(MoodMusicError::InvalidConfiguration(
                format!("Buffer capacity must be a power of two and greater than 0, got {}", capacity)
            ));
        }

        Ok(Self {
            data: vec![0.0; capacity],
            write_pos: AtomicUsize::new(0),
            read_pos: AtomicUsize::new(0),
            capacity,
        })
    }

    /// Get the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the current number of available samples for reading
    pub fn available_read(&self) -> usize {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let read_pos = self.read_pos.load(Ordering::Acquire);
        (write_pos.wrapping_sub(read_pos)) & (self.capacity - 1)
    }

    /// Get the current number of available spaces for writing
    pub fn available_write(&self) -> usize {
        self.capacity - self.available_read() - 1
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.available_read() == 0
    }

    /// Check if the buffer is full
    pub fn is_full(&self) -> bool {
        self.available_write() == 0
    }

    /// Write a single sample to the buffer
    /// Returns true if successful, false if buffer is full
    pub fn write_sample(&mut self, sample: f32) -> bool {
        let write_pos = self.write_pos.load(Ordering::Acquire);
        let next_write = (write_pos + 1) & (self.capacity - 1);
        let read_pos = self.read_pos.load(Ordering::Acquire);

        if next_write == read_pos {
            return false; // Buffer full
        }

        // Safety: We've checked that the write position is valid
        unsafe {
            *self.data.get_unchecked_mut(write_pos) = sample;
        }

        self.write_pos.store(next_write, Ordering::Release);
        true
    }

    /// Read a single sample from the buffer
    /// Returns Some(sample) if successful, None if buffer is empty
    pub fn read_sample(&self) -> Option<f32> {
        let read_pos = self.read_pos.load(Ordering::Acquire);
        let write_pos = self.write_pos.load(Ordering::Acquire);

        if read_pos == write_pos {
            return None; // Buffer empty
        }

        // Safety: We've checked that the read position is valid
        let sample = unsafe { *self.data.get_unchecked(read_pos) };

        let next_read = (read_pos + 1) & (self.capacity - 1);
        self.read_pos.store(next_read, Ordering::Release);

        Some(sample)
    }

    /// Write multiple samples to the buffer
    /// Returns the number of samples actually written
    pub fn write_samples(&mut self, samples: &[f32]) -> usize {
        let mut written = 0;
        for &sample in samples {
            if !self.write_sample(sample) {
                break;
            }
            written += 1;
        }
        written
    }

    /// Read multiple samples from the buffer
    /// Returns the number of samples actually read
    pub fn read_samples(&self, output: &mut [f32]) -> usize {
        let mut read = 0;
        for slot in output.iter_mut() {
            if let Some(sample) = self.read_sample() {
                *slot = sample;
                read += 1;
            } else {
                break;
            }
        }
        read
    }

    /// Clear the buffer
    pub fn clear(&self) {
        self.read_pos.store(0, Ordering::Release);
        self.write_pos.store(0, Ordering::Release);
    }

    /// Get buffer utilization as a percentage (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        self.available_read() as f32 / self.capacity as f32
    }
}

unsafe impl Send for AudioBuffer {}
unsafe impl Sync for AudioBuffer {}

/// Multi-channel audio buffer for stereo or surround sound
#[derive(Debug)]
pub struct MultiChannelBuffer {
    channels: Vec<AudioBuffer>,
    channel_count: usize,
}

impl MultiChannelBuffer {
    /// Create a new multi-channel buffer
    pub fn new(channel_count: usize, capacity_per_channel: usize) -> Result<Self> {
        if channel_count == 0 {
            return Err(MoodMusicError::InvalidConfiguration(
                "Channel count must be greater than 0".to_string()
            ));
        }

        let mut channels = Vec::with_capacity(channel_count);
        for _ in 0..channel_count {
            channels.push(AudioBuffer::new(capacity_per_channel)?);
        }

        Ok(Self {
            channels,
            channel_count,
        })
    }

    /// Get the number of channels
    pub fn channel_count(&self) -> usize {
        self.channel_count
    }

    /// Get a reference to a specific channel buffer
    pub fn channel(&self, index: usize) -> Option<&AudioBuffer> {
        self.channels.get(index)
    }

    /// Write interleaved samples (e.g., LRLRLR for stereo)
    pub fn write_interleaved(&mut self, samples: &[f32]) -> usize {
        let mut written = 0;
        let chunks = samples.chunks_exact(self.channel_count);

        for chunk in chunks {
            let mut all_written = true;
            for (channel_idx, &sample) in chunk.iter().enumerate() {
                if !self.channels[channel_idx].write_sample(sample) {
                    all_written = false;
                    break;
                }
            }
            if all_written {
                written += self.channel_count;
            } else {
                break;
            }
        }

        written
    }

    /// Read interleaved samples
    pub fn read_interleaved(&self, output: &mut [f32]) -> usize {
        let mut read = 0;
        let chunks = output.chunks_exact_mut(self.channel_count);

        for chunk in chunks {
            let mut all_read = true;
            for (channel_idx, slot) in chunk.iter_mut().enumerate() {
                if let Some(sample) = self.channels[channel_idx].read_sample() {
                    *slot = sample;
                } else {
                    all_read = false;
                    break;
                }
            }
            if all_read {
                read += self.channel_count;
            } else {
                break;
            }
        }

        read
    }

    /// Clear all channel buffers
    pub fn clear(&self) {
        for channel in &self.channels {
            channel.clear();
        }
    }

    /// Get the minimum available read samples across all channels
    pub fn min_available_read(&self) -> usize {
        self.channels
            .iter()
            .map(|ch| ch.available_read())
            .min()
            .unwrap_or(0)
    }

    /// Get the minimum available write space across all channels
    pub fn min_available_write(&self) -> usize {
        self.channels
            .iter()
            .map(|ch| ch.available_write())
            .min()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buffer = AudioBuffer::new(1024).unwrap();
        assert_eq!(buffer.capacity(), 1024);
        assert!(buffer.is_empty());
        assert!(!buffer.is_full());
    }

    #[test]
    fn test_invalid_capacity() {
        assert!(AudioBuffer::new(0).is_err());
        assert!(AudioBuffer::new(1023).is_err()); // Not power of two
    }

    #[test]
    fn test_single_sample_operations() {
        let buffer = AudioBuffer::new(4).unwrap();

        assert!(buffer.write_sample(1.0));
        assert!(buffer.write_sample(2.0));
        assert_eq!(buffer.available_read(), 2);

        assert_eq!(buffer.read_sample(), Some(1.0));
        assert_eq!(buffer.read_sample(), Some(2.0));
        assert_eq!(buffer.read_sample(), None);
    }

    #[test]
    fn test_buffer_full() {
        let buffer = AudioBuffer::new(4).unwrap();

        // Fill buffer (capacity - 1 to prevent write_pos == read_pos)
        assert!(buffer.write_sample(1.0));
        assert!(buffer.write_sample(2.0));
        assert!(buffer.write_sample(3.0));
        assert!(!buffer.write_sample(4.0)); // Should fail
    }

    #[test]
    fn test_multi_sample_operations() {
        let buffer = AudioBuffer::new(8).unwrap();
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0; 4];

        let written = buffer.write_samples(&input);
        assert_eq!(written, 4);

        let read = buffer.read_samples(&mut output);
        assert_eq!(read, 4);
        assert_eq!(output, input);
    }

    #[test]
    fn test_multi_channel_buffer() {
        let buffer = MultiChannelBuffer::new(2, 8).unwrap();
        assert_eq!(buffer.channel_count(), 2);

        let input = [1.0, 2.0, 3.0, 4.0]; // L, R, L, R
        let mut output = [0.0; 4];

        let written = buffer.write_interleaved(&input);
        assert_eq!(written, 4);

        let read = buffer.read_interleaved(&mut output);
        assert_eq!(read, 4);
        assert_eq!(output, input);
    }

    #[test]
    fn test_buffer_utilization() {
        let buffer = AudioBuffer::new(8).unwrap();
        assert_eq!(buffer.utilization(), 0.0);

        buffer.write_sample(1.0);
        buffer.write_sample(2.0);
        assert_eq!(buffer.utilization(), 0.25); // 2/8
    }

    #[test]
    fn test_buffer_clear() {
        let buffer = AudioBuffer::new(8).unwrap();
        buffer.write_sample(1.0);
        buffer.write_sample(2.0);

        assert!(!buffer.is_empty());
        buffer.clear();
        assert!(buffer.is_empty());
    }
}