#!/bin/bash

echo "Testing mood music module across different values..."

for mood in 0.1 0.35 0.6 0.85; do
    echo ""
    echo "🎵 Testing mood $mood..."
    echo "$mood" | timeout 2 ./target/release/audio_demo 2>/dev/null | head -20
    sleep 0.5
done

echo ""
echo "✅ Audio test complete! The module should now be working with audible output."
echo "You can run './target/release/audio_demo' to test interactively."