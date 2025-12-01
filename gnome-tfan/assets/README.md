# Cockpit Assets

This directory contains media assets for the T-FAN Cockpit HUD.

## Video Backgrounds

Place looping video files here for the cockpit background animation.

### Supported Formats
- WebM (VP8/VP9) - Preferred
- MP4 (H.264)

### Expected Files
The video background module searches for these files in order:
1. `hologram_loop.webm`
2. `hologram_loop.mp4`
3. `robot_loop.webm`
4. `background.webm`

### Specifications
- **Resolution**: 800×1280 (portrait) or 1080×1920
- **Frame rate**: 24-30 fps
- **Duration**: 10-30 seconds (loops seamlessly)
- **File size**: < 50MB for smooth playback
- **Content**: Low contrast, subtle animations work best

### Recommended Content Types
- Hologram grids with rotating wireframes
- Flowing data streams / particle trails
- Subtle robot/android silhouettes
- Animated circuit board traces
- Pulsing neural network nodes

### Creating Videos

#### Using Blender
```python
# Example Blender script for hologram grid
import bpy
bpy.ops.mesh.primitive_grid_add(x_subdivisions=20, y_subdivisions=20)
# Add wireframe modifier, emission material, animate rotation
```

#### Using ffmpeg to convert
```bash
# Convert MP4 to WebM
ffmpeg -i input.mp4 -c:v libvpx-vp9 -crf 30 -b:v 0 -an hologram_loop.webm

# Scale to portrait
ffmpeg -i input.webm -vf "scale=800:1280" -an hologram_loop.webm
```

### Opacity
The video will be displayed at 20% opacity by default, so high-contrast
content will still show through. Consider this when designing.

---

If no video is found, the cockpit falls back to a CSS-animated gradient background.
