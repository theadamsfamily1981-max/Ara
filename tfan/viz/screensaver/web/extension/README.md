# T-FAN Topology Screensaver - Browser Extension

Replace your browser's new tab page with living topology visualizations.

## Installation

### Chrome / Edge / Brave

1. **Copy files to extension directory:**
   ```bash
   cd tfan/viz/screensaver/web
   cp index.html screensaver.js styles.css extension/
   ```

2. **Open Chrome extensions page:**
   - Navigate to `chrome://extensions/`
   - Enable "Developer mode" (top right toggle)

3. **Load extension:**
   - Click "Load unpacked"
   - Select the `extension/` directory
   - Extension should appear in your list

4. **Test:**
   - Open a new tab
   - Should see T-FAN screensaver
   - Press H for help, M to cycle modes

### Firefox

1. **Package as XPI:**
   ```bash
   cd extension
   zip -r ../tfan-screensaver.xpi *
   ```

2. **Install:**
   - Navigate to `about:addons`
   - Click gear icon → "Install Add-on From File"
   - Select `tfan-screensaver.xpi`

## Configuration

### Metrics Connection

By default, the extension tries to connect to:
- WebSocket: `ws://localhost:8000/ws/metrics`
- HTTP: `http://localhost:8000/api/metrics`

To use custom endpoints, edit `screensaver.js`:

```javascript
getWebSocketURL() {
    // Your custom metrics server
    return 'ws://your-server:port/ws/metrics';
}
```

### Icons

Replace placeholder icons with proper ones:

```bash
# Create icons from SVG or PNG
convert -resize 16x16 icon.svg icon16.png
convert -resize 48x48 icon.svg icon48.png
convert -resize 128x128 icon.svg icon128.png
```

## Permissions

The extension requests:

- **storage**: Save user preferences (future feature)
- **host_permissions**: Access localhost metrics API
  - `http://localhost:8000/*` - T-FAN API
  - `http://localhost:9101/*` - Metrics bridge

To allow remote metrics servers, add to `manifest.json`:

```json
"host_permissions": [
    "http://localhost:8000/*",
    "https://your-metrics-server.com/*"
]
```

## Publishing

### Chrome Web Store

1. **Prepare package:**
   ```bash
   cd extension
   zip -r tfan-screensaver.zip *
   ```

2. **Upload:**
   - Go to [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole)
   - Create new item
   - Upload ZIP
   - Fill in store listing
   - Submit for review

### Firefox Add-ons

1. **Package:**
   ```bash
   cd extension
   zip -r ../tfan-screensaver.xpi *
   ```

2. **Upload:**
   - Go to [Firefox Add-on Developer Hub](https://addons.mozilla.org/developers/)
   - Submit new add-on
   - Upload XPI
   - Fill in listing
   - Submit for review

## Features

- ✅ Replaces new tab page
- ✅ Live metrics from T-FAN API
- ✅ 4 visualization modes
- ✅ Keyboard controls
- ✅ Fullscreen support
- ✅ Auto-rotate camera
- ⬜ Settings page (coming soon)
- ⬜ Offline mode (coming soon)

## Troubleshooting

**Extension won't load:**
- Check manifest.json is valid JSON
- Ensure all files are in extension/ directory
- Look for errors in chrome://extensions/ (click "Errors")

**Metrics not connecting:**
- Ensure T-FAN API is running on localhost:8000
- Check host_permissions in manifest.json
- Look at browser console (F12) for CORS errors

**Performance issues:**
- Reduce particle count in screensaver.js
- Lower pixel ratio
- Disable auto-rotate

## Development

### Hot Reload

Chrome extensions don't hot-reload by default. After changes:

1. Edit files
2. Go to `chrome://extensions/`
3. Click reload icon on extension card
4. Open new tab to see changes

### Debugging

- Right-click on new tab → "Inspect"
- Console shows logs from screensaver.js
- Network tab shows WebSocket/HTTP requests

## Privacy

This extension:
- ✅ No data collection
- ✅ No analytics
- ✅ No external requests (except your metrics server)
- ✅ All processing client-side
- ✅ Open source

## License

Part of the T-FAN project. MIT License.
