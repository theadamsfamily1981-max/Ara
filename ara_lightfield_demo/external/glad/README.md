# GLAD OpenGL Loader

This directory should contain GLAD files generated from https://glad.dav1d.de/

## Generation Settings

1. Go to https://glad.dav1d.de/
2. Select:
   - Language: C/C++
   - Specification: OpenGL
   - API: gl Version 4.5
   - Profile: Core
   - Options: Generate a loader
3. Click Generate
4. Download the zip file
5. Extract:
   - `src/glad.c` → `external/glad/src/glad.c`
   - `include/glad/glad.h` → `external/glad/include/glad/glad.h`
   - `include/KHR/khrplatform.h` → `external/glad/include/KHR/khrplatform.h`

## Quick Command (if you have Python + glad installed)

```bash
pip install glad
glad --generator c --out-path external/glad --api gl:core=4.5
```

## Directory Structure

```
external/glad/
├── include/
│   ├── glad/
│   │   └── glad.h
│   └── KHR/
│       └── khrplatform.h
├── src/
│   └── glad.c
└── README.md
```
