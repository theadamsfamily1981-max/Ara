#version 330 core

/**
 * Fullscreen Quad Vertex Shader
 * =============================
 *
 * Simple passthrough for fullscreen quad rendering.
 */

layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;

out vec2 vUV;

void main() {
    vUV = inUV;
    gl_Position = vec4(inPos, 0.0, 1.0);
}
