#version 330 core

/**
 * Fullscreen Fragment Shader
 * ==========================
 *
 * Samples the quilt texture and displays it on screen.
 *
 * For holographic display output, this is where you would apply
 * the lenticular/parallax barrier mapping to interleave views.
 */

in vec2 vUV;
out vec4 fragColor;

uniform sampler2D quiltTex;

void main() {
    // Simple passthrough - just display the quilt as-is
    // For actual holographic display, replace with view interleaving
    fragColor = texture(quiltTex, vUV);
}
