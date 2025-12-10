/**
 * Ara Light-Field Quilt Demo
 * ===========================
 *
 * Renders a multi-view quilt (8x8 views) using GPU compute shaders.
 * Each view shows the scene from a different camera position on a ring.
 *
 * This is the foundation for holographic / lenticular display output.
 *
 * Build:
 *   mkdir build && cd build && cmake .. && make
 *
 * Run:
 *   ./ara_lightfield_demo
 */

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

static std::string loadFile(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open file: " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static GLuint compileShader(GLenum type, const std::string& src) {
    GLuint shader = glCreateShader(type);
    const char* csrc = src.c_str();
    glShaderSource(shader, 1, &csrc, nullptr);
    glCompileShader(shader);

    GLint status = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        GLint logLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
        std::string log(logLen, '\0');
        glGetShaderInfoLog(shader, logLen, nullptr, log.data());
        throw std::runtime_error("Shader compile error:\n" + log);
    }
    return shader;
}

static GLuint linkProgram(std::initializer_list<GLuint> shaders) {
    GLuint prog = glCreateProgram();
    for (auto s : shaders) glAttachShader(prog, s);
    glLinkProgram(prog);
    GLint status = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    if (!status) {
        GLint logLen = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &logLen);
        std::string log(logLen, '\0');
        glGetProgramInfoLog(prog, logLen, nullptr, log.data());
        throw std::runtime_error("Program link error:\n" + log);
    }
    for (auto s : shaders) glDeleteShader(s);
    return prog;
}

static void APIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id,
                                   GLenum severity, GLsizei length,
                                   const GLchar* message, const void* userParam) {
    // Ignore non-significant error/warning codes
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    std::cerr << "GL Debug [" << id << "]: " << message << "\n";

    switch (source) {
        case GL_DEBUG_SOURCE_API:             std::cerr << "  Source: API\n"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cerr << "  Source: Window System\n"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cerr << "  Source: Shader Compiler\n"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cerr << "  Source: Third Party\n"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     std::cerr << "  Source: Application\n"; break;
        case GL_DEBUG_SOURCE_OTHER:           std::cerr << "  Source: Other\n"; break;
    }

    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:         std::cerr << "  Severity: HIGH\n"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       std::cerr << "  Severity: MEDIUM\n"; break;
        case GL_DEBUG_SEVERITY_LOW:          std::cerr << "  Severity: LOW\n"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: std::cerr << "  Severity: NOTIFICATION\n"; break;
    }
}

int main() {
    // --- Init GLFW ---
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n";
        return -1;
    }

    // Core profile, 4.5 for compute shader
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

    int windowWidth  = 1280;
    int windowHeight = 720;
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight,
                                          "Ara Light-Field Quilt Demo", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // --- Init GLAD ---
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD\n";
        return -1;
    }

    std::cout << "=== Ara Light-Field Quilt Demo ===\n";
    std::cout << "GL Version: " << glGetString(GL_VERSION) << "\n";
    std::cout << "GL Renderer: " << glGetString(GL_RENDERER) << "\n";

    // Enable debug output
    GLint flags;
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(glDebugOutput, nullptr);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    }

    // --- Quilt parameters ---
    // 8x8 views = 64 total views for holographic display
    const int quiltCols = 8;
    const int quiltRows = 8;
    const int viewW     = 320;  // Per-view resolution
    const int viewH     = 180;
    const int quiltW    = quiltCols * viewW;  // 2560
    const int quiltH    = quiltRows * viewH;  // 1440

    std::cout << "Quilt: " << quiltCols << "x" << quiltRows << " views @ "
              << viewW << "x" << viewH << " each\n";
    std::cout << "Total quilt resolution: " << quiltW << "x" << quiltH << "\n";

    // --- Quilt texture ---
    GLuint quiltTex = 0;
    glGenTextures(1, &quiltTex);
    glBindTexture(GL_TEXTURE_2D, quiltTex);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, quiltW, quiltH);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);

    // --- Shaders ---
    try {
        std::string compSrc = loadFile("shaders/quilt.comp");
        GLuint compShader   = compileShader(GL_COMPUTE_SHADER, compSrc);
        GLuint computeProg  = linkProgram({compShader});

        std::string vsSrc = loadFile("shaders/fullscreen.vert");
        std::string fsSrc = loadFile("shaders/fullscreen.frag");
        GLuint vs = compileShader(GL_VERTEX_SHADER,   vsSrc);
        GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
        GLuint drawProg = linkProgram({vs, fs});

        // --- Fullscreen quad VAO/VBO ---
        float quadVertices[] = {
            // positions   // texcoords
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f,

            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f,
            -1.0f,  1.0f,  0.0f, 1.0f
        };

        GLuint quadVAO = 0, quadVBO = 0;
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              (void*)(2 * sizeof(float)));

        glBindVertexArray(0);

        // Uniform locations (compute)
        glUseProgram(computeProg);
        GLint locQuiltCols = glGetUniformLocation(computeProg, "quiltCols");
        GLint locQuiltRows = glGetUniformLocation(computeProg, "quiltRows");
        GLint locViewW     = glGetUniformLocation(computeProg, "viewW");
        GLint locViewH     = glGetUniformLocation(computeProg, "viewH");
        GLint locTime      = glGetUniformLocation(computeProg, "time");
        glUniform1i(locQuiltCols, quiltCols);
        glUniform1i(locQuiltRows, quiltRows);
        glUniform1i(locViewW, viewW);
        glUniform1i(locViewH, viewH);
        glUseProgram(0);

        auto startTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        auto lastFPSTime = startTime;

        // --- Main loop ---
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // Handle ESC key
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                glfwSetWindowShouldClose(window, true);
            }

            auto now = std::chrono::high_resolution_clock::now();
            float t = std::chrono::duration<float>(now - startTime).count();

            // 1) Run compute shader to fill quilt
            glUseProgram(computeProg);

            glUniform1f(locTime, t);

            // Bind quilt as image for write
            glBindImageTexture(0, quiltTex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F);

            // Dispatch groups (16x16 local size)
            const int localSizeX = 16;
            const int localSizeY = 16;
            GLuint groupsX = (quiltW + localSizeX - 1) / localSizeX;
            GLuint groupsY = (quiltH + localSizeY - 1) / localSizeY;
            glDispatchCompute(groupsX, groupsY, 1);

            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            // 2) Draw quilt to screen
            int fbW, fbH;
            glfwGetFramebufferSize(window, &fbW, &fbH);
            glViewport(0, 0, fbW, fbH);
            glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glUseProgram(drawProg);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, quiltTex);
            GLint locTex = glGetUniformLocation(drawProg, "quiltTex");
            glUniform1i(locTex, 0);

            glBindVertexArray(quadVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);

            glfwSwapBuffers(window);

            // FPS counter
            frameCount++;
            auto fpsElapsed = std::chrono::duration<float>(now - lastFPSTime).count();
            if (fpsElapsed >= 1.0f) {
                float fps = frameCount / fpsElapsed;
                std::cout << "\rFPS: " << fps << "  Quilt: " << quiltW << "x" << quiltH
                          << " (" << (quiltCols * quiltRows) << " views)    " << std::flush;
                frameCount = 0;
                lastFPSTime = now;
            }
        }

        std::cout << "\n";

        glDeleteProgram(computeProg);
        glDeleteProgram(drawProg);
        glDeleteTextures(1, &quiltTex);
        glDeleteBuffers(1, &quadVBO);
        glDeleteVertexArrays(1, &quadVAO);

    } catch (const std::exception& e) {
        std::cerr << "Error during setup: " << e.what() << "\n";
        glfwTerminate();
        return -1;
    }

    glfwTerminate();
    return 0;
}
