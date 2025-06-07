#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>



const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;


class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window;

    void initWindow() {
        glfwInit();         // initializes the GLFW library

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);       // specifies that we do not want to use OpenGL or OpenGL ES
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);         // specifies that we do not want the window to be resizable

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);   // creates a window with the specified width, height, and title
    }

    void initVulkan() {
        
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }

    }

    void cleanup() {
        glfwDestroyWindow(window);

        glfwTerminate();
    }
};

int main() {
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}