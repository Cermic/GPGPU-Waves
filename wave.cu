////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/*
    This example demonstrates how to use the Cuda OpenGL bindings to
    dynamically modify a vertex buffer using a Cuda kernel.

    The steps are:
    1. Create an empty vertex buffer object (VBO)
    2. Register the VBO with Cuda
    3. Map the VBO for writing from Cuda
    4. Run Cuda kernel to modify the vertex positions
    5. Unmap the VBO
    6. Render the results using OpenGL

    Host code
*/

#define GLEW_STATIC
#define FREEGLUT_STATIC

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <device_launch_parameters.h>
#include <random>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>
#if defined (__APPLE__) || defined(MACOSX)
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#include <vector_types.h>
#include <array>
#include <thread>
#include <iostream>
#include <fstream>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

////////////////////////////////////////////////////////////////////////////////
// constants

// _WIN32 = we're in windows
#ifdef _WIN32
// Windows
static const unsigned int window_width = GetSystemMetrics(SM_CXSCREEN) - 20;
static const unsigned int window_height = GetSystemMetrics(SM_CYSCREEN) - 40;
#else
static const unsigned int window_width = 1280;
static const unsigned int window_height = 720;
#endif

const unsigned int mesh_width    = 128;
const unsigned int mesh_height   = 128;
float r = 0, g = 0, b = 1;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;
float * d_rand_buffer;
float g_fAnim = 0.0;


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

// Keyboard Inputs
struct Origin {
	float x = 0, y = 0, z = 0;
};

struct WaveProperties
{
	const float CIRCLE_RADIUS = 0.25;
	float freq = 3.0f;
	float amp = 0.5f;
	bool dna = false;
	bool jitter_on = false;
};
// Random number upper and lower limits
struct WaveLimits {
	float lower_limit = -0.05f, upper_limit = 0.05f;
};

Origin origin;
WaveLimits limits;
WaveProperties wp;
static const float INCREMENTER = 0.01f;
int hasRan =0;

// Defining random number buffer size and type.
static const int RAND_BUFFER_SIZE = (mesh_width * mesh_height * 4);
std::array <float, RAND_BUFFER_SIZE> rand_buffer;
std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937  gen{ rd() };

StopWatchInterface *timer = NULL;

// Auto-Verification Code
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
bool g_bQAReadback = false;

int *pArgc = NULL;
char **pArgv = NULL;

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void specialKeyboard(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);
void runAutoTest(int devID, char **argv, char *ref_file);
void checkResultCuda(int argc, char **argv, const GLuint &vbo);

///////////////////////////////////////////////////////////////////////////////
//! Generates a buffer of uniformly distributed values between 2 limits
///////////////////////////////////////////////////////////////////////////////

void RandomNoiseGeneration()
{
	std::uniform_real_distribution<> rands(limits.lower_limit, limits.upper_limit);
	for (int i = 0; i < rand_buffer.size(); i++)
	{
		rand_buffer[i] = (float)rands(gen);
		//rand_buffer[i] = ((rand() / (float)RAND_MAX) - 0.5f) * 0.05f; Different random function if you prefer.
	}
}

const char *sSDKsample = "simpleGL (VBO)";
///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! Generates a movable circle of calm within the wave pattern
//! @param pos - positions of the points in the mesh
//! @param width - mesh width
//! @param height - mesh height
//! @param time - system time
//! @param origin - origin point of the circle
//! @param wp - various wave attributes
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time, Origin origin , WaveProperties wp)
{
	
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate uv coordinatescuda
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0f - 1.0f; // Allows the circle to move left and right on the plane.
    v = v*2.0f - 1.0f; // Allows the circle to move up and down on the plane.

	// calculate simple sine wave pattern
	float w = sinf(u*wp.freq + time) * cosf(v*wp.freq + time) * wp.amp;
	if(wp.dna)
	{
		v = sinf(u*wp.freq + time) *cosf(v* wp.freq + time); //DNA Swizzle
	}

	float centreX = u - origin.x;
	float centreY = v - origin.z;
	// write output vertex
	if (sqrt(centreX*centreX + centreY*centreY) < wp.CIRCLE_RADIUS && !wp.dna) // Perimeter of circle
	{
		w = 0.0f + origin.y;
	}
	pos[y*width + x] = make_float4(u, w, v, 1.0f);
	// Change order or u, v and w to manipulate x, y or z being changed.
}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! Adds Jitter to the wave pattern
//! @param pos - positions of the points in the mesh
//! @param width - mesh width
//! @param rand_buffer - buffer of random floats to apply to the points on the mesh
///////////////////////////////////////////////////////////////////////////////
__global__ void jitter_kernel(float4 *pos, unsigned int width, float *rand_buffer)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// write output vertex
	int index = y*width + x;
	int b_index = index * 4;
	pos[index].x += rand_buffer[b_index];
	pos[index].y += rand_buffer[b_index +1];
	pos[index].z += rand_buffer[b_index +2];
}

void launch_kernel(float4 *pos, unsigned int mesh_width,
                   unsigned int mesh_height, float time)
{
	RandomNoiseGeneration();
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    simple_vbo_kernel<<< grid, block>>>(pos, mesh_width, mesh_height, time, origin, wp);
	if (wp.jitter_on)
	{
		jitter_kernel << < grid, block >> > (pos, mesh_width, d_rand_buffer);
	}
}

bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    char *ref_file = NULL;

    pArgc = &argc;
    pArgv = argv;

#if defined(__linux__)
    setenv ("DISPLAY", ":0", 0);
#endif

    printf("%s starting...\n", sSDKsample);

    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            // In this mode, we are running non-OpenGL and doing a compare of the VBO was generated correctly
            getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
        }
    }

	std::ifstream infile("hasran.txt");
	if (infile.is_open())
	{
		infile >> hasRan;
		std::cout << hasRan;
	}
	if (hasRan == 0)
	{
		printf("\n");
		// _WIN32 = we're in windows
		#ifdef _WIN32
		// Windows
		ShellExecute(0, 0, "https://i.imgur.com/RPXNRiy.jpg", 0, 0, SW_SHOW);
		#else
		// Not windows
		system("xdg-open https://i.imgur.com/RPXNRiy.jpg &");
		#endif

		std::ofstream outfile("hasran.txt");
		outfile << "1" << std::endl;
		outfile.close();
	}
	
    runTest(argc, argv, ref_file);

    printf("%s completed, returned %s\n", sSDKsample, (g_TotalErrors == 0) ? "OK" : "ERROR!");
	cudaDeviceReset();
	cudaFree(d_rand_buffer);
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}

void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Cuda GL Interop (VBO)");
    glutDisplayFunc(display);
	
    glutKeyboardFunc(keyboard); // Key inputs
	glutSpecialFunc(specialKeyboard); // Special key inputs
    glutMotionFunc(motion);
    glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    // initialize necessary OpenGL extensions
    if (! isGLVersionSupported(2,0))
    {
        fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush(stderr);
        return false;
    }

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    SDK_CHECK_ERROR_GL();

    return true;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    // command line mode only
    if (ref_file != NULL)
    {
        // create VBO
        checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, mesh_width*mesh_height*4*sizeof(float)));

        // run the cuda part
        runAutoTest(devID, argv, ref_file);

        // check result of Cuda step
        checkResultCuda(argc, argv, vbo);

        cudaFree(d_vbo_buffer);
        d_vbo_buffer = NULL;
    }
    else
    {
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
		glutSpecialFunc(specialKeyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        // create VBO
        createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        
		// Allocate memory for the random buffer
		checkCudaErrors(cudaMalloc((void **)&d_rand_buffer, rand_buffer.size() * sizeof(float)));
		// run the cuda part
    	runCuda(&cuda_vbo_resource);

        // start rendering mainloop
        glutMainLoop();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;


    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
                                                         *vbo_resource));
	// Copy the buffer to the GPU
	checkCudaErrors(cudaMemcpy(d_rand_buffer, rand_buffer.data(), rand_buffer.size() * sizeof(float), cudaMemcpyHostToDevice));

    launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE *fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
    char *reference_file = NULL;
    void *imageData = malloc(mesh_width*mesh_height*sizeof(float));

    // execute the kernel
    launch_kernel((float4 *)d_vbo_buffer, mesh_width, mesh_height, g_fAnim);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, mesh_width*mesh_height*sizeof(float), cudaMemcpyDeviceToHost));

    sdkDumpBin2(imageData, mesh_width*mesh_height*sizeof(float), "simpleGL.bin");
    reference_file = sdkFindFilePath(ref_file, argv[0]);

    if (reference_file &&
        !sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
                                mesh_width*mesh_height*sizeof(float),
                                MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
    {
        g_TotalErrors++;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(vbo);

    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    *vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
	
	glColor3f(r, g, b);
	// Deafult point size is always 1 so we can use glPointSize here to increase their size
	glPointSize(2.0f);
	glEnable(GL_POINT_SMOOTH); // Anti aliases the points down into circles.
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();

    g_fAnim += 0.05f;
    sdkStopTimer(&timer);
    computeFPS();
}

void timerEvent(int value)
{
    if (glutGetWindow())
    {
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    }
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBO(&vbo, cuda_vbo_resource);
    }
}


void specialKeyboard(int key, int x, int y)
{
	switch (key)
	{
	case GLUT_KEY_UP:
		if (limits.upper_limit < (INCREMENTER * 50))
		{
			limits.lower_limit -= INCREMENTER;
			limits.upper_limit += INCREMENTER;
		}
		break;
	case GLUT_KEY_DOWN:
		if (limits.lower_limit < -INCREMENTER)
		{
			limits.lower_limit += INCREMENTER;
			limits.upper_limit -= INCREMENTER;
		}
		else
		{
			limits.lower_limit = -INCREMENTER;
			limits.upper_limit = INCREMENTER;
		}
		break;
	case GLUT_KEY_RIGHT:
		wp.freq += INCREMENTER * 10;
		break;
	case GLUT_KEY_LEFT:
		wp.freq -= INCREMENTER * 10;
		break;
	}
	
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	const float CIRCLE_LIMIT = 0.9f;
    switch (key)
    {
        case (27) :
            #if defined(__APPLE__) || defined(MACOSX)
                exit(EXIT_SUCCESS);
            #else
                glutDestroyWindow(glutGetWindow());
                return;
            #endif
		case 'a':
			if (origin.x > -CIRCLE_LIMIT)
			{
				origin.x -= INCREMENTER *3;
			}
			break;
		case 'd':
			if (origin.x < CIRCLE_LIMIT)
			{
				origin.x += INCREMENTER *3;
			}
			break;
		case 'w':
			if (origin.z > -CIRCLE_LIMIT)
			{
				origin.z -= INCREMENTER * 3;
			}
			break;
		case 's':
			if (origin.z < CIRCLE_LIMIT)
			{
				origin.z += INCREMENTER * 3;
			}
			break;
		case 'q':
			if (origin.y < CIRCLE_LIMIT)
			{
				origin.y += INCREMENTER * 3;
			}
			break;
		case 'e':
			if (origin.y > -CIRCLE_LIMIT)
			{
				origin.y -= INCREMENTER * 3;
			}
			break;
		case 'j':
			wp.jitter_on = !wp.jitter_on;
			break;
		case 'f':
			wp.dna = !wp.dna;
			break;
		case 'r':
			if (r>=1.0)
			{
				r = 0;
			}
			r += INCREMENTER * 5;
			break;
		case 'g':
		if(g >=1.0)
			{
				g = 0;
			}
			g += INCREMENTER * 5;
			break;
		case 'b':
			if (b >= 1.0)
			{
				b = 0;
			}
			b += INCREMENTER * 5;
			break;
		case 'z':
			wp.amp -= INCREMENTER;
			break;
		case 'x':
			wp.amp += INCREMENTER;
			break;
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
    if (!d_vbo_buffer)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

        // map buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // check result
        if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
        {
            // write file for regression test
            sdkWriteFile<float>("./data/regression.dat",
                                data, mesh_width * mesh_height * 3, 0.0, false);
        }

        // unmap GL buffer object
        if (!glUnmapBuffer(GL_ARRAY_BUFFER))
        {
            fprintf(stderr, "Unmap buffer failed.\n");
            fflush(stderr);
        }

        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsWriteDiscard));

        SDK_CHECK_ERROR_GL();
    }
}
