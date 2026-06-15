#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "metal_sim.hpp"
#include <cassert>
#include <cstdio>

static_assert(sizeof(Particle) == 24,
              "Particle layout must match Metal shader");


static const char *SHADER_PATH = "src/shaders.metal";

static NSString *load_shader_source() {
    NSString *path =
        [[NSString stringWithUTF8String:SHADER_PATH] stringByStandardizingPath];
    NSError *err = nil;
    NSString *src = [NSString stringWithContentsOfFile:path
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (!src) { fprintf(stderr, "Could not read %s: %s\n", [path UTF8String], [[err localizedDescription] UTF8String]);
    }
    return src;
}

struct MetalSim {
    id<MTLDevice> device;
    id<MTLLibrary> lib;
    id<MTLCommandQueue> queue;

    id<MTLComputePipelineState> clearPipeline;
    id<MTLComputePipelineState> countPipeline;
    id<MTLComputePipelineState> scatterPipeline;
    id<MTLComputePipelineState> gridForcePipeline;

    id<MTLRenderPipelineState> renderPipeline;
    CAMetalLayer *metalLayer;

    id<MTLBuffer> particleBuf;
    id<MTLBuffer> sortedBuf;
    id<MTLBuffer> cellCountsBuf;
    id<MTLBuffer> cellStartsBuf;
    id<MTLBuffer> cellWriteBuf;
    id<MTLBuffer> matrixBuf;
    id<MTLBuffer> minRadBuf;
    id<MTLBuffer> maxRadBuf;
    id<MTLBuffer> colorsBuf;

    uint32_t count;
    uint32_t numTypes;
    uint32_t gridDim;
    uint32_t gridNCells;
    int srad;
    uint32_t tgsz;
    dispatch_semaphore_t frameSema;
};

static id<MTLComputePipelineState>
make_pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn =
        [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        fprintf(stderr, "Function '%s' not found\n", name);
        return nil;
    }
    id<MTLComputePipelineState> ps =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps)
        fprintf(stderr, "Pipeline '%s': %s\n", name,
                [[err localizedDescription] UTF8String]);
    return ps;
}

MetalSim *metal_sim_create(Particle *initial, uint32_t count,
                           uint32_t num_types, uint32_t grid_dim, int srad, float rforce) {
    MetalSim *sim = new MetalSim();
    sim->count = count;
    sim->numTypes = num_types;
    sim->gridDim = grid_dim;
    sim->gridNCells = grid_dim * grid_dim;
    sim->srad = srad;
    sim->metalLayer = nil;
    sim->renderPipeline = nil;
    sim->frameSema = dispatch_semaphore_create(1);

    sim->device = MTLCreateSystemDefaultDevice();
    if (!sim->device) {
        fprintf(stderr, "No Metal device\n");
        return nullptr;
    }
    sim->queue = [sim->device newCommandQueue];

    NSString *shaderSrc = load_shader_source();
    if (!shaderSrc)
        return nullptr;

    MTLCompileOptions *options = [MTLCompileOptions new];
    options.preprocessorMacros = @{
        @"GDIM" : @(grid_dim),
        @"SRAD" : @(srad),
        @"MAX_TYPES" : @(num_types),
        @"REPEL_FORCE" : @(rforce)
    };

    NSError *err = nil;
    sim->lib = [sim->device newLibraryWithSource:shaderSrc
                                         options:options
                                           error:&err];
    if (!sim->lib) {
        fprintf(stderr, "Shader compile: %s\n",
                [[err localizedDescription] UTF8String]);
        return nullptr;
    }

    sim->clearPipeline = make_pipeline(sim->device, sim->lib, "clear_cells");
    sim->countPipeline = make_pipeline(sim->device, sim->lib, "count_cells");
    sim->scatterPipeline =
        make_pipeline(sim->device, sim->lib, "scatter_particles");
    sim->gridForcePipeline = make_pipeline(sim->device, sim->lib, "grid_force");
    if (!sim->clearPipeline || !sim->countPipeline || !sim->scatterPipeline ||
        !sim->gridForcePipeline)
        return nullptr;

    sim->tgsz = (uint32_t)sim->gridForcePipeline.maxTotalThreadsPerThreadgroup;
    if (sim->tgsz > 512)
        sim->tgsz = 512;

    auto buf = [&](size_t bytes) {
        return [sim->device newBufferWithLength:bytes
                                        options:MTLResourceStorageModeShared];
    };

    sim->particleBuf =
        [sim->device newBufferWithBytes:initial
                                 length:count * sizeof(Particle)
                                options:MTLResourceStorageModeShared];
    sim->sortedBuf = buf(count * sizeof(Particle));
    sim->cellCountsBuf = buf(sim->gridNCells * sizeof(uint32_t));
    sim->cellStartsBuf = buf(sim->gridNCells * sizeof(uint32_t));
    sim->cellWriteBuf = buf(sim->gridNCells * sizeof(uint32_t));

    size_t matSz = num_types * num_types * sizeof(float);
    sim->matrixBuf = buf(matSz);
    sim->minRadBuf = buf(matSz);
    sim->maxRadBuf = buf(matSz);
    sim->colorsBuf = buf(num_types * 4 * sizeof(float));

    float *m = (float *)sim->matrixBuf.contents;
    float *mn = (float *)sim->minRadBuf.contents;
    float *mx = (float *)sim->maxRadBuf.contents;
    float *c = (float *)sim->colorsBuf.contents;
    for (uint32_t i = 0; i < num_types * num_types; i++) {
        m[i] = 1.0f;
        mn[i] = 0.0f;
        mx[i] = 0.5f;
    }
    for (uint32_t i = 0; i < num_types * 4; i++)
        c[i] = 1.0f;

    printf(
        "Metal ready — %s | tgsz %u | %u particles | %u types | grid %ux%u (rad %d)\n",
        [sim->device.name UTF8String], sim->tgsz, count, num_types,
        sim->gridDim, sim->gridDim, sim->srad);
    return sim;
}

void metal_render_init(MetalSim *sim, GLFWwindow *glfwWin, int fbw, int fbh) {
    NSWindow *win = glfwGetCocoaWindow(glfwWin);

    sim->metalLayer = [CAMetalLayer layer];
    sim->metalLayer.device = sim->device;
    sim->metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    sim->metalLayer.drawableSize = CGSizeMake(fbw, fbh);
    sim->metalLayer.contentsScale = win.backingScaleFactor;
    sim->metalLayer.framebufferOnly = YES;

    NSView *view = win.contentView;
    view.wantsLayer = YES;
    view.layer = sim->metalLayer;

    NSError *err = nil;

    MTLRenderPipelineDescriptor *rpd = [MTLRenderPipelineDescriptor new];
    rpd.vertexFunction   = [sim->lib newFunctionWithName:@"particle_vert"];
    rpd.fragmentFunction = [sim->lib newFunctionWithName:@"particle_frag"];
    rpd.colorAttachments[0].pixelFormat              = MTLPixelFormatBGRA8Unorm;
    rpd.colorAttachments[0].blendingEnabled          = YES;
    rpd.colorAttachments[0].sourceRGBBlendFactor     = MTLBlendFactorSourceAlpha;
    rpd.colorAttachments[0].destinationRGBBlendFactor= MTLBlendFactorOne;
    rpd.colorAttachments[0].sourceAlphaBlendFactor   = MTLBlendFactorOne;
    rpd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorZero;
    sim->renderPipeline =
        [sim->device newRenderPipelineStateWithDescriptor:rpd error:&err];
    if (!sim->renderPipeline)
        fprintf(stderr, "Render pipeline: %s\n",
                [[err localizedDescription] UTF8String]);
}

void metal_step_and_render(MetalSim *sim, float dt, float G, float softening2,
                           float drag, Camera cam) {
    if (!sim->metalLayer || !sim->renderPipeline)
        return;

    struct GridParams {
        uint32_t N, num_types;
        float inv_cell, dt, G, softening2, drag;
    };
    GridParams gp = {sim->count, sim->numTypes, (float)sim->gridDim / 2.0f, dt,
                     G, softening2, drag};

    uint32_t pGroups = (sim->count + sim->tgsz - 1) / sim->tgsz;
    uint32_t cGroups = (sim->gridNCells + sim->tgsz - 1) / sim->tgsz;

    {
        id<MTLCommandBuffer> cb = [sim->queue commandBuffer];

        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:sim->clearPipeline];
        [ce setBuffer:sim->cellCountsBuf offset:0 atIndex:0];
        [ce dispatchThreadgroups:MTLSizeMake(cGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(sim->tgsz, 1, 1)];
        [ce endEncoding];

        ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:sim->countPipeline];
        [ce setBuffer:sim->particleBuf offset:0 atIndex:0];
        [ce setBuffer:sim->cellCountsBuf offset:0 atIndex:1];
        [ce setBytes:&gp length:sizeof(gp) atIndex:2];
        [ce dispatchThreadgroups:MTLSizeMake(pGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(sim->tgsz, 1, 1)];
        [ce endEncoding];

        [cb commit];
        [cb waitUntilCompleted];
    }

    {
        const uint32_t *counts = (const uint32_t *)sim->cellCountsBuf.contents;
        uint32_t *starts = (uint32_t *)sim->cellStartsBuf.contents;
        uint32_t *write = (uint32_t *)sim->cellWriteBuf.contents;
        uint32_t offset = 0;
        for (uint32_t c = 0; c < sim->gridNCells; c++) {
            starts[c] = offset;
            write[c] = offset;
            offset += counts[c];
        }
    }

    dispatch_semaphore_wait(sim->frameSema, DISPATCH_TIME_FOREVER);

    id<CAMetalDrawable> drawable = [sim->metalLayer nextDrawable];
    if (!drawable) {
        dispatch_semaphore_signal(sim->frameSema);
        return;
    }

    id<MTLCommandBuffer> cb = [sim->queue commandBuffer];

    {
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:sim->scatterPipeline];
        [ce setBuffer:sim->particleBuf offset:0 atIndex:0];
        [ce setBuffer:sim->sortedBuf offset:0 atIndex:1];
        [ce setBuffer:sim->cellWriteBuf offset:0 atIndex:2];
        [ce setBytes:&gp length:sizeof(gp) atIndex:3];
        [ce dispatchThreadgroups:MTLSizeMake(pGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(sim->tgsz, 1, 1)];
        [ce endEncoding];
    }

    {
        id<MTLComputeCommandEncoder> ce = [cb computeCommandEncoder];
        [ce setComputePipelineState:sim->gridForcePipeline];
        [ce setBuffer:sim->sortedBuf offset:0 atIndex:0];
        [ce setBuffer:sim->particleBuf offset:0 atIndex:1];
        [ce setBuffer:sim->cellStartsBuf offset:0 atIndex:2];
        [ce setBuffer:sim->cellCountsBuf offset:0 atIndex:3];
        [ce setBytes:&gp length:sizeof(gp) atIndex:4];
        [ce setBuffer:sim->matrixBuf offset:0 atIndex:5];
        [ce setBuffer:sim->minRadBuf offset:0 atIndex:6];
        [ce setBuffer:sim->maxRadBuf offset:0 atIndex:7];
        [ce dispatchThreadgroups:MTLSizeMake(pGroups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(sim->tgsz, 1, 1)];
        [ce endEncoding];
    }

    {
        MTLRenderPassDescriptor *rpd =
            [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture     = drawable.texture;
        rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
        rpd.colorAttachments[0].clearColor  = MTLClearColorMake(0.02, 0.02, 0.06, 1.0);
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

        id<MTLRenderCommandEncoder> re =
            [cb renderCommandEncoderWithDescriptor:rpd];
        [re setRenderPipelineState:sim->renderPipeline];
        [re setVertexBuffer:sim->particleBuf offset:0 atIndex:0];
        [re setVertexBytes:&cam length:sizeof(cam) atIndex:1];
        [re setVertexBuffer:sim->colorsBuf offset:0 atIndex:2];
        [re drawPrimitives:MTLPrimitiveTypePoint
               vertexStart:0
               vertexCount:sim->count];
        [re endEncoding];
    }

    __block dispatch_semaphore_t sema = sim->frameSema;
    [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
      dispatch_semaphore_signal(sema);
    }];
    [cb presentDrawable:drawable];
    [cb commit];
}

void metal_resize(MetalSim *sim, int fbw, int fbh) {
    if (sim->metalLayer)
        sim->metalLayer.drawableSize = CGSizeMake(fbw, fbh);
}

float *metal_sim_matrix(MetalSim *sim) {
    return (float *)sim->matrixBuf.contents;
}
float *metal_sim_min_radius(MetalSim *sim) {
    return (float *)sim->minRadBuf.contents;
}
float *metal_sim_max_radius(MetalSim *sim) {
    return (float *)sim->maxRadBuf.contents;
}
float *metal_sim_colors(MetalSim *sim) {
    return (float *)sim->colorsBuf.contents;
}

void metal_sim_destroy(MetalSim *sim) { delete sim; }
