#include <SDL3/SDL.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_main.h>
#include <SDL3_image/SDL_image.h>
#include <SDL3_mixer/SDL_mixer.h>
#include <SDL3_ttf/SDL_ttf.h>
#include <cmath>
#include <filesystem>
#include <string_view>
#include <thread>

// Shaders
#include "testgputext/shaders/shader-sdf.frag.dxil.h"
#include "testgputext/shaders/shader-sdf.frag.msl.h"
#include "testgputext/shaders/shader-sdf.frag.spv.h"
#include "testgputext/shaders/shader.frag.dxil.h"
#include "testgputext/shaders/shader.frag.msl.h"
#include "testgputext/shaders/shader.frag.spv.h"
#include "testgputext/shaders/shader.vert.dxil.h"
#include "testgputext/shaders/shader.vert.msl.h"
#include "testgputext/shaders/shader.vert.spv.h"

#define SDL_MATH_3D_IMPLEMENTATION
#include "testgputext/SDL_math3d.h"

#define MAX_VERTEX_COUNT 4000
#define MAX_INDEX_COUNT 6000
#define SUPPORTED_SHADER_FORMATS                                               \
  (SDL_GPU_SHADERFORMAT_SPIRV | SDL_GPU_SHADERFORMAT_DXIL |                    \
   SDL_GPU_SHADERFORMAT_MSL)

SDL_AppResult SDL_Fail() {
  SDL_LogError(SDL_LOG_CATEGORY_CUSTOM, "Error %s", SDL_GetError());
  return SDL_APP_FAILURE;
}

typedef enum {
  VertexShader,
  PixelShader,
  PixelShader_SDF,
} Shader;

typedef SDL_FPoint Vec2;

typedef struct Vec3 {
  float x, y, z;
} Vec3;

typedef struct Vertex {
  Vec3 pos;
  SDL_FColor colour;
  Vec2 uv;
} Vertex;

typedef struct Context {
  SDL_GPUDevice *device;
  SDL_Window *window;
  SDL_GPUGraphicsPipeline *pipeline;
  SDL_GPUBuffer *vertex_buffer;
  SDL_GPUBuffer *index_buffer;
  SDL_GPUTransferBuffer *transfer_buffer;
  SDL_GPUSampler *sampler;
  SDL_GPUCommandBuffer *cmd_buf;
} Context;

typedef struct GeometryData {
  Vertex *vertices;
  int vertex_count;
  int *indices;
  int index_count;
} GeometryData;

void check_error_bool(const bool res) {
  if (!res) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "%s", SDL_GetError());
  }
}

void *check_error_ptr(void *ptr) {
  if (!ptr) {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "%s", SDL_GetError());
  }

  return ptr;
}

SDL_GPUShader *load_shader(SDL_GPUDevice *device, Shader shader,
                           Uint32 sampler_count, Uint32 uniform_buffer_count,
                           Uint32 storage_buffer_count,
                           Uint32 storage_texture_count) {
  SDL_GPUShaderCreateInfo createinfo;
  createinfo.num_samplers = sampler_count;
  createinfo.num_storage_buffers = storage_buffer_count;
  createinfo.num_storage_textures = storage_texture_count;
  createinfo.num_uniform_buffers = uniform_buffer_count;
  createinfo.props = 0;

  SDL_GPUShaderFormat format = SDL_GetGPUShaderFormats(device);
  if (format & SDL_GPU_SHADERFORMAT_DXIL) {
    createinfo.format = SDL_GPU_SHADERFORMAT_DXIL;
    switch (shader) {
    case VertexShader:
      createinfo.code = shader_vert_dxil;
      createinfo.code_size = shader_vert_dxil_len;
      createinfo.entrypoint = "VSMain";
      break;
    case PixelShader:
      createinfo.code = shader_frag_dxil;
      createinfo.code_size = shader_frag_dxil_len;
      createinfo.entrypoint = "PSMain";
      break;
    case PixelShader_SDF:
      createinfo.code = shader_sdf_frag_dxil;
      createinfo.code_size = shader_sdf_frag_dxil_len;
      createinfo.entrypoint = "PSMain";
      break;
    }
  } else if (format & SDL_GPU_SHADERFORMAT_MSL) {
    createinfo.format = SDL_GPU_SHADERFORMAT_MSL;
    switch (shader) {
    case VertexShader:
      createinfo.code = shader_vert_msl;
      createinfo.code_size = shader_vert_msl_len;
      createinfo.entrypoint = "main0";
      break;
    case PixelShader:
      createinfo.code = shader_frag_msl;
      createinfo.code_size = shader_frag_msl_len;
      createinfo.entrypoint = "main0";
      break;
    case PixelShader_SDF:
      createinfo.code = shader_sdf_frag_msl;
      createinfo.code_size = shader_sdf_frag_msl_len;
      createinfo.entrypoint = "main0";
      break;
    }
  } else {
    createinfo.format = SDL_GPU_SHADERFORMAT_SPIRV;
    switch (shader) {
    case VertexShader:
      createinfo.code = shader_vert_spv;
      createinfo.code_size = shader_vert_spv_len;
      createinfo.entrypoint = "main";
      break;
    case PixelShader:
      createinfo.code = shader_frag_spv;
      createinfo.code_size = shader_frag_spv_len;
      createinfo.entrypoint = "main";
      break;
    case PixelShader_SDF:
      createinfo.code = shader_sdf_frag_spv;
      createinfo.code_size = shader_sdf_frag_spv_len;
      createinfo.entrypoint = "main";
      break;
    }
  }

  if (shader == VertexShader) {
    createinfo.stage = SDL_GPU_SHADERSTAGE_VERTEX;
  } else {
    createinfo.stage = SDL_GPU_SHADERSTAGE_FRAGMENT;
  }
  return SDL_CreateGPUShader(device, &createinfo);
}

void queue_text_sequence(GeometryData *geometry_data,
                         TTF_GPUAtlasDrawSequence *sequence,
                         SDL_FColor *colour) {
  for (int i = 0; i < sequence->num_vertices; i++) {
    Vertex vert;
    const SDL_FPoint pos = sequence->xy[i];
    vert.pos = Vec3{pos.x, pos.y, 0.0f};
    vert.colour = *colour;
    vert.uv = sequence->uv[i];

    geometry_data->vertices[geometry_data->vertex_count + i] = vert;
  }

  SDL_memcpy(geometry_data->indices + geometry_data->index_count,
             sequence->indices, sequence->num_indices * sizeof(int));

  geometry_data->vertex_count += sequence->num_vertices;
  geometry_data->index_count += sequence->num_indices;
}

void queue_text(GeometryData *geometry_data, TTF_GPUAtlasDrawSequence *sequence,
                SDL_FColor *colour) {
  for (; sequence; sequence = sequence->next) {
    queue_text_sequence(geometry_data, sequence, colour);
  }
}

void set_geometry_data(Context *context, GeometryData *geometry_data) {
  Vertex *transfer_data = static_cast<Vertex *>(SDL_MapGPUTransferBuffer(
      context->device, context->transfer_buffer, false));

  SDL_memcpy(transfer_data, geometry_data->vertices,
             sizeof(Vertex) * geometry_data->vertex_count);
  SDL_memcpy(transfer_data + MAX_VERTEX_COUNT, geometry_data->indices,
             sizeof(int) * geometry_data->index_count);

  SDL_UnmapGPUTransferBuffer(context->device, context->transfer_buffer);
}

void transfer_data(Context *context, GeometryData *geometry_data) {
  SDL_GPUCopyPass *copy_pass = static_cast<SDL_GPUCopyPass *>(
      check_error_ptr(SDL_BeginGPUCopyPass(context->cmd_buf)));
  const SDL_GPUTransferBufferLocation vertex_transfer_buffer_location =
      SDL_GPUTransferBufferLocation{.transfer_buffer = context->transfer_buffer,
                                    .offset = 0};
  const SDL_GPUBufferRegion vertex_buffer_region =
      SDL_GPUBufferRegion{.buffer = context->vertex_buffer,
                          .offset = 0,
                          .size = sizeof(Vertex) * geometry_data->vertex_count};
  SDL_UploadToGPUBuffer(copy_pass, &vertex_transfer_buffer_location,
                        &vertex_buffer_region, false);
  const SDL_GPUTransferBufferLocation index_transfer_buffer_location =
      SDL_GPUTransferBufferLocation{.transfer_buffer = context->transfer_buffer,
                                    .offset =
                                        sizeof(Vertex) * MAX_VERTEX_COUNT};
  const SDL_GPUBufferRegion index_buffer_region =
      SDL_GPUBufferRegion{.buffer = context->index_buffer,
                          .offset = 0,
                          .size = sizeof(int) * geometry_data->index_count};
  SDL_UploadToGPUBuffer(copy_pass, &index_transfer_buffer_location,
                        &index_buffer_region, false);
  SDL_EndGPUCopyPass(copy_pass);
}

void draw(Context *context, SDL_Mat4X4 *matrices, int num_matrices,
          TTF_GPUAtlasDrawSequence *draw_sequence) {
  SDL_GPUTexture *swapchain_texture;
  check_error_bool(SDL_WaitAndAcquireGPUSwapchainTexture(
      context->cmd_buf, context->window, &swapchain_texture, NULL, NULL));

  if (swapchain_texture != NULL) {
    SDL_GPUColorTargetInfo colour_target_info = {0};
    colour_target_info.texture = swapchain_texture;
    colour_target_info.clear_color = SDL_FColor{0.3f, 0.4f, 0.5f, 1.0f};
    colour_target_info.load_op = SDL_GPU_LOADOP_CLEAR;
    colour_target_info.store_op = SDL_GPU_STOREOP_STORE;

    SDL_GPURenderPass *render_pass =
        SDL_BeginGPURenderPass(context->cmd_buf, &colour_target_info, 1, NULL);

    SDL_BindGPUGraphicsPipeline(render_pass, context->pipeline);
    SDL_GPUBufferBinding vertex_buffer_binding = {
        .buffer = context->vertex_buffer, .offset = 0};
    SDL_BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1);
    SDL_GPUBufferBinding index_buffer_binding = {
        .buffer = context->index_buffer, .offset = 0};
    SDL_BindGPUIndexBuffer(render_pass, &index_buffer_binding,
                           SDL_GPU_INDEXELEMENTSIZE_32BIT);
    SDL_PushGPUVertexUniformData(context->cmd_buf, 0, matrices,
                                 sizeof(SDL_Mat4X4) * num_matrices);

    int index_offset = 0, vertex_offset = 0;
    for (TTF_GPUAtlasDrawSequence *seq = draw_sequence; seq != NULL;
         seq = seq->next) {
      SDL_GPUTextureSamplerBinding sampler_binding = {
          .texture = seq->atlas_texture, .sampler = context->sampler};
      SDL_BindGPUFragmentSamplers(render_pass, 0, &sampler_binding, 1);

      SDL_DrawGPUIndexedPrimitives(render_pass, seq->num_indices, 1,
                                   index_offset, vertex_offset, 0);

      index_offset += seq->num_indices;
      vertex_offset += seq->num_vertices;
    }
    SDL_EndGPURenderPass(render_pass);
  }
}

void free_context(Context *context) {
  SDL_ReleaseGPUTransferBuffer(context->device, context->transfer_buffer);
  SDL_ReleaseGPUSampler(context->device, context->sampler);
  SDL_ReleaseGPUBuffer(context->device, context->vertex_buffer);
  SDL_ReleaseGPUBuffer(context->device, context->index_buffer);
  SDL_ReleaseGPUGraphicsPipeline(context->device, context->pipeline);
  SDL_ReleaseWindowFromGPUDevice(context->device, context->window);
  SDL_DestroyGPUDevice(context->device);
  SDL_DestroyWindow(context->window);
}

int main(int argc, char *argv[]) {
  auto basePathPtr = SDL_GetBasePath();
  if (not basePathPtr) {
    return SDL_Fail();
  }
  const std::filesystem::path basePath = basePathPtr;

  const auto fontPath = basePath / "Inter-VariableFont.ttf";

  const char *font_filename = fontPath.string().c_str();
  bool use_SDF = false;

  check_error_bool(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS));

  bool running = true;
  Context context = {0};

  context.window = static_cast<SDL_Window *>(
      check_error_ptr(SDL_CreateWindow("GPU text test", 800, 600, 0)));

  context.device = static_cast<SDL_GPUDevice *>(check_error_ptr(
      SDL_CreateGPUDevice(SUPPORTED_SHADER_FORMATS, true, NULL)));
  check_error_bool(SDL_ClaimWindowForGPUDevice(context.device, context.window));

  SDL_GPUShader *vertex_shader = static_cast<SDL_GPUShader *>(
      check_error_ptr(load_shader(context.device, VertexShader, 0, 1, 0, 0)));
  SDL_GPUShader *fragment_shader = static_cast<SDL_GPUShader *>(check_error_ptr(
      load_shader(context.device, use_SDF ? PixelShader_SDF : PixelShader, 1, 0,
                  0, 0)));

  SDL_GPUColorTargetBlendState blend_state = {
      .src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
      .dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
      .color_blend_op = SDL_GPU_BLENDOP_ADD,
      .src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA,
      .dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_DST_ALPHA,
      .alpha_blend_op = SDL_GPU_BLENDOP_ADD,
      .color_write_mask = 0xF,
      .enable_blend = true,
  };

  SDL_GPUColorTargetDescription color_target_desc = {
      .format =
          SDL_GetGPUSwapchainTextureFormat(context.device, context.window),
      .blend_state = blend_state};

  SDL_GPUColorTargetDescription color_target_descriptions[] = {
      color_target_desc};

  SDL_GPUVertexBufferDescription vertex_buffer_desc = {
      .slot = 0,
      .pitch = sizeof(Vertex),
      .input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX,
      .instance_step_rate = 0,
  };

  SDL_GPUVertexBufferDescription vertex_buffer_descriptions[] = {
      vertex_buffer_desc};

  SDL_GPUVertexAttribute vertex_attributes[] = {
      {.location = 0,
       .buffer_slot = 0,
       .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3,

       .offset = 0},
      {.location = 1,
       .buffer_slot = 0,
       .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4,
       .offset = sizeof(float) * 3},
      {.location = 2,
       .buffer_slot = 0,
       .format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2,
       .offset = sizeof(float) * 7}};

  SDL_GPUGraphicsPipelineCreateInfo pipeline_create_info = {
      .vertex_shader = vertex_shader,
      .fragment_shader = fragment_shader,
      .vertex_input_state =
          SDL_GPUVertexInputState{
              .vertex_buffer_descriptions = vertex_buffer_descriptions,
              .num_vertex_buffers = 1,

              .vertex_attributes = vertex_attributes,
              .num_vertex_attributes = 3,
          },
      .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
      .target_info =
          {
              .color_target_descriptions = color_target_descriptions,
              .num_color_targets = 1,
              .depth_stencil_format =
                  SDL_GPU_TEXTUREFORMAT_INVALID, /* Need to set this to avoid
                                                    missing initializer for
                                                    field error */
              .has_depth_stencil_target = false,
          },
  };
  context.pipeline = static_cast<SDL_GPUGraphicsPipeline *>(check_error_ptr(
      SDL_CreateGPUGraphicsPipeline(context.device, &pipeline_create_info)));

  SDL_ReleaseGPUShader(context.device, vertex_shader);
  SDL_ReleaseGPUShader(context.device, fragment_shader);

  SDL_GPUBufferCreateInfo vbf_info = {.usage = SDL_GPU_BUFFERUSAGE_VERTEX,
                                      .size =
                                          sizeof(Vertex) * MAX_VERTEX_COUNT};
  context.vertex_buffer = static_cast<SDL_GPUBuffer *>(
      check_error_ptr(SDL_CreateGPUBuffer(context.device, &vbf_info)));

  SDL_GPUBufferCreateInfo ibf_info = {.usage = SDL_GPU_BUFFERUSAGE_INDEX,
                                      .size = sizeof(int) * MAX_INDEX_COUNT};
  context.index_buffer = static_cast<SDL_GPUBuffer *>(
      check_error_ptr(SDL_CreateGPUBuffer(context.device, &ibf_info)));

  SDL_GPUTransferBufferCreateInfo tbf_info = {
      .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
      .size = (sizeof(Vertex) * MAX_VERTEX_COUNT) +
              (sizeof(int) * MAX_INDEX_COUNT)};
  context.transfer_buffer = static_cast<SDL_GPUTransferBuffer *>(
      check_error_ptr(SDL_CreateGPUTransferBuffer(context.device, &tbf_info)));

  SDL_GPUSamplerCreateInfo sampler_info = {
      .min_filter = SDL_GPU_FILTER_LINEAR,
      .mag_filter = SDL_GPU_FILTER_LINEAR,
      .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR,
      .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
      .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
      .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE};
  context.sampler = static_cast<SDL_GPUSampler *>(
      check_error_ptr(SDL_CreateGPUSampler(context.device, &sampler_info)));

  GeometryData geometry_data = {0};
  geometry_data.vertices =
      static_cast<Vertex *>(SDL_calloc(MAX_VERTEX_COUNT, sizeof(Vertex)));
  geometry_data.indices =
      static_cast<int *>(SDL_calloc(MAX_INDEX_COUNT, sizeof(int)));

  check_error_bool(TTF_Init());
  TTF_Font *font = static_cast<TTF_Font *>(check_error_ptr(
      TTF_OpenFont(font_filename, 50))); /* Preferably use a Monospaced font */
  if (!font) {
    running = false;
  }
  SDL_Log("SDF %s", use_SDF ? "enabled" : "disabled");
  TTF_SetFontSDF(font, use_SDF);
  TTF_SetFontWrapAlignment(font, TTF_HORIZONTAL_ALIGN_CENTER);
  TTF_TextEngine *engine = static_cast<TTF_TextEngine *>(
      check_error_ptr(TTF_CreateGPUTextEngine(context.device)));

  char str[] = "     \nSDL is cool";
  TTF_Text *text = static_cast<TTF_Text *>(
      check_error_ptr(TTF_CreateText(engine, font, str, 0)));

  SDL_Mat4X4 matrices[] = {
      SDL_MatrixPerspective(SDL_PI_F / 2.0f, 800.0f / 600.0f, 0.1f, 100.0f),
      SDL_MatrixIdentity()};

  float rot_angle = 0;
  SDL_FColor colour = {1.0f, 1.0f, 0.0f, 1.0f};

  while (running) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
      case SDL_EVENT_KEY_UP:
        if (event.key.key == SDLK_ESCAPE) {
          running = false;
        }
        break;
      case SDL_EVENT_QUIT:
        running = false;
        break;
      }
    }

    for (int i = 0; i < 5; i++) {
      str[i] = 65 + SDL_rand(26);
    }
    TTF_SetTextString(text, str, 0);

    int tw, th;
    check_error_bool(TTF_GetTextSize(text, &tw, &th));

    rot_angle = SDL_fmodf(rot_angle + 0.01, 2 * SDL_PI_F);

    // Create a model matrix to make the text rotate
    SDL_Mat4X4 model;
    model = SDL_MatrixIdentity();
    model = SDL_MatrixMultiply(
        model, SDL_MatrixTranslation(SDL_Vec3{0.0f, 0.0f, -80.0f}));
    model = SDL_MatrixMultiply(model,
                               SDL_MatrixScaling(SDL_Vec3{0.3f, 0.3f, 0.3f}));
    model = SDL_MatrixMultiply(model, SDL_MatrixRotationY(rot_angle));
    model = SDL_MatrixMultiply(
        model, SDL_MatrixTranslation(SDL_Vec3{-tw / 2.0f, th / 2.0f, 0.0f}));
    matrices[1] = model;

    // Get the text data and queue the text in a buffer for drawing later
    TTF_GPUAtlasDrawSequence *sequence = TTF_GetGPUTextDrawData(text);
    queue_text(&geometry_data, sequence, &colour);

    set_geometry_data(&context, &geometry_data);

    context.cmd_buf = static_cast<SDL_GPUCommandBuffer *>(
        check_error_ptr(SDL_AcquireGPUCommandBuffer(context.device)));
    transfer_data(&context, &geometry_data);
    draw(&context, matrices, 2, sequence);
    SDL_SubmitGPUCommandBuffer(context.cmd_buf);

    geometry_data.vertex_count = 0;
    geometry_data.index_count = 0;
  }

  SDL_free(geometry_data.vertices);
  SDL_free(geometry_data.indices);
  TTF_DestroyText(text);
  TTF_DestroyGPUTextEngine(engine);
  TTF_CloseFont(font);
  TTF_Quit();
  free_context(&context);
  SDL_Quit();

  return 0;
}