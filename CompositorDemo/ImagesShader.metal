/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
Contains the vertex and fragment shaders for the path and tint renderers.
*/

#include <metal_stdlib>
#include <simd/simd.h>

// Include header shared between this Metal shader code and Swift/C code.
// (This needs to be imported first)
#import "ShaderTypes.h"

// executing Metal API commands.
#import "PathProperties.h"

using namespace metal;

typedef struct {
  float3 position [[attribute(VertexAttributePosition)]];
  float3 color [[attribute(VertexAttributeColor)]];
  int seed [[attribute(VertexAttributeSeed)]];
  float height [[attribute(3)]];
  float2 uv [[attribute(4)]];
} BlockVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
  float3 originalPosition;
  float height;
  float2 uv;
} BlockInOut;

typedef struct {
  float time;
  float3 viewerPosition;
  float viewerScale;
  float viewerRotation;
} Params;

struct CellBase {
  float3 position;
  float3 color;
  float lampIdf;
  float3 velocity;
};

kernel void imagesComputeShader(
    device CellBase *blocks [[buffer(0)]],
    device CellBase *outputLamps [[buffer(1)]],
    constant Params &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  // CellBase block = blocks[id];
  // device CellBase &outputBlock = outputLamps[id];
  // float seed = fract(block.lampIdf / 10.) * 10.;
  // float speed = random1D(seed) + 0.1;
  // float dt = params.time * speed * 0.1;
  // outputBlock.position =
  //     block.position + float3(0.0, dt, 0.0) + block.velocity * dt;
  // outputBlock.color = block.color;
  // outputBlock.lampIdf = block.lampIdf;
}

vertex BlockInOut imagesVertexShader(
    BlockVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device CellBase *blocksData [[buffer(BufferIndexBase)]]) {
  BlockInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  float3 cameraAt = uniforms.cameraPos;

  float3 basePosition = blocksData[in.seed].position;
  float4 position = float4(in.position + basePosition, 1.0);

  out.originalPosition = basePosition.xyz;
  out.uv = in.uv;

  position.w = 1.0; // reset w to 1.0 for the projection matrix

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Premultiply color channel by alpha channel.
  out.color.rgb = out.color.rgb * out.color.a;
  out.height = in.height;

  return out;
}

fragment float4 imagesFragmentShader(BlockInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return float4(in.color.rgb, in.color.a);
}
