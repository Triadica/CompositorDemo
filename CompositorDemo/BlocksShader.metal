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
} LampVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;

} LampInOut;

typedef struct {
  float time;
} Params;

struct CellBase {
  float3 position;
  float3 color;
  float lampIdf;
  float3 velocity;
};

static float random1D(float seed) { return fract(sin(seed) * 43758.5453123); }

kernel void blocksComputeShader(
    device CellBase *lamps [[buffer(0)]],
    device CellBase *outputLamps [[buffer(1)]],
    constant Params &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  CellBase lamp = lamps[id];
  device CellBase &outputLamp = outputLamps[id];
  float seed = fract(lamp.lampIdf / 10.) * 10.;
  float speed = random1D(seed) + 0.1;
  float dt = params.time * speed * 0.1;
  outputLamp.position =
      lamp.position + float3(0.0, dt, 0.0) + lamp.velocity * dt;
  outputLamp.color = lamp.color;
  outputLamp.lampIdf = lamp.lampIdf;
}

vertex LampInOut blocksVertexShader(
    LampVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device CellBase *lampData [[buffer(BufferIndexBase)]]) {
  LampInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  float3 cameraAt = uniforms.cameraPos;

  float4 position = float4(in.position + lampData[in.seed].position, 1.0);

  float lampDistance = distance(cameraAt, position.xyz);
  float distanceDim = 1.0 - clamp(lampDistance / 30.0, 0.0, 1.0);
  float randSeed = random1D(lampData[in.seed].lampIdf);
  float breathDim = 1.0 - sin(params.time * 1. * randSeed) * 0.8;

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Premultiply color channel by alpha channel.
  out.color.rgb = out.color.rgb * out.color.a * distanceDim * breathDim;

  return out;
}

fragment float4 blocksFragmentShader(LampInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
