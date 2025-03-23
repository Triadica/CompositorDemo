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
} VertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;

} TintInOut;

typedef struct {
  float time;
} Params;

struct LampBase {
  float3 position;
  float3 color;
  float seed;
  float3 velocity;
};

static float random1D(float seed) { return fract(sin(seed) * 43758.5453123); }

kernel void lampsComputeShader(
    device LampBase *lamps [[buffer(0)]],
    device LampBase *outputLamps [[buffer(1)]],
    constant Params &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  LampBase lamp = lamps[id];
  device LampBase &outputLamp = outputLamps[id];
  float seed = fract(lamp.seed / 10.) * 10.;
  float speed = random1D(seed) + 0.1;
  float dt = params.time * speed * 0.1;
  outputLamp.position =
      lamp.position + float3(0.0, dt, 0.0) + lamp.velocity * dt;
  outputLamp.color = lamp.color;
  outputLamp.seed = lamp.seed;
}

vertex TintInOut lampsVertexShader(
    VertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniformsArray [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device LampBase *lampData [[buffer(BufferIndexBase)]]) {
  TintInOut out;

  UniformsPerView uniformsPerView = uniformsArray.perView[amp_id];
  float4 position = float4(in.position + lampData[in.seed].position, 1.0);
  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Premultiply color channel by alpha channel.
  out.color.rgb = out.color.rgb * out.color.a;

  return out;
}

fragment float4 lampsFragmentShader(TintInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
