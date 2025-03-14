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
  float seed [[attribute(VertexAttributeSeed)]];
} VertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;

} TintInOut;

typedef struct {
  float time;
} Params;

static float random1D(float seed) { return fract(sin(seed) * 43758.5453123); }

vertex TintInOut tintVertexShader(
    VertexIn in [[stage_in]], ushort amp_id [[amplification_id]],
    constant Uniforms &uniformsArray [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]]) {
  TintInOut out;

  UniformsPerView uniformsPerView = uniformsArray.perView[amp_id];
  float seed = fract(in.seed / 10.) * 10.;
  float speed = random1D(seed) + 0.4;
  float yFloating = sin((speed * params.time) * 0.02) * 10.0;
  float4 position = float4(in.position, 1.0) + float4(0.0, yFloating, 0.0, 0.0);
  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Premultiply color channel by alpha channel.
  out.color.rgb = out.color.rgb * out.color.a;

  return out;
}

fragment float4 tintFragmentShader(TintInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
