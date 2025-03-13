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
} VertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
} TintInOut;

vertex TintInOut tintVertexShader(VertexIn in [[stage_in]],
                                  ushort amp_id [[amplification_id]],
                                  constant Uniforms &uniformsArray
                                  [[buffer(BufferIndexUniforms)]],
                                  constant TintUniforms &tintUniform
                                  [[buffer(BufferIndexTintUniforms)]]) {
  TintInOut out;

  UniformsPerView uniformsPerView = uniformsArray.perView[amp_id];

  float4 position = float4(in.position, 1.0);
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
