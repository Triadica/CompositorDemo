//
//  PolylinesShader.metal
//  CompositorDemo
//
//  Created by chen on 2025/3/30.
//  Copyright © 2025 Apple. All rights reserved.
//

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

} LampInOut;

typedef struct {
  float time;
} Params;

struct LampBase {
  float3 position;
  float3 color;
  float lampIdf;
  float3 velocity;
};

vertex LampInOut polylinesVertexShader(
    VertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]]) {
  LampInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];

  float4 position = float4(in.position, 1.0);

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Premultiply color channel by alpha channel.
  out.color.rgb = out.color.rgb * out.color.a;

  return out;
}

fragment float4 polylinesFragmentShader(LampInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
