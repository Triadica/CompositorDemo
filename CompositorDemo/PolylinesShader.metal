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
  float3 position [[attribute(PolylineVertexAttributePosition)]];
  float3 color [[attribute(PolylineVertexAttributeColor)]];
  float3 direction [[attribute(PolylineVertexAttributeDirection)]];
  int seed [[attribute(PolylineVertexAttributeSeed)]];
} PolylineVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;

} PolylineVertexInInOut;

typedef struct {
  float time;
} Params;

struct LampBase {
  float3 position;
  float3 color;
  float lampIdf;
  float3 velocity;
};

vertex PolylineVertexInInOut polylinesVertexShader(
    PolylineVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]]) {
  PolylineVertexInInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  simd_float3 cameraDirection = uniforms.cameraDirection;
  float3 brush = cross(in.direction, cameraDirection);
  brush = brush * 0.0001 * in.seed;

  float4 position = float4(in.position + brush, 1.0);

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Premultiply color channel by alpha channel.
  out.color.rgb = out.color.rgb * out.color.a;

  return out;
}

fragment float4 polylinesFragmentShader(PolylineVertexInInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
