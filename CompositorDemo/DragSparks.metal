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
  float3 position [[attribute(0)]];
  float3 color [[attribute(1)]];
  float3 direction [[attribute(2)]];
  float brushWidth [[attribute(3)]];
  float brushValue [[attribute(4)]];
  float birthTime [[attribute(5)]];
} PolylineVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
  float lifetime;
} TrianglesVertexInInOut;

typedef struct {
  float time;
} Params;

vertex TrianglesVertexInInOut dragSparksVertexShader(
    PolylineVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]]) {
  TrianglesVertexInInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  simd_float3 cameraDirection = uniforms.cameraDirection;
  float3 brush = cross(in.direction, cameraDirection);
  brush = brush * 0.5 * in.brushWidth;

  float4 position = float4(in.position + brush, 1.0);

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Don't override the alpha here, we'll handle it in the fragment shader
  out.lifetime = (params.time - in.birthTime) * 6 - in.brushValue;

  return out;
}

fragment float4 dragSparksFragmentShader(TrianglesVertexInInOut in
                                         [[stage_in]]) {
  // Calculate fade value based on lifetime
  float v = sin(in.lifetime * 3.14);

  // If lifetime is negative, or fade value is too low, discard fragment
  if (in.lifetime < 0.0 || v <= 0.01) {
    discard_fragment();
  }

  // Start with base color
  float3 sparkColor = in.color.rgb * v;

  // Create final color with appropriate alpha
  // With additive blending, lower alpha makes particles less intense
  return float4(sparkColor, v);
}
