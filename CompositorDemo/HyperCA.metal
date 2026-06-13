/*
 See the LICENSE.txt file for this sample’s licensing information.

 Abstract:
 Metal shader for 有状态超图元胞自动机 (HyperCA).
 Supports directed edges with gradient colors and direction width tapering, as well as 3D node meshes.
 */

#include <metal_stdlib>
#include <simd/simd.h>

// Include shared header specifying Vertex attributes, indices and uniforms.
#import "ShaderTypes.h"
#import "PathProperties.h"

using namespace metal;

typedef struct {
  float3 position [[attribute(PolylineVertexAttributePosition)]];
  float3 color [[attribute(PolylineVertexAttributeColor)]];
  float3 direction [[attribute(PolylineVertexAttributeDirection)]];
  int seed [[attribute(PolylineVertexAttributeSeed)]]; // Seed is reused to store brush width
} HyperCAVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
} HyperCAVertexInOut;

typedef struct {
  float3 viewerPosition;
  float time;
  float viewerScale;
  float viewerRotation;
  float4 _padding;
} Params;

static float4 applyGestureViewer(
    float4 p0,
    float3 viewerPosition,
    float viewerScale,
    float viewerRotation,
    float3 cameraAt) {

  float4 position = p0;

  // rotate xz by viewerRotation
  float cosTheta = cos(viewerRotation);
  float sinTheta = sin(viewerRotation);
  float x = position.x * cosTheta - position.z * sinTheta;
  float z = position.x * sinTheta + position.z * cosTheta;
  position.x = x;
  position.z = z;

  // scale
  position *= viewerScale;

  // translate
  position = position - float4(viewerPosition, 0.0);

  return position;
}

vertex HyperCAVertexInOut hyperCAVertexShader(
    HyperCAVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]]) {
  HyperCAVertexInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  float3 cameraDirection = uniforms.cameraDirection;

  float3 finalPosition = in.position;

  // If in.direction is not zero, perform billboarding expansion for directed edges
  if (length_squared(in.direction) > 0.0001) {
    // Generate a vector perpendicular to both the line direction and camera direction
    float3 brush = cross(in.direction, cameraDirection);
    if (length_squared(brush) > 0.0001) {
      brush = normalize(brush) * 0.0001 * float(in.seed);
      finalPosition += brush;
    }
  }

  float4 position = float4(finalPosition, 1.0);
  position = applyGestureViewer(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation,
      uniforms.cameraPos);
  
  position.w = 1.0; // required for perspective projection division

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);

  // Premultiply color by alpha for transparent compositor rendering
  out.color.rgb = out.color.rgb * out.color.a;

  return out;
}

fragment float4 hyperCAFragmentShader(HyperCAVertexInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }
  return in.color;
}
