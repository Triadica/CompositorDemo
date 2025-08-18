
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
} DomeVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;

} DomeInOut;

typedef struct {
  float3 viewerPosition;
  float time;
  float viewerScale;
  float viewerRotation;
} Params;

struct DomeSegmentBase {
  float3 position;
  float3 color;
  float segmentId;
  float3 velocity;
  float activateTime; // 激活时间
  bool isActive;      // 是否激活
};

static float random1D(float seed) { return fract(sin(seed) * 43758.5453123); }

static float4 applyGestureViewerOnScene(
    float4 p0,
    float3 viewerPosition,
    float viewerScale,
    float viewerRotation,
    float3 cameraAt) {

  float4 position = p0;

  // position -= cameraAt4;

  // translate
  position = position - float4(viewerPosition, 0.0);

  // rotate xz by viewerRotation
  float cosTheta = cos(viewerRotation);
  float sinTheta = sin(viewerRotation);
  float x = position.x * cosTheta - position.z * sinTheta;
  float z = position.x * sinTheta + position.z * cosTheta;
  position.x = x;
  position.z = z;

  // scale
  position *= viewerScale;

  // position += cameraAt4;

  return position;
}

kernel void domeComputeShader(
    device DomeSegmentBase *domeSegments [[buffer(0)]],
    device DomeSegmentBase *outputDomeSegments [[buffer(1)]],
    constant Params &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {

  DomeSegmentBase domeSegment = domeSegments[id];
  device DomeSegmentBase &outputDomeSegment = outputDomeSegments[id];

  float seed = domeSegment.segmentId + float(id) * 0.1;

  // Use a more stable time delta
  float dt = max(abs(params.time), 0.016); // At least 16ms (60fps)
  dt = min(dt, 0.1);                       // Cap at 100ms to prevent huge jumps

  // Copy basic properties
  outputDomeSegment.color = domeSegment.color;
  outputDomeSegment.segmentId = domeSegment.segmentId;

  // 缓慢旋转和浮动效果
  float rotationSpeed = 0.1 + random1D(seed) * 0.05;
  float floatSpeed = 0.02 + random1D(seed * 2.0) * 0.01;
  
  // 圆形路径旋转
  float angle = params.time * rotationSpeed + seed * 6.28;
  float radius = length(domeSegment.position.xz);
  
  outputDomeSegment.position.x = cos(angle) * radius;
  outputDomeSegment.position.z = sin(angle) * radius;
  outputDomeSegment.position.y = domeSegment.position.y + sin(params.time * floatSpeed + seed) * 0.5;
  
  outputDomeSegment.velocity = domeSegment.velocity;
  outputDomeSegment.isActive = domeSegment.isActive;
  outputDomeSegment.activateTime = domeSegment.activateTime;
}

vertex DomeInOut domeVertexShader(
    DomeVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device DomeSegmentBase *domeSegmentData [[buffer(BufferIndexBase)]]) {
  DomeInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  float3 cameraAt = uniforms.cameraPos;

  // Use position directly from compute buffer
  float4 position = float4(in.position + domeSegmentData[in.seed].position, 1.0);

  float domeSegmentDistance = distance(cameraAt, position.xyz);
  float distanceDim = 1.0 - clamp(domeSegmentDistance / 40.0, 0.0, 0.8);

  position = applyGestureViewerOnScene(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation,
      uniforms.cameraPos);

  position.w = 1;

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(in.color, tintUniform.tintOpacity);
  // Premultiply color channel by alpha channel.
  out.color.rgb = out.color.rgb * out.color.a * distanceDim;

  return out;
}

fragment float4 domeFragmentShader(DomeInOut in [[stage_in]]) {
  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  return in.color;
}
