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
  float3 viewerPosition;
  float time;
  float viewerScale;
  float viewerRotation;
} Params;

struct CellBase {
  float3 position;        // 三角形在正八面体内的相对位置
  float3 color;           // 三角形颜色（与所属正八面体一致）
  float octahedronId;     // 所属正八面体的ID
  float3 octahedronCenter; // 正八面体中心位置
  float rotationAngle;    // 正八面体当前旋转角度
  float triangleSize;     // 三角形大小
};

static float random1D(float seed) { return fract(sin(seed) * 43758.5453123); }


static float4 applyGestureViewer(
    float4 p0,
    float3 viewerPosition,
    float viewerScale,
    float viewerRotation,
    float3 cameraAt) {

  float4 position = p0;

  // position -= cameraAt;

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

  // position += cameraAt;

  return position;
}
// 移除compute shader以优化性能，所有计算移至vertex shader

vertex LampInOut octahedronVertexShader(
    LampVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device CellBase *triangleData [[buffer(BufferIndexBase)]]) {
  LampInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  // float3 cameraAt = uniforms.cameraPos;

  // 获取三角形数据
  int triangleIndex = in.seed;
  CellBase triangleInfo = triangleData[triangleIndex];
  
  // 优化的旋转计算
  float angle = params.time * 0.5; // 缓慢旋转
  // uint octahedronId = uint(triangleIndex) / 1000; // 每个正八面体1000个三角形
  
  // 直接计算旋转后的位置，避免矩阵乘法
  float cosAngle = cos(angle);
  float sinAngle = sin(angle);
  float3 pos = in.position;
  float3 rotatedPos = float3(
    pos.x * cosAngle - pos.z * sinAngle,
    pos.y,
    pos.x * sinAngle + pos.z * cosAngle
  );
  
  // 计算最终世界位置
  float4 position = float4(triangleInfo.octahedronCenter + rotatedPos, 1.0);

  position = applyGestureViewer(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation,
      uniforms.cameraPos);

  position.w = 1;

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  // 简化颜色处理，确保可见性
  out.color = float4(triangleInfo.color, 1.0); // 设置完全不透明
  out.color.rgb = triangleInfo.color; // 直接使用原始颜色，不做复杂计算

  return out;
}

fragment float4 octahedronFragmentShader(LampInOut in [[stage_in]]) {
  // 简化实现，直接返回颜色
  return in.color;
}
