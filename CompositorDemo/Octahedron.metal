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
kernel void octahedronComputeShader(
    device CellBase *triangles [[buffer(0)]],
    device CellBase *outputTriangles [[buffer(1)]],
    constant Params &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  CellBase triangle = triangles[id];
  device CellBase &outputTriangle = outputTriangles[id];
  
  // 计算正八面体的旋转角度（围绕y轴缓慢旋转）
  float rotationSpeed = 0.5; // 旋转速度
  float currentRotation = triangle.rotationAngle + params.time * rotationSpeed;
  
  // 应用y轴旋转变换到三角形的相对位置
  float cosTheta = cos(currentRotation);
  float sinTheta = sin(currentRotation);
  
  float3 rotatedPosition = triangle.position;
  float x = rotatedPosition.x * cosTheta - rotatedPosition.z * sinTheta;
  float z = rotatedPosition.x * sinTheta + rotatedPosition.z * cosTheta;
  rotatedPosition.x = x;
  rotatedPosition.z = z;
  
  // 最终位置 = 正八面体中心 + 旋转后的相对位置
  outputTriangle.position = triangle.octahedronCenter + rotatedPosition;
  outputTriangle.color = triangle.color;
  outputTriangle.octahedronId = triangle.octahedronId;
  outputTriangle.octahedronCenter = triangle.octahedronCenter;
  outputTriangle.rotationAngle = currentRotation;
  outputTriangle.triangleSize = triangle.triangleSize;
}

vertex LampInOut octahedronVertexShader(
    LampVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device CellBase *triangleData [[buffer(BufferIndexBase)]]) {
  LampInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  float3 cameraAt = uniforms.cameraPos;

  // 获取三角形数据
  int triangleIndex = in.seed;
  CellBase triangleInfo = triangleData[triangleIndex];
  
  // 计算旋转角度（基于时间）
  float time = params.time;
  float rotationSpeed = 0.5; // 缓慢旋转
  float angle = time * rotationSpeed;
  
  // 根据种子确定所属的正八面体
   uint octahedronId = uint(triangleIndex) / 2500; // 每个正八面体2500个三角形
  
  // 创建Y轴旋转矩阵
  float cosAngle = cos(angle);
  float sinAngle = sin(angle);
  float3x3 rotationMatrix = float3x3(
    float3(cosAngle, 0.0, sinAngle),
    float3(0.0, 1.0, 0.0),
    float3(-sinAngle, 0.0, cosAngle)
  );
  
  // 应用旋转到顶点位置（小三角形相对于正八面体中心的位置）
  float3 rotatedPosition = rotationMatrix * in.position;
  
  // 计算最终世界位置：正八面体中心 + 旋转后的相对位置
  float4 position = float4(triangleInfo.octahedronCenter + rotatedPosition, 1.0);

  // 简化实现，确保基本可见性
  float distanceDim = 1.0; // 移除距离衰减
  float breathDim = 1.0;   // 移除闪烁效果

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
