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
} FlowerVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
} FlowerInOut;

typedef struct {
  float3 viewerPosition;
  float time;
  float viewerScale;
  float viewerRotation;
} Params;

struct CellBase {
  float3 position;        // 花瓣在花朵内的相对位置
  float3 color;           // 花瓣颜色（与所属花朵一致）
  float flowerId;         // 所属花朵的ID
  float3 flowerCenter;    // 花朵中心位置
  float rotationAngle;    // 花朵当前旋转角度
  float petalSize;        // 花瓣大小
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

vertex FlowerInOut flowersVertexShader(
                                        FlowerVertexIn in [[stage_in]],
                                        ushort amp_id [[amplification_id]],
                                        constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
                                        constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
                                        constant Params &params [[buffer(BufferIndexParams)]],
                                        const device CellBase *petalData [[buffer(BufferIndexBase)]]) {
  FlowerInOut out;
  
  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
    // float3 cameraAt = uniforms.cameraPos;
  
    // 获取花瓣数据
  int petalIndex = in.seed;
  CellBase petalInfo = petalData[petalIndex];
  
    // 优化的旋转计算
  float angle = params.time * 0.5; // 缓慢旋转
                                   // uint flowerId = uint(petalIndex) / 1000; // 每朵花1000个花瓣
  
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
  float4 position = float4(petalInfo.flowerCenter + rotatedPos, 1.0);
  
  position = applyGestureViewer(
                                position,
                                params.viewerPosition,
                                params.viewerScale,
                                params.viewerRotation,
                                uniforms.cameraPos);
  
  position.w = 1;
  
  out.position = uniformsPerView.modelViewProjectionMatrix * position;
    // 简化颜色处理，确保可见性
  out.color = float4(petalInfo.color, 1.0); // 设置完全不透明
  out.color.rgb = petalInfo.color; // 直接使用原始颜色，不做复杂计算
  
  return out;
}

fragment float4 flowersFragmentShader(FlowerInOut in [[stage_in]]) {
    // 简化实现，直接返回颜色
  return in.color;
}
