
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

struct SpherePoint {
  float3 position; // 球壳上的点位置
  float3 velocity; // 移动速度
  float pointId;   // 点ID
};

struct SphereVertex {
  float3 position;
  float3 color;
  int seed;
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
    device SpherePoint *spherePoints [[buffer(0)]],
    device SpherePoint *outputSpherePoints [[buffer(1)]],
    constant Params &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {

  SpherePoint point = spherePoints[id];
  device SpherePoint &outputPoint = outputSpherePoints[id];

  float seed = point.pointId + float(id) * 0.1;

  // 使用稳定的时间增量
  float dt = max(abs(params.time), 0.016); // 至少16ms (60fps)
  dt = min(dt, 0.1);                       // 限制在100ms以防止大跳跃

  // 复制基本属性
  outputPoint.pointId = point.pointId;

  // 缓慢移动，保持在球壳上
  float3 newPos = point.position + point.velocity * dt;

  // 将点重新投影到球壳上
  float currentRadius = length(newPos);
  if (currentRadius > 0.0) {
    newPos = normalize(newPos) * 5.0; // 保持在半径5m的球壳上
  }

  outputPoint.position = newPos;

  // 更新速度，添加一些随机扰动
  float3 newVelocity = point.velocity;
  newVelocity += float3(
      (random1D(seed + params.time) - 0.5) * 0.001,
      (random1D(seed + params.time + 1.0) - 0.5) * 0.001,
      (random1D(seed + params.time + 2.0) - 0.5) * 0.001);

  // 限制速度大小
  float velocityMag = length(newVelocity);
  if (velocityMag > 0.05) {
    newVelocity = normalize(newVelocity) * 0.05;
  }

  outputPoint.velocity = newVelocity;
}

vertex DomeInOut domeVertexShader(
    DomeVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device SpherePoint *spherePoints [[buffer(BufferIndexBase)]]) {
  DomeInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  // float3 cameraAt = uniforms.cameraPos;

  // 使用球壳顶点位置
  float4 position = float4(in.position, 1.0);

  // float sphereDistance = distance(cameraAt, position.xyz);
  // float distanceDim = 1.0 - clamp(sphereDistance / 40.0, 0.0, 0.8);

  position = applyGestureViewerOnScene(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation,
      uniforms.cameraPos);

  position.w = 1;

  out.position = uniformsPerView.modelViewProjectionMatrix * position;

  // 传递顶点位置到fragment shader用于计算连线效果
  out.color = float4(in.position, tintUniform.tintOpacity);

  return out;
}

fragment float4 domeFragmentShader(
    DomeInOut in [[stage_in]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device SpherePoint *spherePoints [[buffer(BufferIndexBase)]]) {

  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  // 当前片段在球壳上的位置
  float3 fragPos = in.color.xyz;

  // 默认透明
  float4 finalColor = float4(0.0, 0.0, 0.0, 0.0);

  // 检查是否在点的位置显示小圆点
  for (uint i = 0; i < 40; i++) { // pointCount = 40
    float3 pointPos = spherePoints[i].position;
    float distToPoint = distance(fragPos, pointPos);

    // 在点的位置显示小圆点
    if (distToPoint < 0.1) {
      finalColor = float4(1.0, 1.0, 1.0, 0.8); // 白色圆点
      break;
    }
  }

  // 连线效果计算
  if (finalColor.a < 0.1) { // 如果不是圆点位置
    for (uint i = 0; i < 40; i++) {
      for (uint j = i + 1; j < 40; j++) {
        float3 point1 = spherePoints[i].position;
        float3 point2 = spherePoints[j].position;

        float pointDistance = distance(point1, point2);

        // 只处理距离小于1.5m的点对
        if (pointDistance < 1.5) {
          // 计算当前片段到线段的距离
          float3 lineDir = point2 - point1;
          float lineLength = length(lineDir);

          if (lineLength > 0.0) {
            lineDir = lineDir / lineLength;
            float3 toFrag = fragPos - point1;
            float projLength = dot(toFrag, lineDir);

            // 确保投影在线段范围内
            projLength = clamp(projLength, 0.0, lineLength);
            float3 closestPoint = point1 + lineDir * projLength;
            float distToLine = distance(fragPos, closestPoint);

            // 如果片段在连线附近（线宽0.08m）
            if (distToLine < 0.08) {
              float brightness = 1.0 - (pointDistance / 1.5);
              brightness = max(brightness, 0.2);

              // 根据距离线段的远近调整透明度
              float lineAlpha = 1.0 - (distToLine / 0.08);
              float alpha = brightness * lineAlpha * 0.4;

              finalColor = max(finalColor, float4(1.0, 1.0, 1.0, alpha));
            }
          }
        }
      }
    }
  }

  return finalColor;
}
