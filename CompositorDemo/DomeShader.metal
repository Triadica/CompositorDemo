
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
  float is_facing_camera;
} DomeInOut;

typedef struct {
  float3 viewerPosition;
  float time;
  float viewerScale;
  float viewerRotation;
} Params;

struct SpherePoint {
  float3 position;     // 球壳上的点位置
  float3 rotationAxis; // 旋转轴（过球心的直线方向）
  float angularSpeed;  // 角速度（弧度/秒）
  float pointId;       // 点ID
};

struct SphereVertex {
  float3 position;
  float3 color;
  int seed;
};

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
  // 防止越界线程访问
  if (id >= 80) {
    return;
  }

  SpherePoint point = spherePoints[id];
  SpherePoint outputPoint;

  float dt = params.time;
  if (dt > 0.1) {
    dt = 0.016;
  }

  // Rodrigues' rotation formula
  float3 v = point.position;
  float3 k = point.rotationAxis;
  float theta = point.angularSpeed * dt;
  float cos_theta = cos(theta);
  float sin_theta = sin(theta);
  float3 rotated_v =
      v * cos_theta + cross(k, v) * sin_theta + k * dot(k, v) * (1 - cos_theta);

  outputPoint.position = rotated_v;
  outputPoint.rotationAxis = point.rotationAxis;
  outputPoint.angularSpeed = point.angularSpeed;
  outputPoint.pointId = point.pointId;

  outputSpherePoints[id] = outputPoint;
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

  // Object-space position and normal
  float4 position_obj = float4(in.position, 1.0);
  float3 normal_obj = normalize(-in.position);

  // Apply model transform to position to get world position
  float4 position_world = applyGestureViewerOnScene(
      position_obj,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation,
      uniforms.cameraPos);

  // To get the world-space normal, we need to apply the rotation part of the transform.
  // The applyGestureViewerOnScene function has rotation logic.
  float cosTheta = cos(params.viewerRotation);
  float sinTheta = sin(params.viewerRotation);
  float normal_x = normal_obj.x * cosTheta - normal_obj.z * sinTheta;
  float normal_z = normal_obj.x * sinTheta + normal_obj.z * cosTheta;
  float3 normal_world = normalize(float3(normal_x, normal_obj.y, normal_z));

  // Calculate visibility
  float3 view_dir = normalize(uniforms.cameraPos - position_world.xyz);
  out.is_facing_camera = dot(normal_world, view_dir);

  // Final position
  position_world.w = 1;
  out.position = uniformsPerView.modelViewProjectionMatrix * position_world;

  // Pass object-space position to fragment shader via 'color'
  out.color = float4(in.position, tintUniform.tintOpacity);

  return out;
}

fragment float4 domeFragmentShader(
    DomeInOut in [[stage_in]],
    constant Params &params [[buffer(BufferIndexParams)]],
    const device SpherePoint *spherePoints [[buffer(BufferIndexBase)]]) {

  if (in.is_facing_camera <= 0.0) {
    return float4(0.2, 0.2, 0.2, 1.0);
  }

  if (in.color.a <= 0.0) {
    discard_fragment();
  }

  // 当前片段在球壳上的位置
  float3 fragPos = in.color.xyz;

  // 默认透明
  float4 finalColor = float4(0.0, 0.0, 0.0, 0.0);

  // 渲染球壳上的点
  for (uint i = 0; i < 80; i++) {
    float3 pointPos = spherePoints[i].position;
    float dist = distance(fragPos, pointPos);
    if (dist < 0.05) {
      finalColor = float4(1.0, 1.0, 1.0, 1.0);
      break;
    }
  }

  // 连线效果计算 (Optimized)
  if (finalColor.a < 0.1) { // 如果不是圆点位置

    for (uint i = 0; i < 80; i++) {
      float3 point1 = spherePoints[i].position;

      // 优化：如果片段离 point1 太远，则不可能在任何从 point1 出发的线段上。
      // 最大线长为 1.5，线宽为 0.02。因此，片段到 point1 的最大距离为 1.5 +
      // 0.02 = 1.52。
      if (distance(fragPos, point1) > 1.52) {
        continue;
      }

      for (uint j = i + 1; j < 80; j++) {
        float3 point2 = spherePoints[j].position;

        float pointDistance = distance(point1, point2);

        // 只处理距离小于 1.5m 且大于 0 的点对
        if (pointDistance > 1e-5 && pointDistance < 1.5) {
          // 计算当前片段到线段的距离
          float3 lineDir = (point2 - point1) / pointDistance;
          float3 toFrag = fragPos - point1;
          float projLength = dot(toFrag, lineDir);

          // 确保投影在线段范围内
          projLength = clamp(projLength, 0.0, pointDistance);
          float3 closestPoint = point1 + lineDir * projLength;
          float distToLine = distance(fragPos, closestPoint);

          // 如果片段在连线附近（线宽 0.02m）
          if (distToLine < 0.02) {
            float brightness = 1.0 - (pointDistance / 1.5);
            brightness = max(brightness, 0.2);

            // 根据距离线段的远近调整透明度
            float lineAlpha = 1.0 - (distToLine / 0.02);
            float alpha = brightness * lineAlpha * 0.4;

            finalColor = max(finalColor, float4(1.0, 1.0, 1.0, alpha));
          }
        }
      }
    }
  }

  return finalColor;
}
