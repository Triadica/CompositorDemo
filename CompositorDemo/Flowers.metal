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

// --- 百日菊 (Zinnia) 花瓣参数 ---

// 几何精度
constant int kSegmentsPerPetal = 24;    // 轮廓线段数 (更高更平滑)
constant int kLinesPerPetal    = 18;    // 内部填充线数
constant int kVerticesPerPetal = (kSegmentsPerPetal + kLinesPerPetal) * 2;

// 基本外形比例 (以长度 L 为基准)
constant float kL     = 0.10;           // 基准长度
constant float kWb    = 0.18 * kL;      // 基部宽度
constant float kWm    = 0.42 * kL;      // 最大宽度
constant float kWt    = 0.28 * kL;      // 尖端宽度 (略宽，呈圆钝感)
constant float kSPeak = 0.48;           // 最大宽度所在位置 (0..1, 从基部到尖端)
constant float kKb    = 0.35;           // 基部外扩控制 (Bézier 控制点)

// 3D 立体形态参数
constant float kBaseConcave = 0.25 * kL;   // 基部内凹深度 (纵向)
constant float kTipLift     = 0.20 * kL;   // 尖端上翘高度 (纵向)
constant float kEdgeDrop    = 0.15 * kL;   // 边缘下垂深度 (横向, 形成 V 形截面)
constant float kTwistMaxDeg = 28.0;        // 沿长度方向的最大扭转角度


// 三次贝塞尔
inline float2 cubicBezier2(float2 p0, float2 p1, float2 p2, float2 p3, float t) {
  float u = 1.0 - t;
  return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3;
}

// 左/右边界（局部坐标：y 沿长度，x 为横向宽度）
inline float2 petalBoundaryL(float L, float Wb, float Wm, float Wt, float s) {
  float2 P0 = float2(-0.5*Wb, 0.0);
  float2 P1 = float2(-0.5*Wb - kKb*Wb, 0.22*L);
  float2 P2 = float2(-0.5*Wm, kSPeak*L);
  float2 P3 = float2(-0.5*Wt, L);
  return cubicBezier2(P0, P1, P2, P3, clamp(s, 0.0, 1.0));
}
inline float2 petalBoundaryR(float L, float Wb, float Wm, float Wt, float s) {
  float2 p = petalBoundaryL(L, Wb, Wm, Wt, s);
  p.x = -p.x;
  return p;
}

// 中线（略微前后弯曲可选，这里采用直线，可按需调整）
inline float2 petalCenterline(float L, float s) {
  return float2(0.0, s * L);
}

// 花瓣中线的纵向高度 (实现基部内凹、尖端上翘)
inline float centerlineHeight(float s) {
    float base = -kBaseConcave * (1.0 - smoothstep(0.0, 0.3, s));
    float tip = kTipLift * smoothstep(0.6, 1.0, s);
    return base + tip;
}

// 边缘相对中线的下垂量 (实现横向V形截面)
inline float edgeDropAt(float s) {
    float intensity = smoothstep(0.1, 0.4, s) * (1.0 - smoothstep(0.7, 0.95, s));
    return -kEdgeDrop * intensity;
}


// 绕局部 y 轴（长度方向）扭转
inline float3 applyLocalTwist(float3 p, float s) {
  const float DEG2RAD = 0.017453292519943295f; // pi/180
  float theta = (kTwistMaxDeg * DEG2RAD) * s;
  float c = cos(theta), si = sin(theta);
  float x =  c * p.x + si * p.z;
  float z = -si * p.x + c * p.z;
  return float3(x, p.y, z);
}

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
  float2 _padding; // align with Swift Params
} Params;

struct CellBase {
  float3 position;        // 花瓣在花朵内的相对位置
  float3 color;           // 花瓣颜色（与所属花朵一致）
  float flowerId;         // 所属花朵的ID
  float3 flowerCenter;    // 花朵中心位置
  float petalId;          // 花瓣ID（0-...）
  float lineType;         // 线条类型（0=轮廓，1=填充）
  float petalAngle;       // 花瓣在花朵中的角度
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
                                        uint vid [[vertex_id]],
                                        constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
                                        constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
                                        constant Params &params [[buffer(BufferIndexParams)]],
                                        const device CellBase *petalData [[buffer(BufferIndexBase)]]) {
  FlowerInOut out;
  UniformsPerView uniformsPerView = uniforms.perView[amp_id];

  // 依据 seed 获取花瓣实例数据
  int petalIndex = in.seed;
  CellBase petalInfo = petalData[petalIndex];

  // 动画（保持原先的朵级动画）
  float flowerId = petalInfo.flowerId;
  float time = params.time;
  float rotationSpeed = 0.3 + flowerId * 0.15;
  float rotationPhase = flowerId * 1.57;
  float rotationAngle = time * rotationSpeed + rotationPhase;
  float floatSpeed = 0.8 + flowerId * 0.2;
  float floatPhase = flowerId * 2.1;
  float verticalOffset = sin(time * floatSpeed + floatPhase) * 0.15;
  float swaySpeed = 0.6 + flowerId * 0.1;
  float swayPhase = flowerId * 1.3;
  float horizontalSway = sin(time * swaySpeed + swayPhase) * 0.08;

  // 局部参数
  float L  = max(petalInfo.petalSize, 1e-4);
  float Wb = (kWb / kL) * L;
  float Wm = (kWm / kL) * L;
  float Wt = (kWt / kL) * L;
  
  // 为每个花瓣引入轻微随机的立体感缩放，增强自然感
  float shapeScale = mix(0.9, 1.1, random1D(petalInfo.petalId + petalInfo.flowerId * 13.0));

  // 从 vertex_id 推导所在花瓣内的元素与端点
  uint localVertex = vid % kVerticesPerPetal;
  uint elementIdx = localVertex / 2;   // 第几个线段（轮廓/填充）
  uint endpoint   = localVertex & 1;   // 0 或 1，线段的两个端点

  float3 localPos = float3(0.0);
  
  if (elementIdx < kSegmentsPerPetal) {
    // 轮廓：前半段取右边界，后半段取左边界
    uint halfSeg = kSegmentsPerPetal / 2;
    float s;
    float2 b;
    if (elementIdx < halfSeg) {
        float tA = float(elementIdx) / float(halfSeg);
        float tB = float(elementIdx + 1) / float(halfSeg);
        s = (endpoint == 0) ? tA : tB;
        b = petalBoundaryR(L, Wb, Wm, Wt, s);
    } else {
        uint idx = elementIdx - halfSeg;
        float tA = float(idx) / float(halfSeg);
        float tB = float(idx + 1) / float(halfSeg);
        s = (endpoint == 0) ? (1.0 - tA) : (1.0 - tB);
        b = petalBoundaryL(L, Wb, Wm, Wt, s);
    }
    float z = (centerlineHeight(s) + edgeDropAt(s)) * shapeScale;
    localPos = float3(b.x, b.y, z);
  } else {
    // 填充线：中心到边界，左右交替
    uint fillIdx = elementIdx - kSegmentsPerPetal;
    float s = (kLinesPerPetal > 1) ? (float(fillIdx) / float(kLinesPerPetal - 1)) : 0.0;
    bool toRight = (fillIdx % 2u) == 0u;
    
    float2 center2 = petalCenterline(L, s);
    float zCenter = centerlineHeight(s) * shapeScale;
    
    float2 edge2 = toRight ? petalBoundaryR(L, Wb, Wm, Wt, s)
                           : petalBoundaryL(L, Wb, Wm, Wt, s);
    float zEdge = (centerlineHeight(s) + edgeDropAt(s)) * shapeScale;
    
    float3 p0 = float3(center2.x, center2.y, zCenter);
    float3 p1 = float3(edge2.x,   edge2.y,   zEdge);
    localPos = (endpoint == 0) ? p0 : p1;
  }

  // 扭转（绕局部长度方向）
  localPos = applyLocalTwist(localPos, clamp(localPos.y / max(L, 1e-4), 0.0, 1.0));

  // 将局部花瓣坐标映射到世界局部基：
  // - 长度(local y) -> 水平径向(X)
  // - 拱弧(local z) -> 垂直高度(Y)
  // - 宽度(local x) -> 水平切向(Z)
  float3 pWorldLocal = float3(localPos.y, localPos.z, localPos.x);

  // 基于长度的整体下垂（绕切向 Z 轴旋转），尖端更明显，带轻微随机
  {
    float sLen = clamp(localPos.y / max(L, 1e-4), 0.0, 1.0);
    float droopRandDeg = mix(0.0, 8.0, random1D(petalInfo.petalId * 5.13 + petalInfo.flowerId * 2.71));
    float droopDeg = 10.0 + droopRandDeg; // 10° 基础 + 随机 0..8°
    float droop = -(droopDeg * 0.017453292519943295f) * smoothstep(0.15, 0.95, sLen);
    float cd = cos(droop), sd = sin(droop);
    float xDroop = pWorldLocal.x * cd - pWorldLocal.y * sd;
    float yDroop = pWorldLocal.x * sd + pWorldLocal.y * cd;
    pWorldLocal.x = xDroop;
    pWorldLocal.y = yDroop;
  }

  // 绕世界 Y 轴按 petalAngle 进行布置（花瓣绕花心均匀排布）
  float ca = cos(petalInfo.petalAngle), sa = sin(petalInfo.petalAngle);
  float3 petalLocalOriented = float3(
    pWorldLocal.x * ca - pWorldLocal.z * sa,
    pWorldLocal.y,
    pWorldLocal.x * sa + pWorldLocal.z * ca
  );

  // 叠加朵级动画（绕世界 y 的自转 + 摆动/浮动 + 平移到花心）
  float c = cos(rotationAngle), s = sin(rotationAngle);
  float3 rotatedPos = float3(
    petalLocalOriented.x * c - petalLocalOriented.z * s,
    petalLocalOriented.y,
    petalLocalOriented.x * s + petalLocalOriented.z * c
  );

  float3 animatedCenter = petalInfo.flowerCenter + float3(horizontalSway, verticalOffset, 0.0);
  float4 position = float4(animatedCenter + rotatedPos, 1.0);

  position = applyGestureViewer(
                                position,
                                params.viewerPosition,
                                params.viewerScale,
                                params.viewerRotation,
                                uniforms.cameraPos);

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(petalInfo.color, 1.0);
  return out;
}

fragment float4 flowersFragmentShader(FlowerInOut in [[stage_in]]) {
    // 简化实现，直接返回颜色
  return in.color;
}
