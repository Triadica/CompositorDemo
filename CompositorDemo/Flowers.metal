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

// 桃花 (Peach Blossom) 花瓣参数
// 修复：针对三角形网格的分段参数
constant int kRadialSegments = 32;  // 长度方向分段
constant int kLateralSegments = 16; // 宽度方向分段
// 每个长方形由 2 个三角形组成 = 6 个顶点
constant uint kVerticesPerPetal = kRadialSegments * kLateralSegments * 6;

// 基本外形比例 (以长度 L 为基准)
constant float kL = 0.55;       // 进一步放大整体结构 (0.32 -> 0.55)
constant float kWb = 0.08 * kL; // 极窄基部
constant float kWm = 0.85 * kL; // 宽阔中部 (桃花更丰满)
constant float kWt = 0.30 * kL; // 尖端
constant float kSPeak = 0.70;   // 最宽处偏顶部

// 3D 立体形态参数
constant float kBaseConcave = 0.12 * kL;
// kTipLift 已移除
constant float kEdgeUp = 0.22 * kL;
constant float kTwistMaxDeg = 3.0;

// 三次贝塞尔
inline float2
    cubicBezier2(float2 p0, float2 p1, float2 p2, float2 p3, float t) {
  float u = 1.0 - t;
  return u * u * u * p0 + 3.0 * u * u * t * p1 + 3.0 * u * t * t * p2 +
         t * t * t * p3;
}

// 桃花瓣边界：倒卵形 + 尖端微缺 (Notch)
inline float2 petalBoundaryL(float L, float Wb, float Wm, float Wt, float s) {
  float2 P0 = float2(-0.5 * Wb, 0.0);
  float2 P1 = float2(-0.4 * Wm, 0.2 * L); // 更圆润的展开
  float2 P2 = float2(-0.5 * Wm, kSPeak * L);
  float2 P3 = float2(-0.5 * Wt, L);

  // 模拟桃花尖端的标志性缺口
  float notch = 1.0;
  if (s > 0.93) {
    notch = 1.0 - smoothstep(0.93, 1.0, s) * 0.15;
  }

  float2 pos = cubicBezier2(P0, P1, P2, P3, clamp(s, 0.0, 1.0));
  pos.x *= notch;
  return pos;
}

// 纵向弯曲
inline float centerlineHeight(float s) {
  // 桃花瓣相对舒展，基部微凹，中后部略高
  return -kBaseConcave * sin(s * 3.14159f);
}

// 边缘提升（浅碟状截面）
inline float edgeDropAt(float s, float sideways) {
  // sideways: 0 为中心, 1 为边缘
  float arch = sideways * sideways;
  return kEdgeUp * arch * smoothstep(0.05, 0.8, s);
}

// 计算花瓣表面任意一个点的 3D 坐标
// s: 长度比例 (0..1), sideways: 横向比例 (-1..1)
inline float3 computeSurfacePoint(
    float L,
    float Wb,
    float Wm,
    float Wt,
    float s,
    float sideways,
    float shapeScale) {
  float2 bL = petalBoundaryL(L, Wb, Wm, Wt, s);
  // 核心修复：对称性逻辑确保中间对齐。
  // bL.x 是左侧边界(负值)，我们取 abs(bL.x) 得到宽度
  float width = abs(bL.x);

  // X: 横向位置 (sideways 为 -1..1)
  // 当 sideways 为 0 时，x 严格为 0，确保左右两半完全贴合
  float x = sideways * width;
  // Y: 沿长度方向
  float y = s * L;
  // Z: 碟形立体高度
  float zCenter = centerlineHeight(s);
  float zEdge = edgeDropAt(s, abs(sideways));
  float z = (zCenter + zEdge) * shapeScale;

  return float3(x, y, z);
}

// 绕局部 y 轴（长度方向）扭转
inline float3 applyLocalTwist(float3 p, float s) {
  const float DEG2RAD = 0.017453292519943295f; // pi/180
  float theta = (kTwistMaxDeg * DEG2RAD) * s;
  float c = cos(theta), si = sin(theta);
  float x = c * p.x + si * p.z;
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
  float3 position;     // 花瓣在花朵内的相对位置
  float3 color;        // 花瓣颜色（与所属花朵一致）
  float flowerId;      // 所属花朵的ID
  float3 flowerCenter; // 花朵中心位置
  float petalId;       // 花瓣ID（0-...）
  float lineType;      // 线条类型（0=轮廓，1=填充）
  float petalAngle;    // 花瓣在花朵中的角度
  float petalSize;     // 花瓣大小
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
  float L = max(petalInfo.petalSize, 1e-4);
  float Wb = (kWb / kL) * L;
  float Wm = (kWm / kL) * L;
  float Wt = (kWt / kL) * L;

  // 为每个花瓣引入轻微随机的立体感缩放，增强自然感
  float shapeScale =
      mix(0.85, 1.15, random1D(petalInfo.petalId + petalInfo.flowerId * 13.0));

  // 从 vertex_id 推导所在的三角形网格位置 (Triangle-ready topology)
  uint triIdx = (vid % kVerticesPerPetal) / 6;
  uint vertIdxInTri = vid % 6;

  // 网格行列坐标
  uint r = triIdx / kLateralSegments;
  uint c = triIdx % kLateralSegments;

  // 决定当前顶点在网格 cell (r, c) 中的归一化 UV (s: 0..1, sw: -1..1)
  float s = 0.0, sw = 0.0;

  // 三角形顶点顺序映射 (0,0)-(1,0)-(0,1) and (1,0)-(1,1)-(0,1)
  uint i = r, j = c;
  if(vertIdxInTri == 0) { i = r;   j = c; }
  else if(vertIdxInTri == 1) { i = r+1; j = c; }
  else if(vertIdxInTri == 2) { i = r;   j = c+1; }
  else if(vertIdxInTri == 3) { i = r+1; j = c; }
  else if(vertIdxInTri == 4) { i = r+1; j = c+1; }
  else if(vertIdxInTri == 5) { i = r;   j = c+1; }

  s = float(i) / float(kRadialSegments);
  sw = (float(j) / float(kLateralSegments)) * 2.0 - 1.0;

  float3 localPos = computeSurfacePoint(L, Wb, Wm, Wt, s, sw, shapeScale);

  // 扭转（绕局部长度方向）
  localPos =
      applyLocalTwist(localPos, clamp(localPos.y / max(L, 1e-4), 0.0, 1.0));

  // 将局部花瓣坐标映射到世界局部基：
  // - 长度(local y) -> 水平径向(X)
  // - 拱弧(local z) -> 垂直高度(Y)
  // - 宽度(local x) -> 水平切向(Z)
  float3 pWorldLocal = float3(localPos.y, localPos.z, localPos.x);

  // 基于长度的整体下垂（绕切向 Z 轴旋转），尖端更明显，带轻微随机
  {
    float sLen = clamp(localPos.y / max(L, 1e-4), 0.0, 1.0);
    float droopRandDeg =
        mix(0.0,
            8.0,
            random1D(petalInfo.petalId * 5.13 + petalInfo.flowerId * 2.71));
    float droopDeg = 10.0 + droopRandDeg; // 10° 基础 + 随机 0..8°
    float droop =
        -(droopDeg * 0.017453292519943295f) * smoothstep(0.15, 0.95, sLen);
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
      pWorldLocal.x * sa + pWorldLocal.z * ca);

  // 叠加朵级动画（绕世界 y 的自转 + 摆动/浮动 + 平移到花心）
  float cr = cos(rotationAngle), sr = sin(rotationAngle);
  float3 rotatedPos = float3(
      petalLocalOriented.x * cr - petalLocalOriented.z * sr,
      petalLocalOriented.y,
      petalLocalOriented.x * sr + petalLocalOriented.z * cr);

  float3 animatedCenter =
      petalInfo.flowerCenter + float3(horizontalSway, verticalOffset, 0.0);
  float4 position = float4(animatedCenter + rotatedPos, 1.0);

  position = applyGestureViewer(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation,
      uniforms.cameraPos);

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(petalInfo.color, 1.0);
  (void)tintUniform; // 明确标记未使用，消除 warning
  return out;
}

fragment float4 flowersFragmentShader(FlowerInOut in [[stage_in]]) {
  // 简化实现，直接返回颜色
  return in.color;
}
