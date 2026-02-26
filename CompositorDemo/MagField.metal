/*
 See the LICENSE.txt file for this sample's licensing information.

 Abstract:
 Earth magnetic dipole field simulation.
 Positive and negative charged particles are emitted from the left side
 and deflected by the Earth's magnetic dipole field (Lorentz force).
 Positive charges: orange/red  |  Negative charges: cyan/blue
 Physics: F = q * (v × B),  B = M*(3*(m̂·r̂)r̂ - m̂)/|r|³
 */

// clang-format off
#include <metal_stdlib>
#include <simd/simd.h>
#import "ShaderTypes.h"   // must precede PathProperties.h (defines NS_ENUM macros)
#import "PathProperties.h"
// clang-format on

using namespace metal;

// ─── Data types ──────────────────────────────────────────────────────────────

typedef struct {
  float3 position [[attribute(0)]];
  int lineNumber [[attribute(1)]];
  int groupNumber [[attribute(2)]];
  int cellSide [[attribute(3)]];
} MagFieldVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
} MagFieldInOut;

typedef struct {
  float time;
  float elapsed;
  int groupSize;
  float3 viewerPosition; // offset 12
  float viewerScale;     // offset 24
  float viewerRotation;  // offset 28
  int totalLines;        // offset 32
} MagFieldParams;

struct MagFieldBase {
  float3 position;
  float3 color;
  float3 velocity;
  float3 extra; // x = charge (+1 or -1)
};

// ─── Helper functions
// ─────────────────────────────────────────────────────────

/// Low-quality but fast hash → [0, 1)
static float hashFloat(uint seed) {
  seed ^= seed << 13u;
  seed ^= seed >> 17u;
  seed ^= seed << 5u;
  return float(seed & 0xFFFFFu) / float(0xFFFFFu);
}

/// Earth magnetic dipole field at world position `pos`.
/// Dipole is at `dipoleCenter`, moment pointing +y, strength `M`.
/// B = M * ( 3*(m̂·r̂)*r̂ - m̂ ) / |r|³
static float3 earthDipoleField(float3 pos, float3 dipoleCenter, float M) {
  float3 r = pos - dipoleCenter;
  float rLen = length(r);
  if (rLen < 0.08)
    return float3(0.0);
  float3 rHat = r / rLen;
  float3 mDir = float3(0.0, 1.0, 0.0); // dipole axis (geographic north ≈ +y)
  float mDotR = dot(mDir, rHat);
  // Dipole formula
  return M * (3.0 * mDotR * rHat - mDir) / (rLen * rLen * rLen);
}

/// Apply viewer gesture transform (matches other demos).
static float4 applyGestureViewer(
    float4 p, float3 viewerPosition, float viewerScale, float viewerRotation) {
  float cosT = cos(viewerRotation);
  float sinT = sin(viewerRotation);
  float x = p.x * cosT - p.z * sinT;
  float z = p.x * sinT + p.z * cosT;
  p.x = x;
  p.z = z;
  p *= viewerScale;
  p = p - float4(viewerPosition, 0.0);
  return p;
}

// ─── Constants
// ────────────────────────────────────────────────────────────────

/// Dipole center in world space (roughly 2 m in front of the user, centred).
constant float3 kDipoleCenter = float3(0.0, 0.0, -2.0);
/// Dipole strength – tuned for clearly visible Larmor-radius curvature.
constant float kDipoleStrength = 3.5;
/// 2×2×2 cubic region centred on kDipoleCenter.
constant float3 kBoxCenter = float3(0.0, 0.0, -2.0);
constant float kBoxHalf = 1.0f;   // half-extent → full size = 2
constant float kFlowSpeed = 0.5;  // rightward flow speed (uniform)
/// Magnetosphere boundary: dipole force only applied within this radius.
constant float kMagnetosphereRadius = 1.0;

/// Emit a particle from the left face of the 2×2×2 box with rightward velocity.
static void initialState(
    uint lineIdx,
    int totalLines,
    float time,
    thread float3 &outPos,
    thread float3 &outVel,
    thread float3 &outColor,
    thread float &outCharge) {
  bool positive = lineIdx < uint(totalLines / 2);
  outCharge = positive ? 1.0 : -1.0;
  outColor = positive ? float3(1.0, 0.38, 0.07)  // orange-red
                      : float3(0.07, 0.48, 1.0); // cyan-blue

  float h1 = hashFloat(lineIdx * 7u + 1u);
  float h2 = hashFloat(lineIdx * 7u + 2u);

  // Random position on the left face (x = boxCenter.x - boxHalf)
  float y = kBoxCenter.y + (h1 * 2.0f - 1.0f) * kBoxHalf;
  float z = kBoxCenter.z + (h2 * 2.0f - 1.0f) * kBoxHalf;
  outPos = float3(kBoxCenter.x - kBoxHalf, y, z);
  outVel = float3(kFlowSpeed, 0.0f, 0.0f);
}

// ─── Compute kernel
// ───────────────────────────────────────────────────────────

kernel void magFieldComputeShader(
    device MagFieldBase *particles [[buffer(0)]],
    device MagFieldBase *outputParticles [[buffer(1)]],
    constant MagFieldParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  MagFieldBase cell = particles[id];
  device MagFieldBase &out = outputParticles[id];

  bool leading = (id % uint(params.groupSize + 1) == 0);

  if (leading) {
    float3 pos = cell.position;
    float3 vel = cell.velocity;
    float charge = cell.extra.x;
    float age = cell.extra.y;    // accumulated physics time (seconds)
    float stuckT = cell.extra.z; // time spent with near-zero speed

    uint lineIdx = id / uint(params.groupSize + 1);

    // ── Scaled time step (1/8 of original for slower visual motion) ───────
    float dt = params.elapsed * 0.3125f;
    age += dt;

    // ── Magnetic force: F = q * (v × B), only inside the magnetosphere ──
    float distToDipole = length(pos - kDipoleCenter);
    float3 force = float3(0.0);
    if (distToDipole < kMagnetosphereRadius) {
      // Smoothly ramp force to zero at the boundary to avoid sharp jumps
      float boundary =
          1.0 -
          smoothstep(
              kMagnetosphereRadius * 0.7, kMagnetosphereRadius, distToDipole);
      float3 B = earthDipoleField(pos, kDipoleCenter, kDipoleStrength);
      force = charge * cross(vel, B) * boundary;
    }

    // ── Semi-implicit Euler integration ──────────────────────────────────
    float3 newVel = vel + force * dt;
    float3 newPos = pos + newVel * dt;

    // ── Periodic velocity/direction perturbation ──────────────────────────
    // Keep this extremely small to avoid visible wave bands.
    float pPhase = float(lineIdx) * 2.39996f;
    float pFreq = 0.5f + hashFloat(lineIdx * 5u + 6u) * 0.9f; // [0.5, 1.4] Hz
    float kick =
        sin(params.time * pFreq + pPhase) * 0.001f; // very small perturbation
    float3 pDir = normalize(float3(
        cos(params.time * pFreq * 0.7f + pPhase),
        sin(params.time * pFreq * 1.3f + pPhase * 0.5f),
        sin(params.time * pFreq * 0.4f + pPhase * 1.2f)));
    newVel += pDir * kick;

    // ── Stuck-near-origin timer ───────────────────────────────────────────
    float spd = length(newVel);
    stuckT = (spd < 0.08) ? stuckT + dt : 0.0;

    // ── Recycle conditions: particle left the 2×2×2 box ─────────────────
    float3 rel = newPos - kBoxCenter;
    bool outsideBox = (abs(rel.x) > kBoxHalf) || (abs(rel.y) > kBoxHalf) ||
                      (abs(rel.z) > kBoxHalf);
    bool tooOld = age > 40.0;
    bool stuck = stuckT > 1.0;

    if (outsideBox || tooOld || stuck) {
      float3 c;
      float q;
      uint seed = lineIdx + uint(params.time * 37.0f);
      initialState(seed, params.totalLines, params.time, newPos, newVel, c, q);
      age = hashFloat(seed * 13u + 11u) * 2.0f;
      stuckT = 0.0;
    }

    out.position = newPos;
    out.velocity = newVel;
    out.color = cell.color;
    out.extra = float3(charge, age, stuckT);

  } else {
    // Trail: follow the leading particle's newly-written state
    device MagFieldBase &leader = outputParticles[id - 1];
    out.position = leader.position;
    out.velocity = leader.velocity;
    out.color = leader.color;
    out.extra = leader.extra;
  }
}

// ─── Vertex shader
// ────────────────────────────────────────────────────────────

vertex MagFieldInOut magFieldVertexShader(
    MagFieldVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant MagFieldParams &params [[buffer(BufferIndexParams)]],
    const device MagFieldBase *linesData [[buffer(BufferIndexBase)]]) {
  MagFieldInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  simd_float3 cameraDir = uniforms.cameraDirection;

  int lineNumber = in.lineNumber;
  int groupNumber = in.groupNumber;
  int cellSide = in.cellSide;

  int stride = params.groupSize + 1;
  MagFieldBase cell = linesData[lineNumber * stride + groupNumber + 1];
  MagFieldBase prevCell = linesData[lineNumber * stride + groupNumber];

  float3 dir = cell.position - prevCell.position;
  float segLen = length(dir);
  if (segLen < 0.0001)
    dir = float3(0.001, 0.0, 0.0);

  // Speed → trail length: collapse negligibly short segments.
  // Also suppress ghost lines on recycle: if the two segment endpoints have
  // very different ages (age discontinuity), a recycle just happened and the
  // trail history is stale – collapse to invisible degenerate triangles.
  float ageDiff = abs(cell.extra.y - prevCell.extra.y);
  // Collapse segment if: too short (speed-proportional trail), too long
  // (teleport / recycle artifact), or age discontinuity (stale history).
  float3 brush = (segLen < 0.0003 || segLen > 0.4 || ageDiff > 0.5)
                     ? float3(0.0)
                     : normalize(cross(dir, cameraDir)) * 0.0007;

  float4 position;
  if (cellSide == 0)
    position = float4(prevCell.position + brush, 1.0);
  else if (cellSide == 1)
    position = float4(prevCell.position - brush, 1.0);
  else if (cellSide == 2)
    position = float4(cell.position + brush, 1.0);
  else
    position = float4(cell.position - brush, 1.0);

  position = applyGestureViewer(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation);
  position.w = 1.0;

  out.position = uniformsPerView.modelViewProjectionMatrix * position;
  out.color = float4(cell.color, 1.0);
  return out;
}

// ─── Fragment shader
// ──────────────────────────────────────────────────────────

fragment float4 magFieldFragmentShader(MagFieldInOut in [[stage_in]]) {
  if (in.color.a <= 0.0)
    discard_fragment();
  return in.color;
}
