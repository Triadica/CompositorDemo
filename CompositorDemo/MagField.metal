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

static uint mixBits(uint x) {
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

static float rand01(uint seed) {
  return float(mixBits(seed) & 0x00FFFFFFu) / 16777215.0f;
}

/// Earth magnetic dipole field at world position `pos`.
/// Dipole is at `dipoleCenter`, moment pointing +y, strength `M`.
/// B = M * ( 3*(m̂·r̂)*r̂ - m̂ ) / |r|³
static float3 earthDipoleField(float3 pos, float3 dipoleCenter, float M) {
  float3 r = pos - dipoleCenter;
  float rLen = length(r);
  // Prevent division by zero and singularity at the core (Earth's radius)
  if (rLen < 0.12)
    return float3(0.0);
  float3 rHat = r / rLen;
  float3 mDir = float3(0.0, 1.0, 0.0); // Dipole axis (Magnetic South ≈ +y)
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

/// 2×2×2 cubic region centred on kDipoleCenter.
constant float3 kBoxCenter = float3(0.0, 0.0, -2.0);
/// Earth center is exactly at cube center.
constant float3 kDipoleCenter = kBoxCenter;
/// Dipole strength (tuned for capturing particles).
constant float kDipoleStrength = 4.5;
constant float kBoxHalf = 1.0f;   // half-extent → full size = 2
constant float kFlowSpeed = 0.18; // solar wind speed (uniform, slower)
/// Magnetosphere boundary: dipole force only applied within this radius.
constant float kMagnetosphereRadius = 0.95;
constant float kSpawnWindow = 5.0f; // seconds for staggered emission

/// Emit a particle from the left face of the 2×2×2 box with rightward velocity.
static void initialState(
    uint lineIdx,
    uint spawnNonce,
    int totalLines,
    thread float3 &outPos,
    thread float3 &outVel,
    thread float3 &outColor,
    thread float &outCharge) {
  bool positive = lineIdx < uint(totalLines / 2);
  outCharge = positive ? 1.0 : -1.0;
  outColor = positive ? float3(1.0, 0.38, 0.07)  // orange-red
                      : float3(0.07, 0.48, 1.0); // cyan-blue

  float h1 =
      rand01(lineIdx * 747796405u + spawnNonce * 2891336453u + 277803737u);
  float h2 =
      rand01(lineIdx * 3266489917u + spawnNonce * 668265263u + 2246822519u);

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
    float age =
        cell.extra
            .y; // accumulated physics time (seconds), can be <0 as spawn delay
    float stuckT = cell.extra.z; // time spent with near-zero speed
    float3 color = cell.color;

    uint lineIdx = id / uint(params.groupSize + 1);

    // ── Scaled time step (higher for integration stability) ───────────────
    float dt = params.elapsed * 0.45f;
    age += dt;

    // Spawn delay phase: keep particle at inlet before activation.
    if (age < 0.0f) {
      out = cell;
      out.extra.y = age;
      return;
    }

    // ── Magnetic force: F = q * (v × B), only inside the magnetosphere ──
    float3 rRel = pos - kDipoleCenter;
    float distSq = dot(rRel, rRel);
    float distToDipole = sqrt(distSq);
    float3 force = float3(0.0);

    if (distToDipole < kMagnetosphereRadius) {
      // 1. Calculate B-field
      float3 B = earthDipoleField(pos, kDipoleCenter, kDipoleStrength);

      // 2. Lorentz Force: F = q * (v × B)
      force = charge * cross(vel, B);

      // 3. Collision with Earth (radius 0.12)
      // If particles hit the atmosphere at the poles, they should vanish or
      // "glow".
      if (distToDipole < 0.16) {
        // If close to axis (poles), it's "trapped" and hits atmosphere
        if (abs(rRel.y) > 0.06) {
          // Mark as recycle (too old / stuck)
          age = 100.0;
        } else {
          // Rebound/Bounce off earth mantle
          float3 n = rRel / distToDipole;
          pos = kDipoleCenter + n * 0.161;
          vel = reflect(vel, n) * 0.4; // lose even more energy
        }
      }
    }

    // ── Semi-implicit Euler integration (energy-conserving speed clamp) ─
    float baseSpeed = max(length(vel), 1e-4f);
    float3 newVel = vel + force * dt;
    float newLen = length(newVel);
    if (newLen > 1e-4f) {
      // Stronger atmospheric drag at poles to simulate trapping/energy loss
      float polarTrap = (abs(rRel.y) > 0.15 && distToDipole < 0.4) ? 0.92 : 1.0;
      newVel = (newVel / newLen) * baseSpeed * polarTrap;
    } else {
      newVel = float3(kFlowSpeed, 0.0f, 0.0f);
    }
    float3 newPos = pos + newVel * dt;

    // ── Visual Glow (Aurora) ──────────────────────────────────────────────
    // Wider aurora trigger and brighter green
    if (distToDipole < 0.35 && abs(rRel.y) > 0.1) {
      color =
          mix(cell.color, float3(0.0, 1.0, 0.2), 0.8); // Brighter Aurora Green
    }

    // ── Recycle conditions: particle left the 2×2×2 box ─────────────────
    float3 rel = newPos - kBoxCenter;
    bool outsideBox = (abs(rel.x) > kBoxHalf) || (abs(rel.y) > kBoxHalf) ||
                      (abs(rel.z) > kBoxHalf);
    bool tooOld = age > 30.0;

    if (outsideBox || tooOld) {
      float3 c;
      float q;
      uint timeTick = uint(max(params.time, 0.0f) * 10000.0f);
      uint seed = mixBits(
          lineIdx * 2246822519u ^ timeTick * 3266489917u ^
          as_type<uint>(cell.position.y * 997.0f) ^
          as_type<uint>(cell.position.z * 619.0f) ^
          as_type<uint>(age * 431.0f));
      initialState(lineIdx, seed, params.totalLines, newPos, newVel, c, q);
      color = c;
      age = -rand01(seed ^ 0xa511e9b3u) * kSpawnWindow;
      stuckT = 0.0;
    }

    out.position = newPos;
    out.velocity = newVel;
    out.color = color;
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
