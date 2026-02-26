/*
 See the LICENSE.txt file for this sample's licensing information.

 Abstract:
 Simulation of photon (null geodesic) trajectories near a Schwarzschild black
 hole.

 Physics:
   Schwarzschild metric in isotropic coordinates, c = G = 1.
   Null-geodesic acceleration (1PN, exact for the photon-sphere topology):

     a = -2M/r³ · [(1 + v²)·r⃗ − 2·(v⃗·r̂)·r·v⃗]

   where r = |pos − bhCenter|,  v⃗ = photon velocity,  M = kBHMass.

   Photon sphere (unstable circular orbit) sits at r = 4M in these coordinates.
   Schwarzschild horizon at rs = 2M; photons reaching r < 0.9·rs are swallowed.

   Integration: 4th-order Runge–Kutta for accurate orbit shapes.

 Recycling:
   • Swallowed: r < rs · 0.85
   • Escaped:   r > kEscapeR
   • Too old:   age > kMaxAge seconds
   • Stuck:     low-speed timer > kMaxStuck seconds

 Trail:
   Each particle head extends a fixed-length history chain.
   In the vertex shader, segments shorter than kMinSegLen become degenerate
   (zero-area triangles, discarded by rasteriser) → speed ∝ visible trail.

 Color:
   Warm (orange/gold) ↔ cool (blue/white) gradient by initial emission latitude.
   Near the black hole the vertex shader blueshifts all photons slightly.
 */

// clang-format off
#include <metal_stdlib>
#include <simd/simd.h>
#import "ShaderTypes.h"   // must precede PathProperties.h (defines NS_ENUM macros)
#import "PathProperties.h"
// clang-format on

using namespace metal;

// ─── Struct definitions
// ───────────────────────────────────────────────────────

typedef struct {
  float3 position [[attribute(0)]];
  int lineNumber [[attribute(1)]];
  int groupNumber [[attribute(2)]];
  int cellSide [[attribute(3)]];
} BHVertexIn;

typedef struct {
  float4 position [[position]];
  float4 color;
} BHInOut;

/// CPU/GPU shared params struct.
/// All float3 members are packed (12 bytes, 4-byte aligned) – matches Swift
/// SIMD3<Float>.
typedef struct {
  float time;            // offset  0
  float elapsed;         // offset  4
  int groupSize;         // offset  8
  float3 viewerPosition; // offset 12 (packed float3)
  float viewerScale;     // offset 24
  float viewerRotation;  // offset 28
  int totalLines;        // offset 32
} BHParams;

struct BHBase {
  float3 position; // current world position
  float3 color;    // frozen at particle birth
  float3 velocity; // current velocity vector
  float3 extra;    // x = age(s), y = colorParam[0..1], z = stuckTimer(s)
};

// ─── Constants
// ────────────────────────────────────────────────────────────────

/// Black hole gravitational parameter GM (scene units).
constant float kBHMass = 0.30;
/// Schwarzschild radius rs = 2M.
constant float kBHrs = 0.60;
/// Black hole centre in world space (2 m in front, centred).
constant float3 kBHCenter = float3(0.0, 0.0, -2.0);
/// Emission sphere radius (photons start here).
constant float kEmitR = 20.0f;
/// Photons farther than this are considered escaped and recycled.
constant float kEscapeR = 30.0f;
/// Maximum particle lifetime in physics seconds (4× extended for long orbits
/// near the photon sphere).
constant float kMaxAge = 720.0f;
/// Maximum time with near-zero speed before forced recycle.
constant float kMaxStuck = 1.0f;
/// Speed threshold for "stuck" detection (scene units/s).
constant float kStuckSpeedThr = 0.05;
/// Segment length below which the ribbon quad is collapsed to invisible.
constant float kMinSegLen = 0.0003;

// ─── Hash / emission helpers ─────────────────────────────────────────────────

static float hashF(uint seed) {
  seed ^= seed << 13u;
  seed ^= seed >> 17u;
  seed ^= seed << 5u;
  return float(seed & 0xFFFFFu) / float(0xFFFFFu);
}

/// Initialise (or re-initialise) photon idx at birth.
/// All values are deterministic from idx so recycling re-uses the same slot.
/// idx should be varied across recycles (e.g. by adding a time-based offset).
static void bhInitState(
    uint idx,
    thread float3 &outPos,
    thread float3 &outVel,
    thread float3 &outColor,
    thread float &outColorParam) {
  float h1 = hashF(idx * 7u + 1u);
  float h2 = hashF(idx * 7u + 2u);
  float h3 = hashF(idx * 7u + 3u);
  float h4 = hashF(idx * 7u + 4u);
  float h5 = hashF(idx * 7u + 5u);

  // ── Emission point on sphere ─────────────────────────────────────────────
  float phi_e = h1 * 2.0f * M_PI_F;
  float cosT_e = 2.0f * h2 - 1.0f;
  float sinT_e = sqrt(max(0.0f, 1.0f - cosT_e * cosT_e));
  float3 emitDir = float3(sinT_e * cos(phi_e), sinT_e * sin(phi_e), cosT_e);
  outPos = kBHCenter + emitDir * kEmitR;

  // ── Velocity direction ───────────────────────────────────────────────────
  // Real Schwarzschild capture: photons with impact parameter
  // b = r·sin θ < 3√3·M ≈ 1.56 (at r=20 that's only ~4.5°) are captured.
  // We replicate this by giving nearly all photons a very small tangential
  // spread so almost all fall straight in.  Only equatorial emitters
  // (accretion disk, |emitDir.y| < 0.28) receive a larger tangential kick
  // so some orbit the photon sphere and escape outward.
  float phi_v = h3 * 2.0f * M_PI_F;
  float cosT_v = 2.0f * h4 - 1.0f;
  float sinT_v = sqrt(max(0.0f, 1.0f - cosT_v * cosT_v));
  float3 rndDir = float3(sinT_v * cos(phi_v), sinT_v * sin(phi_v), cosT_v);
  float absy = abs(emitDir.y);
  float diskFactor =
      smoothstep(0.28f, 0.0f, absy); // 1 at equator, 0 outside disk
  float tangential =
      mix(0.04f, 0.55f, diskFactor); // 0.04 (polar) … 0.55 (equatorial)
  float3 velDir =
      normalize(rndDir * tangential + (-emitDir) * (1.0f - tangential));

  // Speed: photons travel at c = 1 (scene units / physics second).
  outVel = velDir;

  // ── Color ────────────────────────────────────────────────────────────────
  // Emission latitude (0 = south pole, 1 = north pole) drives warm↔cool blend.
  float latitude = emitDir.y * 0.5f + 0.5f;    // [0,1]
  float3 warmCol = float3(1.0f, 0.72f, 0.25f); // gold / accretion-disk
  float3 coolCol = float3(0.35f, 0.75f, 1.0f); // starlight blue
  outColor = mix(warmCol, coolCol, h5 * 0.6f + latitude * 0.4f);
  outColorParam = latitude;
}

// ─── Schwarzschild photon force
// ───────────────────────────────────────────────

/// Acceleration of a massless particle (photon) in Schwarzschild geometry
/// in isotropic harmonic coordinates, c = G = 1.
/// Derived from the 1PN null-geodesic equations:
///   a⃗ = (-2M / r³) · [(1 + v²)·r⃗ − 2·(v⃗·r⃗/r)·r·v⃗]
static float3 bhAccel(float3 pos, float3 vel) {
  float3 r_vec = pos - kBHCenter;
  float r = length(r_vec);
  if (r < 1e-4f)
    return float3(0.0f);

  float v2 = dot(vel, vel);
  float vr = dot(vel, r_vec / r); // radial speed component

  return (-2.0f * kBHMass / (r * r * r)) *
         ((1.0f + v2) * r_vec - 2.0f * vr * r * vel);
}

// ─── Viewer transform (identical to other demos)
// ──────────────────────────────

static float4
    applyGestureViewer(float4 p, float3 vpos, float vscale, float vrot) {
  float cT = cos(vrot), sT = sin(vrot);
  float x = p.x * cT - p.z * sT;
  float z = p.x * sT + p.z * cT;
  p.x = x;
  p.z = z;
  p *= vscale;
  p = p - float4(vpos, 0.0f);
  return p;
}

// ─── Compute kernel
// ───────────────────────────────────────────────────────────

kernel void blackHoleComputeShader(
    device BHBase *particles [[buffer(0)]],
    device BHBase *outputParticles [[buffer(1)]],
    constant BHParams &params [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  BHBase cell = particles[id];
  device BHBase &out = outputParticles[id];

  bool leading = (id % uint(params.groupSize + 1) == 0);

  if (leading) {
    float3 pos = cell.position;
    float3 vel = cell.velocity;
    float age = cell.extra.x;
    float stuckT = cell.extra.z;

    // ── Physics time step (reduced to ~1/4 for slower visual motion) ─────
    float dt = params.elapsed * 0.5f;
    age += dt;

    // ── RK4 integration ───────────────────────────────────────────────────
    float3 k1v = bhAccel(pos, vel) * dt;
    float3 k1x = vel * dt;

    float3 k2v = bhAccel(pos + k1x * 0.5f, vel + k1v * 0.5f) * dt;
    float3 k2x = (vel + k1v * 0.5f) * dt;

    float3 k3v = bhAccel(pos + k2x * 0.5f, vel + k2v * 0.5f) * dt;
    float3 k3x = (vel + k2v * 0.5f) * dt;

    float3 k4v = bhAccel(pos + k3x, vel + k3v) * dt;
    float3 k4x = (vel + k3v) * dt;

    float3 newPos = pos + (k1x + 2.0f * k2x + 2.0f * k3x + k4x) / 6.0f;
    float3 newVel = vel + (k1v + 2.0f * k2v + 2.0f * k3v + k4v) / 6.0f;

    // ── Stuck timer ───────────────────────────────────────────────────────
    float spd = length(newVel);
    stuckT = (spd < kStuckSpeedThr) ? stuckT + dt : 0.0f;

    // ── Recycle decision ──────────────────────────────────────────────────
    float dist = length(newPos - kBHCenter);
    bool swallowed = dist < kBHrs * 0.85f;
    bool escaped = dist > kEscapeR;
    bool tooOld = age > kMaxAge;
    bool stuck = stuckT > kMaxStuck;

    if (swallowed || escaped || tooOld || stuck) {
      // Re-seed with a varying index so recycled particles differ from their
      // original birth state (time-based offset decorrelates siblings).
      uint lineIdx = id / uint(params.groupSize + 1);
      uint seed = lineIdx + uint(params.time * 31.7f);
      float3 c;
      float cp;
      bhInitState(seed, newPos, newVel, c, cp);
      cell.color = c;
      age = 0.0f;
      stuckT = 0.0f;
    }

    out.position = newPos;
    out.velocity = newVel;
    out.color = cell.color;
    out.extra = float3(age, cell.extra.y, stuckT);

  } else {
    // Trail: copy from the freshly-written leader entry above.
    out = outputParticles[id - 1];
  }
}

// ─── Vertex shader
// ────────────────────────────────────────────────────────────

vertex BHInOut blackHoleVertexShader(
    BHVertexIn in [[stage_in]],
    ushort amp_id [[amplification_id]],
    constant Uniforms &uniforms [[buffer(BufferIndexUniforms)]],
    constant TintUniforms &tintUniform [[buffer(BufferIndexTintUniforms)]],
    constant BHParams &params [[buffer(BufferIndexParams)]],
    const device BHBase *linesData [[buffer(BufferIndexBase)]]) {
  BHInOut out;

  UniformsPerView uniformsPerView = uniforms.perView[amp_id];
  float3 cameraDir = uniforms.cameraDirection;

  int stride = params.groupSize + 1;
  int lineNumber = in.lineNumber;
  int grp = in.groupNumber;
  int side = in.cellSide;

  BHBase cell = linesData[lineNumber * stride + grp + 1];
  BHBase prevCell = linesData[lineNumber * stride + grp];

  // ── Trail ribbon ─────────────────────────────────────────────────────────
  float3 dir = cell.position - prevCell.position;
  float segLen = length(dir);
  if (segLen < 1e-5f)
    dir = float3(1e-4f, 0.0f, 0.0f);

  // Suppress ghost lines on recycle: a large age discontinuity between the
  // two trail entries means a recycle just happened and the history is stale.
  float ageDiff = abs(cell.extra.x - prevCell.extra.x);
  // Speed → trail length: segments below threshold become degenerate quads.
  float3 brush = (segLen < kMinSegLen || ageDiff > 0.5f)
                     ? float3(0.0f)
                     : normalize(cross(dir, cameraDir)) * 0.0008f;

  float4 position;
  if (side == 0)
    position = float4(prevCell.position + brush, 1.0f);
  else if (side == 1)
    position = float4(prevCell.position - brush, 1.0f);
  else if (side == 2)
    position = float4(cell.position + brush, 1.0f);
  else
    position = float4(cell.position - brush, 1.0f);

  position = applyGestureViewer(
      position,
      params.viewerPosition,
      params.viewerScale,
      params.viewerRotation);
  position.w = 1.0f;
  out.position = uniformsPerView.modelViewProjectionMatrix * position;

  // ── Gravitational blueshift tint ─────────────────────────────────────────
  // Photons deep in the gravitational well appear blueshifted.
  // blueshift ∈ [0, 1] as r goes from kEscapeR down to kBHrs.
  float dist = length(cell.position - kBHCenter);
  float normalized =
      clamp(1.0f - (dist - kBHrs) / (kEscapeR - kBHrs), 0.0f, 1.0f);
  float blueshift = normalized * normalized; // quadratic ramp near horizon

  float3 col = cell.color;
  // Push toward blue-white as blueshift increases
  col = mix(col, float3(0.8f, 0.9f, 1.0f), blueshift * 0.55f);
  // Slightly increase brightness close to the horizon
  col = col * (1.0f + blueshift * 0.4f);

  out.color = float4(col, 1.0f);
  return out;
}

// ─── Fragment shader
// ──────────────────────────────────────────────────────────

fragment float4 blackHoleFragmentShader(BHInOut in [[stage_in]]) {
  if (in.color.a <= 0.0f)
    discard_fragment();
  return in.color;
}
