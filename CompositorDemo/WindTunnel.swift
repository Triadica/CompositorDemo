/*
 See the LICENSE.txt file for this sample's licensing information.

 Abstract:
 Wind tunnel particle demo.
 Particles move left-to-right inside a cylindrical tunnel with a sphere obstacle,
 including pressure/viscosity-like interactions and outlet recycling.
 */

import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

private let maxFramesInFlight = 3
private let linesCount: Int = 192_000
private let lineGroupSize: Int = 2
private var controlCountPerLine: Int { lineGroupSize + 1 }
private var controlCount: Int { linesCount * controlCountPerLine }
private let verticesCount: Int = controlCount * 6
private let indexesCount: Int = controlCount * 6

private func windMixBits(_ x: UInt32) -> UInt32 {
  var v = x
  v ^= v >> 16
  v &*= 0x7feb_352d
  v ^= v >> 15
  v &*= 0x846c_a68b
  v ^= v >> 16
  return v
}

private func windRand01(_ seed: UInt32) -> Float {
  Float(windMixBits(seed) & 0x00FF_FFFF) / 16_777_215.0
}

private struct WindTunnelBase {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var velocity: SIMD3<Float>
  var extra: SIMD3<Float>  // x = age, y = inletRadial, z = stuckTimer
}

private struct Params {
  var time: Float
  var elapsed: Float
  var groupSize: Int32 = Int32(lineGroupSize)
  var viewerPosition: SIMD3<Float>
  var viewerScale: Float
  var viewerRotation: Float = 0.0
  var totalLines: Int32 = Int32(linesCount)
}

@MainActor
class WindTunnelRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  var vertexBuffer: MTLBuffer!
  var indexBuffer: MTLBuffer!

  let computeDevice: MTLDevice
  var computeBuffer: PingPongBuffer?
  let computePipeLine: MTLComputePipelineState
  let computeCommandQueue: MTLCommandQueue

  var gestureManager = GestureManager()
  private var viewStartTime: Date = Date()
  private var frameDelta: Float = 0.0

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    computeDevice = MTLCreateSystemDefaultDevice()!
    let library = computeDevice.makeDefaultLibrary()!
    let kernelFn = library.makeFunction(name: "windTunnelComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: kernelFn)
    computeCommandQueue = computeDevice.makeCommandQueue()!

    buildVertexBuffer(device: layerRenderer.device)
    buildIndexBuffer(device: layerRenderer.device)
    buildComputeBuffer(device: layerRenderer.device)
  }

  private func buildVertexBuffer(device: MTLDevice) {
    let len = MemoryLayout<AttractorCellVertex>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: len)!
    vertexBuffer.label = "WindTunnel vertex buffer"
    let v = vertexBuffer.contents().assumingMemoryBound(to: AttractorCellVertex.self)

    for i in 0..<linesCount {
      let base = i * lineGroupSize * 6
      for j in 0..<lineGroupSize {
        let idx = base + j * 6
        let sides: [Int32] = [0, 1, 2, 1, 2, 3]
        for (k, side) in sides.enumerated() {
          v[idx + k] = AttractorCellVertex(
            position: SIMD3<Float>(0, 0, 0),
            lineNumber: Int32(i),
            groupNumber: Int32(j),
            cellSide: side)
        }
      }
    }
  }

  private func buildIndexBuffer(device: MTLDevice) {
    let len = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: len)!
    indexBuffer.label = "WindTunnel index buffer"
    let p = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: indexesCount)
    for i in 0..<indexesCount { p[i] = UInt32(i) }
  }

  func resetComputeState() {
    buildComputeBuffer(device: computeDevice)
  }

  private func buildComputeBuffer(device: MTLDevice) {
    let len = MemoryLayout<WindTunnelBase>.stride * controlCount
    computeBuffer = PingPongBuffer(device: device, length: len)
    guard let buf = computeBuffer else { return }
    buf.addLabel("WindTunnel compute buffer")

    let ptr = buf.currentBuffer.contents().bindMemory(
      to: WindTunnelBase.self, capacity: controlCount)

    let tunnelCenterY: Float = 0.0
    let tunnelCenterZ: Float = -2.0
    let tunnelHalfLength: Float = 1.5
    let inletHalfY: Float = 0.48
    let inletHalfZ: Float = 0.48

    for i in 0..<linesCount {
      // Random rectangular cross-section position
      let h1 = windRand01(UInt32(i) &* 747_796_405 &+ 277_803_737)
      let h2 = windRand01(UInt32(i) &* 3_266_489_917 &+ 2_246_822_519)
      let dy = (h1 * 2.0 - 1.0) * inletHalfY
      let dz = (h2 * 2.0 - 1.0) * inletHalfZ

      // Spread uniformly along the full tunnel length using a second hash
      let xHash = windRand01(UInt32(i) &* 668_265_263 &+ 0x9e37_79b9)
      let px = -tunnelHalfLength + xHash * (2.0 * tunnelHalfLength)
      let emitPos = SIMD3<Float>(px, tunnelCenterY + dy, tunnelCenterZ + dz)

      let vel = SIMD3<Float>(0.75, 0.0, 0.0)

      let radial = max(abs(dy) / inletHalfY, abs(dz) / inletHalfZ)
      // Time-based color: simulate what windTimeColor would produce
      // at the "virtual emission time" for this x-position
      let virtualTime = (px + tunnelHalfLength) / 0.75  // time to reach px from inlet
      let batch = Int(floor(virtualTime / 0.4))
      let color: SIMD3<Float> =
        (batch % 2 == 0)
        ? SIMD3<Float>(1.0, 0.45, 0.0)  // orange
        : SIMD3<Float>(0.0, 0.7, 1.0)  // cyan

      // All particles active immediately, no stagger
      let age: Float = 0.0

      for j in 0..<controlCountPerLine {
        let index = i * controlCountPerLine + j
        ptr[index] = WindTunnelBase(
          position: emitPos,
          color: color,
          velocity: vel,
          extra: SIMD3<Float>(age, radial, 0))
      }
    }

    buf.copyToNext()
  }

  class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
    let desc = MTLVertexDescriptor()
    var offset = 0

    desc.attributes[0].format = .float3
    desc.attributes[0].offset = offset
    desc.attributes[0].bufferIndex = 0
    offset += MemoryLayout<SIMD3<Float>>.stride

    desc.attributes[1].format = .int
    desc.attributes[1].offset = offset
    desc.attributes[1].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    desc.attributes[2].format = .int
    desc.attributes[2].offset = offset
    desc.attributes[2].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    desc.attributes[3].format = .int
    desc.attributes[3].offset = offset
    desc.attributes[3].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    desc.layouts[0].stride = MemoryLayout<AttractorCellVertex>.stride
    desc.layouts[0].stepRate = 1
    desc.layouts[0].stepFunction = .perVertex
    return desc
  }

  private static func makeRenderPipelineDescriptor(layerRenderer: LayerRenderer) throws
    -> MTLRenderPipelineState
  {
    let pd = Renderer.defaultRenderPipelineDescriptor(layerRenderer: layerRenderer)
    let lib = layerRenderer.device.makeDefaultLibrary()!
    pd.vertexFunction = lib.makeFunction(name: "windTunnelVertexShader")
    pd.fragmentFunction = lib.makeFunction(name: "windTunnelFragmentShader")
    pd.label = "WindTunnelRenderPipeline"
    pd.vertexDescriptor = buildMetalVertexDescriptor()
    return try layerRenderer.device.makeRenderPipelineState(descriptor: pd)
  }

  func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
    TintDrawCommand(
      frameIndex: frame.frameIndex,
      uniforms: uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)],
      verticesCount: verticesCount)
  }

  func computeCommandCommit() {
    guard let computeBuffer,
      let cmd = computeCommandQueue.makeCommandBuffer(),
      let enc = cmd.makeComputeCommandEncoder()
    else { return }

    let elapsed = -Float(viewStartTime.timeIntervalSinceNow)
    let dt = elapsed - frameDelta
    frameDelta = elapsed

    var params = Params(
      time: elapsed,
      elapsed: dt,
      viewerPosition: gestureManager.viewerPosition,
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation,
      totalLines: Int32(linesCount))

    enc.setComputePipelineState(computePipeLine)
    enc.setBuffer(computeBuffer.currentBuffer, offset: 0, index: 0)
    enc.setBuffer(computeBuffer.nextBuffer, offset: 0, index: 1)
    enc.setBytes(&params, length: MemoryLayout<Params>.stride, index: 2)

    let tgs = min(computePipeLine.maxTotalThreadsPerThreadgroup, 256)
    let tpg = MTLSize(width: tgs, height: 1, depth: 1)
    let groups = MTLSize(width: (controlCount + tgs - 1) / tgs, height: 1, depth: 1)
    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tpg)
    enc.endEncoding()
    cmd.commit()
    computeBuffer.swap()
  }

  func encodeDraw(
    _ drawCommand: TintDrawCommand,
    encoder: MTLRenderCommandEncoder,
    drawable: LayerRenderer.Drawable,
    device: MTLDevice,
    tintValue: Float,
    buffer: MTLBuffer,
    indexBuffer: MTLBuffer
  ) {
    encoder.setCullMode(.none)
    encoder.setRenderPipelineState(renderPipelineState)

    var uni = TintUniforms(tintOpacity: tintValue)
    encoder.setVertexBytes(
      &uni, length: MemoryLayout<TintUniforms>.size,
      index: BufferIndex.tintUniforms.rawValue)
    encoder.setVertexBuffer(drawCommand.uniforms, offset: 0, index: BufferIndex.uniforms.rawValue)
    encoder.setVertexBuffer(buffer, offset: 0, index: BufferIndex.meshPositions.rawValue)

    let elapsed = -Float(viewStartTime.timeIntervalSinceNow)
    let dt = elapsed - frameDelta

    var params = Params(
      time: elapsed,
      elapsed: dt,
      viewerPosition: gestureManager.viewerPosition,
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation,
      totalLines: Int32(linesCount))

    let pb = device.makeBuffer(
      bytes: &params, length: MemoryLayout<Params>.stride,
      options: .storageModeShared)!
    encoder.setVertexBuffer(pb, offset: 0, index: BufferIndex.params.rawValue)
    encoder.setVertexBuffer(
      computeBuffer?.currentBuffer, offset: 0,
      index: BufferIndex.base.rawValue)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexesCount,
      indexType: .uint32,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0)
  }

  func updateUniformBuffers(_ drawCommand: TintDrawCommand, drawable: LayerRenderer.Drawable) {
    drawCommand.uniforms.contents()
      .assumingMemoryBound(to: Uniforms.self).pointee = Uniforms(drawable: drawable)
  }

  func onSpatialEvents(events: SpatialEventCollection) {
    for event in events { gestureManager.onSpatialEvent(event: event) }
  }
}
