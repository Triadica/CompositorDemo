/*
 See the LICENSE.txt file for this sample's licensing information.

 Abstract:
 A renderer that displays a set of color swatches.
 */

import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

private let maxFramesInFlight = 3

/// how many lines for this attractor
private let linesCount: Int = 90000
/// how many rectangles in a line
private let lineGroupSize: Int = 2
/// 1 for leading point, others are following points
private var controlCountPerLine: Int {
  lineGroupSize + 1
}
/// all control points in the scene
private var controlCount: Int {
  linesCount * controlCountPerLine
}

private let verticesCount = controlCount * 6

/// rectangle indexes per rectangle
private let indexesCount: Int = controlCount * 6

private struct SpreadInBallBase {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var velocity: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
}

private struct SpreadInBallParams {
  var time: Float
  var groupSize: Int32 = Int32(lineGroupSize)
  var viewerPosition: SIMD3<Float>
  var viewerScale: Float
  var viewerRotation: Float = 0.0
  var _padding: SIMD2<Float> = SIMD2<Float>(0, 0)  // Pad to 48 bytes, remove if shader expects 36 bytes
}

@MainActor
class SpreadInBallRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the lamp
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  let computeDevice: MTLDevice
  var computeBuffer: PingPongBuffer?
  let computePipeLine: MTLComputePipelineState
  let computeCommandQueue: MTLCommandQueue

  var gestureManager: GestureManager = GestureManager()

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    self.computeDevice = MTLCreateSystemDefaultDevice()!
    let library = computeDevice.makeDefaultLibrary()!
    let attractorUpdateBase = library.makeFunction(name: "spreadInBallComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: attractorUpdateBase)

    computeCommandQueue = computeDevice.makeCommandQueue()!

    self.createAttractorVerticesBuffer(device: layerRenderer.device)
    self.createAttractorIndexBuffer(device: layerRenderer.device)
    self.createAttractorComputeBuffer(device: layerRenderer.device)
  }

  /// create and sets the vertices of the lamp
  private func createAttractorVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<AttractorCellVertex>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Attractor vertex buffer"
    var attractorVertices: UnsafeMutablePointer<AttractorCellVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: AttractorCellVertex.self)
    }

    for i in 0..<linesCount {
      let baseIndex = i * lineGroupSize * 6

      for j in 0..<lineGroupSize {
        let index = baseIndex + j * 6
        // set 6 vertices for each cell
        // 1st vertex
        attractorVertices[index] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 0)
        // 2nd vertex
        attractorVertices[index + 1] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 1)
        // 3rd vertex
        attractorVertices[index + 2] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 2)
        // 4th vertex
        attractorVertices[index + 3] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 1)
        // 5th vertex
        attractorVertices[index + 4] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 2)
        // 6th vertex
        attractorVertices[index + 5] = AttractorCellVertex(
          position: SIMD3<Float>(0, 0, 0),
          lineNumber: Int32(i),
          groupNumber: Int32(j),
          cellSide: 3)
      }

    }
  }

  func resetComputeState() {
    self.createAttractorComputeBuffer(device: computeDevice)
  }

  private func createAttractorIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Lamp index buffer"

    let attractorIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)
    for i in 0..<indexesCount {
      attractorIndices[i] = UInt32(i)
    }

  }

  // Generate multiple small spheres positioned inside the target sphere
  private func generateSmallSpheres(
    numSpheres: Int, particlesPerSphere: Int, targetCenter: SIMD3<Float>, targetRadius: Float
  ) -> [(position: SIMD3<Float>, velocity: SIMD3<Float>, color: SIMD3<Float>)] {
    var particles: [(position: SIMD3<Float>, velocity: SIMD3<Float>, color: SIMD3<Float>)] = []

    // Create small spheres inside the target sphere (no random positioning)
    for sphereIndex in 0..<numSpheres {
      // Position each small sphere uniformly in a ring inside the target sphere
      let angle = Float(sphereIndex) * 2.0 * Float.pi / Float(numSpheres)
      let distance = Float(0.4)  // Fixed ring radius
      let height = Float(0.0)  // All spheres at same height level

      let sphereCenter =
        targetCenter
        + SIMD3<Float>(
          cos(angle) * distance,
          height,
          sin(angle) * distance
        )

      let smallSphereRadius = 0.05  // Fixed radius instead of random

      // Generate fixed velocity parameters for this small sphere (reduced for internal movement)
      let baseSpeed = Float(0.02)  // Fixed base speed
      let randomDirection = Float(sphereIndex) * 0.1  // Deterministic variation

      // Generate fixed direction for this small sphere (no random offset)
      let sphereDirection = SIMD3<Float>(
        cos(angle + Float.pi * 0.5),
        0.2,  // Slight upward component
        sin(angle + Float.pi * 0.5)
      )
      let normalizedDirection = normalize(sphereDirection)

      // Generate particles within this small sphere
      for particleIndex in 0..<particlesPerSphere {
        // Use global fibonacci grid function for uniform sphere distribution
        let unitPosition = fibonacciGrid(n: Float(particleIndex), total: Float(particlesPerSphere))
        let particlePosition = sphereCenter + unitPosition * Float(smallSphereRadius)

        // Calculate velocity for internal movement (no target attraction)
        let expansionDirection = normalize(particlePosition - sphereCenter)

        // Apply fixed direction for this small sphere with slight expansion
        let velocity = normalizedDirection * baseSpeed + expansionDirection * (baseSpeed * 0.1)

        // Color based on sphere index
        let hue = Float(sphereIndex) / Float(numSpheres)
        let color = SIMD3<Float>(
          0.5 + 0.5 * cos(hue * 2.0 * Float.pi),
          0.5 + 0.5 * cos(hue * 2.0 * Float.pi + 2.0),
          0.5 + 0.5 * cos(hue * 2.0 * Float.pi + 4.0)
        )

        particles.append((position: particlePosition, velocity: velocity, color: color))
      }
    }

    return particles
  }

  private func createAttractorComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<SpreadInBallBase>.stride * controlCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Attractor compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let attractorBase = contents.bindMemory(to: SpreadInBallBase.self, capacity: controlCount)

    let targetCenter = SIMD3<Float>(0.0, 0.0, -1.0)
    let targetRadius: Float = 1.6

    // Generate multiple small spheres with particles
    let numSpheres = 16
    let particlesPerSphere = linesCount / numSpheres
    let particles = generateSmallSpheres(
      numSpheres: numSpheres,
      particlesPerSphere: particlesPerSphere,
      targetCenter: targetCenter,
      targetRadius: targetRadius
    )

    // Fill remaining particles if any
    var particleIndex = 0
    for i in 0..<linesCount {
      let particle = particles[min(particleIndex, particles.count - 1)]
      particleIndex += 1

      for j in 0..<controlCountPerLine {
        let index = i * controlCountPerLine + j
        attractorBase[index] = SpreadInBallBase(
          position: particle.position,
          color: particle.color,
          velocity: particle.velocity
        )
      }
    }

    computeBuffer.copyToNext()
  }

  class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
    // Create a vertex descriptor specifying how Metal lays out vertices for input into the render pipeline.

    let mtlVertexDescriptor = MTLVertexDescriptor()
    var offset: Int = 0

    mtlVertexDescriptor.attributes[0].format = MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[0].offset = offset
    mtlVertexDescriptor.attributes[0].bufferIndex = 0
    offset += MemoryLayout<SIMD3<Float>>.stride

    mtlVertexDescriptor.attributes[1].format = MTLVertexFormat.int
    mtlVertexDescriptor.attributes[1].offset = offset
    mtlVertexDescriptor.attributes[1].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    mtlVertexDescriptor.attributes[2].format = MTLVertexFormat.int
    mtlVertexDescriptor.attributes[2].offset = offset
    mtlVertexDescriptor.attributes[2].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    mtlVertexDescriptor.attributes[3].format = MTLVertexFormat.int
    mtlVertexDescriptor.attributes[3].offset = offset
    mtlVertexDescriptor.attributes[3].bufferIndex = 0
    offset += MemoryLayout<Int32>.stride

    // layout is special
    mtlVertexDescriptor.layouts[0].stride = MemoryLayout<AttractorCellVertex>.stride
    mtlVertexDescriptor.layouts[0].stepRate = 1
    mtlVertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunction.perVertex

    return mtlVertexDescriptor
  }

  private static func makeRenderPipelineDescriptor(layerRenderer: LayerRenderer) throws
    -> MTLRenderPipelineState
  {
    let pipelineDescriptor = Renderer.defaultRenderPipelineDescriptor(
      layerRenderer: layerRenderer)

    let library = layerRenderer.device.makeDefaultLibrary()!

    let vertexFunction = library.makeFunction(name: "spreadInBallVertexShader")
    let fragmentFunction = library.makeFunction(name: "spreadInBallFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "TriangleRenderPipeline"
    pipelineDescriptor.vertexDescriptor = self.buildMetalVertexDescriptor()

    return try layerRenderer.device.makeRenderPipelineState(descriptor: pipelineDescriptor)
  }

  func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
    return TintDrawCommand(
      frameIndex: frame.frameIndex,
      uniforms: self.uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)],
      verticesCount: verticesCount)
  }

  func computeCommandCommit() {
    guard let computeBuffer: PingPongBuffer = computeBuffer,
      let commandBuffer = computeCommandQueue.makeCommandBuffer(),
      let computeEncoder = commandBuffer.makeComputeCommandEncoder()
    else {
      print("Failed to create compute command buffer")
      return
    }

    computeEncoder.setComputePipelineState(computePipeLine)
    computeEncoder.setBuffer(computeBuffer.currentBuffer, offset: 0, index: 0)
    computeEncoder.setBuffer(computeBuffer.nextBuffer, offset: 0, index: 1)

    let delta = -Float(viewStartTime.timeIntervalSinceNow)
    let dt = delta - frameDelta
    frameDelta = delta

    var params = SpreadInBallParams(
      time: dt, viewerPosition: self.gestureManager.viewerPosition,
      viewerScale: self.gestureManager.viewerScale,
      viewerRotation: self.gestureManager.viewerRotation)
    computeEncoder.setBytes(&params, length: MemoryLayout<SpreadInBallParams>.size, index: 2)
    let threadGroupSize = min(computePipeLine.maxTotalThreadsPerThreadgroup, 256)
    let threadsPerThreadgroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
    let threadGroups = MTLSize(
      width: (controlCount + threadGroupSize - 1) / threadGroupSize,
      height: 1,
      depth: 1
    )
    computeEncoder.dispatchThreadgroups(
      threadGroups, threadsPerThreadgroup: threadsPerThreadgroup)
    computeEncoder.endEncoding()
    commandBuffer.commit()

    computeBuffer.swap()
  }

  // in seconds
  func getTimeSinceStart() -> Float {
    let time = DispatchTime.now().uptimeNanoseconds
    let timeSinceStart = Float(time) / 1_000_000_000
    return timeSinceStart
  }

  private var viewStartTime: Date = Date()
  private var frameDelta: Float = 0.0

  func encodeDraw(
    _ drawCommand: TintDrawCommand,
    encoder: MTLRenderCommandEncoder,
    drawable: LayerRenderer.Drawable,
    device: MTLDevice, tintValue: Float,
    buffer: MTLBuffer,
    indexBuffer: MTLBuffer
  ) {
    encoder.setCullMode(.none)

    encoder.setRenderPipelineState(renderPipelineState)

    var demoUniform: TintUniforms = TintUniforms(tintOpacity: tintValue)
    encoder.setVertexBytes(
      &demoUniform,
      length: MemoryLayout<TintUniforms>.size,
      index: BufferIndex.tintUniforms.rawValue)

    encoder.setVertexBuffer(
      drawCommand.uniforms,
      offset: 0,
      index: BufferIndex.uniforms.rawValue)

    encoder.setVertexBuffer(
      buffer,
      offset: 0,
      index: BufferIndex.meshPositions.rawValue)

    var params_data = SpreadInBallParams(
      time: getTimeSinceStart(),
      viewerPosition: self.gestureManager.viewerPosition,
      viewerScale: self.gestureManager.viewerScale,
      viewerRotation: self.gestureManager.viewerRotation)

    let params: any MTLBuffer = device.makeBuffer(
      bytes: &params_data,
      length: MemoryLayout<SpreadInBallParams>.size,
      options: .storageModeShared
    )!

    encoder.setVertexBuffer(
      params,
      offset: 0,
      index: BufferIndex.params.rawValue)

    encoder.setVertexBuffer(
      computeBuffer?.currentBuffer, offset: 0, index: BufferIndex.base.rawValue)

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexesCount,
      indexType: .uint32,
      indexBuffer: indexBuffer,
      indexBufferOffset: 0
    )
  }

  func updateUniformBuffers(
    _ drawCommand: TintDrawCommand,
    drawable: LayerRenderer.Drawable
  ) {
    drawCommand.uniforms.contents().assumingMemoryBound(to: Uniforms.self).pointee = Uniforms(
      drawable: drawable)
  }

  /// track the position pinch started, following pinches define the velocity of moving, to update self.viewerPosition .
  /// other other chirality events are used for scaling the entity
  func onSpatialEvents(events: SpatialEventCollection) {
    for event in events {
      gestureManager.onSpatialEvent(event: event)
    }
  }
}
