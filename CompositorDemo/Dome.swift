import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

private let maxFramesInFlight = 3

// Dome parameters
private let sphereRadius: Float = 5.0  // Dome radius 5m
private let pointCount: Int = 120  // Number of points on the dome

// Dome mesh parameters - further optimized density for better performance
private let sphereSegments: Int = 24  // Dome longitude segments (reduced from 32 to 24)
private let sphereRings: Int = 12  // Dome latitude segments (reduced from 16 to 12)
private let verticesCount = (sphereRings + 1) * (sphereSegments + 1)
private let indexesCount = sphereRings * sphereSegments * 6

// Optimized memory layout: consistent with Metal shader
private struct SpherePoint {
  var position: SIMD3<Float>  // Point position on the dome (12 bytes)
  var angularSpeed: Float  // Angular speed (radians/second) (4 bytes) - total 16 bytes
  var rotationAxis: SIMD3<Float>  // Rotation axis (line direction through sphere center) (12 bytes)
  var pointId: Float  // Point ID (4 bytes) - total 16 bytes
}

private struct Params {
  var viewerPosition: SIMD3<Float>
  var time: Float
  var viewerScale: Float
  var viewerRotation: Float = .zero
  var _padding: SIMD2<Float> = .zero  // required for 48 bytes alignment
}

@MainActor
class DomeRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  private var paramsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the lamp
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  let computeDevice: MTLDevice
  var computeBuffer: PingPongBuffer?
  let computePipeLine: MTLComputePipelineState
  let computeCommandQueue: MTLCommandQueue

  var gestureManager: GestureManager = GestureManager(onScene: true)

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<Uniforms>.uniformStride)!
    }

    paramsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(
        length: MemoryLayout<Params>.stride, options: .storageModeShared)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    self.computeDevice = MTLCreateSystemDefaultDevice()!
    let library = computeDevice.makeDefaultLibrary()!
    let cellUpdateBase = library.makeFunction(name: "domeComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: cellUpdateBase)

    computeCommandQueue = computeDevice.makeCommandQueue()!

    self.createDomeVerticesBuffer(device: layerRenderer.device)
    self.createDomeIndexBuffer(device: layerRenderer.device)
    self.createDomeComputeBuffer(device: layerRenderer.device)
  }

  /// Create dome vertex buffer
  private func createDomeVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<VertexWithSeed>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Sphere vertex buffer"

    let vertices = vertexBuffer.contents().assumingMemoryBound(to: VertexWithSeed.self)

    var vertexIndex = 0

    // Generate dome vertices
    for ring in 0...sphereRings {
      let phi = Float(ring) * Float.pi / Float(sphereRings)  // Latitude angle
      let y = cos(phi) * sphereRadius
      let ringRadius = sin(phi) * sphereRadius

      for segment in 0...sphereSegments {
        let theta = Float(segment) * 2.0 * Float.pi / Float(sphereSegments)  // Longitude angle
        let x = cos(theta) * ringRadius
        let z = sin(theta) * ringRadius

        let position = SIMD3<Float>(x, y, z)
        let color = SIMD3<Float>(0.2, 0.2, 0.2)  // Default gray

        vertices[vertexIndex] = VertexWithSeed(
          position: position,
          color: color,
          seed: Int32(vertexIndex)
        )
        vertexIndex += 1

      }
    }
  }

  func resetComputeState() {
    self.createDomeComputeBuffer(device: self.computeDevice)
  }

  /// Create dome index buffer
  private func createDomeIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Sphere index buffer"

    let indices = indexBuffer.contents().assumingMemoryBound(to: UInt32.self)
    var indexOffset = 0

    // Generate dome triangle indices
    for ring in 0..<sphereRings {
      for segment in 0..<sphereSegments {
        let current = UInt32(ring * (sphereSegments + 1) + segment)
        let next = UInt32(ring * (sphereSegments + 1) + (segment + 1))
        let currentNext = UInt32((ring + 1) * (sphereSegments + 1) + segment)
        let nextNext = UInt32((ring + 1) * (sphereSegments + 1) + (segment + 1))

        // First triangle
        indices[indexOffset] = current
        indices[indexOffset + 1] = currentNext
        indices[indexOffset + 2] = next

        // Second triangle
        indices[indexOffset + 3] = next
        indices[indexOffset + 4] = currentNext
        indices[indexOffset + 5] = nextNext

        indexOffset += 6
      }
    }
  }

  /// Create point data on the dome
  private func createDomeComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<SpherePoint>.stride * pointCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Sphere points compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let spherePoints = contents.bindMemory(to: SpherePoint.self, capacity: pointCount)

    for i in 0..<pointCount {
      // Randomly distribute points on the dome
      let phi = Float.random(in: 0...Float.pi)  // Latitude angle
      let theta = Float.random(in: 0...(2 * Float.pi))  // Longitude angle

      let x = sin(phi) * cos(theta) * sphereRadius
      let y = cos(phi) * sphereRadius
      let z = sin(phi) * sin(theta) * sphereRadius

      let position = SIMD3<Float>(x, y, z)

      // Generate random rotation axis (line direction through sphere center)
      let axisTheta = Float.random(in: 0...(2 * Float.pi))
      let axisPhi = Float.random(in: 0...Float.pi)
      let rotationAxis = normalize(
        SIMD3<Float>(
          sin(axisPhi) * cos(axisTheta),
          cos(axisPhi),
          sin(axisPhi) * sin(axisTheta)
        ))

      // Set angular speed (radians/second), range from 0.1 to 1.0 radians/second
      let angularSpeed = Float.random(in: 0.1...1.0)

      spherePoints[i] = SpherePoint(
        position: position,
        angularSpeed: angularSpeed,
        rotationAxis: rotationAxis,
        pointId: Float(i)
      )
    }

    computeBuffer.copyToNext()
  }

  class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
    // Create a vertex descriptor specifying how Metal lays out vertices for input into the render pipeline.

    let mtlVertexDescriptor = MTLVertexDescriptor()

    mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].format =
      MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].offset = 0
    mtlVertexDescriptor.attributes[VertexAttribute.position.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue

    let offset = MemoryLayout<SIMD3<Float>>.stride
    mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].format =
      MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].offset = offset
    mtlVertexDescriptor.attributes[VertexAttribute.color.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue

    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride =
      MemoryLayout<VertexWithSeed>.stride
    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction =
      MTLVertexStepFunction.perVertex
    // add params for seed value
    let nextOffset = offset + MemoryLayout<SIMD3<Float>>.stride
    mtlVertexDescriptor.attributes[VertexAttribute.seed.rawValue].format =
      MTLVertexFormat.int
    mtlVertexDescriptor.attributes[VertexAttribute.seed.rawValue].offset = nextOffset
    mtlVertexDescriptor.attributes[VertexAttribute.seed.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue

    return mtlVertexDescriptor
  }

  private static func makeRenderPipelineDescriptor(layerRenderer: LayerRenderer) throws
    -> MTLRenderPipelineState
  {
    let pipelineDescriptor = Renderer.defaultRenderPipelineDescriptor(
      layerRenderer: layerRenderer)

    let library = layerRenderer.device.makeDefaultLibrary()!

    let vertexFunction = library.makeFunction(name: "domeVertexShader")
    let fragmentFunction = library.makeFunction(name: "domeFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "DomeRenderPipeline"
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
    let currentTime = -Float(viewStartTime.timeIntervalSinceNow)
    
    // Optimization: limit compute shader execution frequency
    if currentTime - lastComputeTime < computeInterval {
      return  // Skip computation for this frame
    }
    
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

    let delta = currentTime
    frameDelta = delta
    lastComputeTime = currentTime

    var params = Params(
      viewerPosition: gestureManager.viewerPosition,
      time: computeInterval,  // Use fixed time step
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation
    )
    computeEncoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 2)
    
    // Optimization: use smaller thread group size
    let threadGroupSize = min(computePipeLine.maxTotalThreadsPerThreadgroup, 64)
    let threadsPerThreadgroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
    let threadGroups = MTLSize(
      width: (pointCount + threadGroupSize - 1) / threadGroupSize,
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
  private var lastComputeTime: Float = 0.0
  private let computeInterval: Float = 1.0/60.0  // Limit compute shader to 60FPS

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

    let currentParamsBuffer = paramsBuffer[Int(drawCommand.frameIndex % UInt64(Renderer.maxFramesInFlight))]

    var params_data = Params(
      viewerPosition: gestureManager.viewerPosition,
      time: 0.016,  // Use fixed time step to avoid flickering
      viewerScale: gestureManager.viewerScale,
      viewerRotation: gestureManager.viewerRotation
    )

    currentParamsBuffer.contents().copyMemory(
      from: &params_data, byteCount: MemoryLayout<Params>.stride)

    encoder.setVertexBuffer(
      currentParamsBuffer,
      offset: 0,
      index: BufferIndex.params.rawValue)

    encoder.setVertexBuffer(
      computeBuffer?.currentBuffer, offset: 0, index: BufferIndex.base.rawValue)

    // Set buffers for fragment shader
    encoder.setFragmentBuffer(
      currentParamsBuffer,
      offset: 0,
      index: BufferIndex.params.rawValue)

    encoder.setFragmentBuffer(
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

  func onSpatialEvents(events: SpatialEventCollection) {
    for event in events {
      gestureManager.onSpatialEvent(event: event)
    }
  }
}
