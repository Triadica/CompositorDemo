/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
A renderer that displays a set of color swatches.
*/

import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

private struct Params {
  var time: Float
}

/// a line during extending tracks last point, new points that are too closer are skipped
/// if line is stable, then all points are in the list `stablePoints`
private struct ExtendingLine {
  var stablePoints: [Point3D] = []
  var lastPoint: Point3D? = .none
  var miniSkip: Double = 0.01

  /// method that combines the stable points and the last point
  func getPoints() -> [Point3D] {
    if let lastP = lastPoint {
      var points = Array(stablePoints)  // Create a copy of stablePoints
      points.append(lastP)
      return points
    } else {
      return stablePoints
    }
  }

  var count: Int {
    if lastPoint != nil {
      return stablePoints.count + 1
    } else {
      return stablePoints.count
    }
  }

  /// if point hat
  mutating func addPoint(_ point: Point3D) {
    if let lastP = lastPoint {
      let distance = lastP.distance(to: point)
      if distance > miniSkip {
        stablePoints.append(lastP)
        lastPoint = point
      }
    } else {
      lastPoint = point
    }
  }

  mutating func stablize() {
    if let lastP = lastPoint {
      stablePoints.append(lastP)
      lastPoint = nil
    }
  }

  mutating func isStable() -> Bool {
    if let lastP = lastPoint {
      return stablePoints.contains { $0.distance(to: lastP) < miniSkip }
    }
    return false
  }
}

private struct LinesManager {
  var lines: [ExtendingLine] = []
  var maxLines: Int = 100
  var currentLine: ExtendingLine = ExtendingLine()

  mutating func addPoint(_ point: Point3D) {
    if lines.count < maxLines {
      currentLine.addPoint(point)
    } else {
      currentLine.stablize()
      lines.append(currentLine)
      currentLine = ExtendingLine()
      currentLine.addPoint(point)
    }
  }

  mutating func finishCurrent() {
    currentLine.stablize()
    lines.append(currentLine)
    currentLine = ExtendingLine()
  }

  func estimateVerticesCount() -> Int {
    var count = 0
    for line in lines {
      count += line.count - 1
    }

    if count == 0 {
      return 6
    }
    return count * 6
  }

  var count: Int {
    lines.count
  }
}

extension Point3D {
  /// turn into SIMD3
  var to_simd3: SIMD3<Float> {
    return SIMD3<Float>(Float(x), Float(y), Float(z))
  }
}

@MainActor
class PolylinesRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the lamp
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  /// tracks buffer size, increased when points getting enormous, should be larger than 0
  var currentVertexBufferSize: Int = 6

  // let computeDevice: MTLDevice
  // var computeBuffer: PingPongBuffer?
  // let computePipeLine: MTLComputePipelineState
  // let computeCommandQueue: MTLCommandQueue

  private var linesManager = LinesManager()

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    // self.computeDevice = MTLCreateSystemDefaultDevice()!
    // let library = computeDevice.makeDefaultLibrary()!
    // let lampsUpdateBase = library.makeFunction(name: "lampsComputeShader")!
    // computePipeLine = try computeDevice.makeComputePipelineState(function: lampsUpdateBase)

    // computeCommandQueue = computeDevice.makeCommandQueue()!

    self.createLampVerticesBuffer(device: layerRenderer.device)
    self.createLampIndexBuffer(device: layerRenderer.device)
    self.createLampComputeBuffer(device: layerRenderer.device)
  }

  /// create and sets the vertices of the lamp
  private func createLampVerticesBuffer(device: MTLDevice) {
    let verticesCount = currentVertexBufferSize
    var bufferLength = MemoryLayout<Vertex>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Lamp vertex buffer"
    var lampVertices: UnsafeMutablePointer<Vertex> {
      vertexBuffer.contents().assumingMemoryBound(to: Vertex.self)
    }

    var pos = 0

    for i in 0..<linesManager.count {
      let line = linesManager.lines[i]
      let color = SIMD3<Float>(1.0, 1.0, 1.0)

      var prevPoint: Point3D = .zero
      for j in 0..<line.count {
        if j == 0 {
          prevPoint = line.stablePoints[j]
          continue
        }
        let ll: Float = 0.01
        let point = line.stablePoints[j]
        let pointSimed3 = point.to_simd3
        let prevPointSimed3 = prevPoint.to_simd3
        let pointUpSimed3 = pointSimed3 + SIMD3<Float>(0, ll, 0)
        let prevPointUpSimed3 = prevPointSimed3 + SIMD3<Float>(0, ll, 0)
        // 6 vertices per rectangle, use (0,1,0) as brush for now
        lampVertices[pos] = Vertex(
          position: pointSimed3,
          color: color,
          seed: Int32(i)
        )
        pos += 1
        lampVertices[pos] = Vertex(
          position: pointUpSimed3,
          color: color,
          seed: Int32(i)
        )
        pos += 1
        lampVertices[pos] = Vertex(
          position: prevPointSimed3,
          color: color,
          seed: Int32(i)
        )
        pos += 1
        lampVertices[pos] = Vertex(
          position: pointUpSimed3,
          color: color,
          seed: Int32(i)
        )
        pos += 1
        lampVertices[pos] = Vertex(
          position: prevPointUpSimed3,
          color: color,
          seed: Int32(i)
        )
        pos += 1
        lampVertices[pos] = Vertex(
          position: prevPointSimed3,
          color: color,
          seed: Int32(i)
        )
        pos += 1
        prevPoint = point
      }
    }
  }

  private func createLampIndexBuffer(device: MTLDevice) {
    let indexesCount = currentVertexBufferSize
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Lamp index buffer"

    let lampIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)
    for i in 0..<indexesCount {
      lampIndices[i] = UInt32(i)
    }

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
      MemoryLayout<Vertex>.stride
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

    let vertexFunction = library.makeFunction(name: "polylinesVertexShader")
    let fragmentFunction = library.makeFunction(name: "polylinesFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "TriangleRenderPipeline"
    pipelineDescriptor.vertexDescriptor = LampsRenderer.buildMetalVertexDescriptor()

    return try layerRenderer.device.makeRenderPipelineState(descriptor: pipelineDescriptor)
  }

  func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
    let verticesCount = currentVertexBufferSize
    return TintDrawCommand(
      frameIndex: frame.frameIndex,
      uniforms: self.uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)],
      verticesCount: verticesCount)
  }

  func resetComputeState() {
    linesManager = LinesManager()
    currentVertexBufferSize = 6
  }

  private func createLampComputeBuffer(device: MTLDevice) {
    // no compute
  }

  func computeCommandCommit() {
    // no compute
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

    // let bufferLength = MemoryLayout<Vertex>.stride * numVertices

    encoder.setVertexBuffer(
      buffer,
      offset: 0,
      index: BufferIndex.meshPositions.rawValue)

    var params_data = Params(time: getTimeSinceStart())

    let params: any MTLBuffer = device.makeBuffer(
      bytes: &params_data,
      length: MemoryLayout<Params>.size,
      options: .storageModeShared
    )!

    encoder.setVertexBuffer(
      params,
      offset: 0,
      index: BufferIndex.params.rawValue)

    // encoder.setVertexBuffer(
    //   computeBuffer?.currentBuffer, offset: 0, index: BufferIndex.base.rawValue)

    let indexesCount = currentVertexBufferSize

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
    // Handle spatial events here
    for event in events {
      // let _chirality = event.chirality
      switch event.phase {
      case .active:
        let position = event.inputDevicePose!.pose3D.position
        linesManager.addPoint(position)
      case .ended:
        linesManager.finishCurrent()
      case .cancelled:
        linesManager.finishCurrent()

      default:
        print("Other event: \(event)")
        break
      }
    }
    let verticesCount = linesManager.estimateVerticesCount()
    if verticesCount + 1000 > currentVertexBufferSize {
      while verticesCount >= currentVertexBufferSize {
        currentVertexBufferSize *= 2
      }
      self.createLampVerticesBuffer(device: vertexBuffer.device)
      self.createLampIndexBuffer(device: vertexBuffer.device)
    }
  }
}
