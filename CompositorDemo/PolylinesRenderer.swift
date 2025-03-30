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
  private var stablePoints: [Point3D] = []
  private var lastPoint: Point3D? = .none
  var miniSkip: Double = 0.01

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

  func getPointAt(_ index: Int) -> Point3D {
    if index < stablePoints.count {
      return stablePoints[index]
    } else if index == stablePoints.count {
      if let lastP = lastPoint {
        return lastP
      } else {
        fatalError("No last point")
      }
    } else {
      fatalError("Index out of bounds")
    }
  }

  mutating func stablize() {
    if let lastP = lastPoint {
      stablePoints.append(lastP)
      lastPoint = nil
    }
  }

  mutating func isStable() -> Bool {
    if let lastP: Point3D = lastPoint {
      return stablePoints.contains { $0.distance(to: lastP) < miniSkip }
    }
    return false
  }
}

private struct LinesManager {
  private var lines: [ExtendingLine] = []
  var maxLines: Int = 100
  private var currentLine: ExtendingLine = ExtendingLine()

  mutating func addPoint(_ point: Point3D) {
    if lines.count < maxLines {
      currentLine.addPoint(point)
    } else {
      print("Max lines reached")
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

    if currentLine.count > 0 {
      count += currentLine.count - 1
    }

    if count == 0 {
      return 1  // at least one vertex
    }
    return count * 6
  }

  var count: Int {
    lines.count + 1
  }

  func getLineAt(_ index: Int) -> ExtendingLine {
    if index < lines.count {
      return lines[index]
    } else if index == lines.count {
      return currentLine
    } else {
      fatalError("Index out of bounds")
    }
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
  /// a buffer to hold the vertices of the polyline
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  /// tracks buffer size, increased when points getting enormous, should be larger than 0
  private var currentVertexBufferSize: Int = 6

  private var linesManager = LinesManager()

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    self.createPolylinesVerticesBuffer(device: layerRenderer.device, count: currentVertexBufferSize)
    self.createPolylinesIndexBuffer(device: layerRenderer.device, count: currentVertexBufferSize)
    // self.createLampComputeBuffer(device: layerRenderer.device)
  }

  /// create and sets the vertices of the polyline
  private func createPolylinesVerticesBuffer(
    device: MTLDevice,
    count: Int = 0
  ) {
    let bufferLength: Int = MemoryLayout<PolylineVertex>.stride * count
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Lamp vertex buffer"

    updateVertexBuffer()
  }

  private func updateVertexBuffer() {
    var polylineVertices: UnsafeMutablePointer<PolylineVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: PolylineVertex.self)
    }

    var pos = 0

    for i in 0..<linesManager.count {
      let line = linesManager.getLineAt(i)
      let color = SIMD3<Float>(1.0, 1.0, 1.0)

      var prevPoint: Point3D = .zero
      for j in 0..<line.count {
        if j == 0 {
          prevPoint = line.getPointAt(0)
          continue
        }
        let ll: Float = 0.001
        let point = line.getPointAt(j)
        let pointSimed3 = point.to_simd3
        let prevPointSimed3 = prevPoint.to_simd3
        let pointUpSimed3 = pointSimed3 + SIMD3<Float>(0, ll, 0)
        let prevPointUpSimed3 = prevPointSimed3 + SIMD3<Float>(0, ll, 0)
        let direction = simd_normalize(pointSimed3 - prevPointSimed3)
        // 6 vertices per rectangle, use (0,1,0) as brush for now
        polylineVertices[pos] = PolylineVertex(
          position: pointSimed3,
          color: color,
          direction: direction,
          seed: Int32(-10)
        )
        pos += 1
        polylineVertices[pos] = PolylineVertex(
          position: pointUpSimed3,
          color: color,
          direction: direction,
          seed: Int32(10)
        )
        pos += 1
        polylineVertices[pos] = PolylineVertex(
          position: prevPointSimed3,
          color: color,
          direction: direction,
          seed: Int32(-10)
        )
        pos += 1
        polylineVertices[pos] = PolylineVertex(
          position: pointUpSimed3,
          color: color,
          direction: direction,
          seed: Int32(10)
        )
        pos += 1
        polylineVertices[pos] = PolylineVertex(
          position: prevPointUpSimed3,
          color: color,
          direction: direction,
          seed: Int32(10)
        )
        pos += 1
        polylineVertices[pos] = PolylineVertex(
          position: prevPointSimed3,
          color: color,
          direction: direction,
          seed: Int32(-10)
        )
        pos += 1
        prevPoint = point
      }
    }
  }

  private func createPolylinesIndexBuffer(device: MTLDevice, count: Int) {
    let indexesCount = count
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
    var offset = 0

    // position
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.position.rawValue].format =
      MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.position.rawValue].offset = offset
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.position.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue
    offset += MemoryLayout<SIMD3<Float>>.stride

    // color
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.color.rawValue].format =
      MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.color.rawValue].offset = offset
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.color.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue
    offset += MemoryLayout<SIMD3<Float>>.stride

    // direction
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.direction.rawValue].format =
      MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.direction.rawValue].offset = offset
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.direction.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue
    offset += MemoryLayout<SIMD3<Float>>.stride

    // seed
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.seed.rawValue].format =
      MTLVertexFormat.int
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.seed.rawValue].offset = offset
    mtlVertexDescriptor.attributes[PolylineVertexAttribute.seed.rawValue].bufferIndex =
      BufferIndex.meshPositions.rawValue
    offset += MemoryLayout<Int32>.stride

    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stride =
      MemoryLayout<PolylineVertex>.stride
    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepRate = 1
    mtlVertexDescriptor.layouts[BufferIndex.meshPositions.rawValue].stepFunction =
      MTLVertexStepFunction.perVertex

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
    pipelineDescriptor.vertexDescriptor = PolylinesRenderer.buildMetalVertexDescriptor()

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
    currentVertexBufferSize = 1
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

    // let bufferLength = MemoryLayout<PolylineVertex>.stride * numVertices

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
    if verticesCount + 100 > currentVertexBufferSize {
      while verticesCount + 100 >= currentVertexBufferSize {
        currentVertexBufferSize = currentVertexBufferSize * 2
      }
      self.createPolylinesVerticesBuffer(
        device: vertexBuffer.device, count: currentVertexBufferSize)
      self.createPolylinesIndexBuffer(device: vertexBuffer.device, count: currentVertexBufferSize)
    }
    updateVertexBuffer()
  }
}
