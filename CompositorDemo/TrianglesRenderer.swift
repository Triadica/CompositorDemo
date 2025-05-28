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

extension Point3D {
  /// turn into SIMD3
  fileprivate var to_simd3: SIMD3<Float> {
    return SIMD3<Float>(Float(x), Float(y), Float(z))
  }
}

extension LinesManager {

  /// build triangles from lines with each 3 sibsequent points
  fileprivate func estimateVerticesCount() -> Int {

    var count = 0
    for i in 0..<self.count {
      let line = self.getLineAt(i)
      if line.count > 2 {
        count += line.count - 2
      }
    }

    if count == 0 {
      return 1  // at least one vertex
    }
    let verticesPerTriangle = 6
    return count * verticesPerTriangle
  }
}

@MainActor
class TrianglesRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the polyline
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  /// tracks buffer size, increased when points getting enormous, should be larger than 0
  private var currentVertexBufferSize: Int = 6

  private var linesManager = LinesManager(miniSkip: 0.02)

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
      // Generate a random color for each line
      let color = line.color

      var prevPoint: Point3D = .zero
      var beforePrevPoint: Point3D = .zero
      for j in 0..<line.count {
        if j == 0 {
          beforePrevPoint = line.getPointAt(0)
          continue
        }
        if j == 1 {
          prevPoint = line.getPointAt(0)
          continue
        }
        let point = line.getPointAt(j)
        let pointSimed3 = point.to_simd3
        let prevPointSimed3 = prevPoint.to_simd3
        let beforePrevPointSimed3 = beforePrevPoint.to_simd3

        let direction = simd_normalize(pointSimed3 - prevPointSimed3)

        let width: Float = 6
        // 6 vertices per rectangle, use (0,1,0) as brush for now
        polylineVertices[pos] = PolylineVertex(
          position: beforePrevPointSimed3,
          color: color,
          direction: direction,
          seed: Int32(-width)
        )
        pos += 1
        polylineVertices[pos] = PolylineVertex(
          position: prevPointSimed3,
          color: color,
          direction: direction,
          seed: Int32(width)
        )
        pos += 1
        polylineVertices[pos] = PolylineVertex(
          position: pointSimed3,
          color: color,
          direction: direction,
          seed: Int32(-width)
        )
        pos += 1
        beforePrevPoint = prevPoint
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

    let vertexFunction = library.makeFunction(name: "trianglesVertexShader")
    let fragmentFunction = library.makeFunction(name: "trianglesFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "TriangleRenderPipeline"
    pipelineDescriptor.vertexDescriptor = self.buildMetalVertexDescriptor()

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
    linesManager.removeLastLine()

    createPolylinesVerticesBuffer(device: vertexBuffer.device, count: currentVertexBufferSize)
    createPolylinesIndexBuffer(device: vertexBuffer.device, count: currentVertexBufferSize)
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

    // print("onSpatialEvents \(events.count)")

    let sortedEvents = events.sorted {
      guard let chirality1 = $0.chirality, let chirality2 = $1.chirality else {
        return false
      }
      return chirality1.hashValue <= chirality2.hashValue
    }

    // Handle spatial events here
    for event in sortedEvents {
      // let _chirality = event.chirality
      switch event.phase {
      case .active:
        let position = event.inputDevicePose!.pose3D.position
        // print("  chilarity: \(event.chirality!), position: \(position)")
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
    if verticesCount + 200 > currentVertexBufferSize {
      while verticesCount + 200 >= currentVertexBufferSize {
        currentVertexBufferSize = currentVertexBufferSize * 2
      }
      self.createPolylinesVerticesBuffer(
        device: vertexBuffer.device, count: currentVertexBufferSize)
      self.createPolylinesIndexBuffer(device: vertexBuffer.device, count: currentVertexBufferSize)
    }
    updateVertexBuffer()
  }
}
