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

private struct SparkVertex {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var direction: SIMD3<Float>
  var seed: Float
}

private struct SparkLine {
  var from: SIMD3<Float> = .zero
  var to: SIMD3<Float> = .zero
  var birthTime: Float = 0
}

let sparksLimit = 4000

/// it has a limit of 1000 lines, if succeeds, it will overwrite from start, tracked with cursorIdx
private struct SparksCollection {
  var coll: [SparkLine] = []
  var cursorIdx: Int = 0

  mutating func add(_ from: SIMD3<Float>, _ to: SIMD3<Float>, time: Float) {
    let l = SparkLine(from: from, to: to, birthTime: time)
    if coll.count < sparksLimit {
      coll.append(l)
    } else {
      coll[cursorIdx] = l
      cursorIdx += 1
      if cursorIdx >= coll.count {
        cursorIdx = 0
      }
    }
  }

  func estimateVerticesCount() -> Int {
    return coll.count * 6
  }

  func getLineAt(_ idx: Int) -> SparkLine {
    return coll[idx]
  }

}

@MainActor
class DragSparksRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the polyline
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  /// tracks buffer size, increased when points getting enormous, should be larger than 0
  private var currentVertexBufferSize: Int = 6

  private var linesManager = SparksCollection()

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    self.createPolylinesVerticesBuffer(device: layerRenderer.device, count: currentVertexBufferSize)
    self.createPolylinesIndexBuffer(device: layerRenderer.device, count: currentVertexBufferSize)
    // self.createSparksComputeBuffer(device: layerRenderer.device)
  }

  /// create and sets the vertices of the polyline
  private func createPolylinesVerticesBuffer(
    device: MTLDevice,
    count: Int = 0
  ) {
    let bufferLength: Int = MemoryLayout<SparkVertex>.stride * count
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Sparks vertex buffer"

    updateVertexBuffer()
  }

  private func updateVertexBuffer() {
    var polylineVertices: UnsafeMutablePointer<SparkVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: SparkVertex.self)
    }

    var pos = 0
    let width: Float = 0.01

    for i in 0..<linesManager.coll.count {
      let line = linesManager.getLineAt(i)
      // Generate a random color for each line
      let color = SIMD3<Float>(
        Float.random(in: 0.0...1.0),
        Float.random(in: 0.0...1.0),
        Float.random(in: 0.0...1.0)
      )

      let direction = line.to - line.from

      let spark1 = SparkVertex(
        position: line.from,
        color: color,
        direction: direction,
        seed: Float(-width)
      )
      let spark2 = SparkVertex(
        position: line.from,
        color: color,
        direction: direction,
        seed: Float(width)
      )
      let spark3 = SparkVertex(
        position: line.to,
        color: color,
        direction: direction,
        seed: Float(-width)
      )
      let spark4 = SparkVertex(
        position: line.to,
        color: color,
        direction: direction,
        seed: Float(width)
      )

      // 2 triangles per line
      polylineVertices[pos] = spark1
      pos += 1
      polylineVertices[pos] = spark2
      pos += 1
      polylineVertices[pos] = spark3
      pos += 1
      polylineVertices[pos] = spark1
      pos += 1
      polylineVertices[pos] = spark3
      pos += 1
      polylineVertices[pos] = spark4
      pos += 1

    }

  }

  private func createPolylinesIndexBuffer(device: MTLDevice, count: Int) {
    let indexesCount = count
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "sparks index buffer"

    let sparksIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)
    for i in 0..<indexesCount {
      sparksIndices[i] = UInt32(i)
    }

  }

  class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
    // Create a vertex descriptor specifying how Metal lays out vertices for input into the render pipeline.

    let mtlVertexDescriptor = MTLVertexDescriptor()
    var offset = 0
    var idx: Int = 0

    // position
    mtlVertexDescriptor.attributes[idx].format = MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[idx].offset = offset
    mtlVertexDescriptor.attributes[idx].bufferIndex = 0
    offset += MemoryLayout<SIMD3<Float>>.stride
    idx += 1

    // color
    mtlVertexDescriptor.attributes[idx].format = MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[idx].offset = offset
    mtlVertexDescriptor.attributes[idx].bufferIndex = 0
    offset += MemoryLayout<SIMD3<Float>>.stride
    idx += 1

    // direction
    mtlVertexDescriptor.attributes[idx].format = MTLVertexFormat.float3
    mtlVertexDescriptor.attributes[idx].offset = offset
    mtlVertexDescriptor.attributes[idx].bufferIndex = 0
    offset += MemoryLayout<SIMD3<Float>>.stride
    idx += 1

    // seed
    mtlVertexDescriptor.attributes[idx].format = MTLVertexFormat.float
    mtlVertexDescriptor.attributes[idx].offset = offset
    mtlVertexDescriptor.attributes[idx].bufferIndex = 0
    offset += MemoryLayout<Float>.stride

    mtlVertexDescriptor.layouts[0].stride = MemoryLayout<SparkVertex>.stride
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

    let vertexFunction = library.makeFunction(name: "dragSparksVertexShader")
    let fragmentFunction = library.makeFunction(name: "dragSparksFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "DragSparksRenderPipeline"
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
    // createPolylinesVerticesBuffer(device: vertexBuffer.device, count: currentVertexBufferSize)
    // createPolylinesIndexBuffer(device: vertexBuffer.device, count: currentVertexBufferSize)
  }

  private func createSparksComputeBuffer(device: MTLDevice) {
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

    // let bufferLength = MemoryLayout<SparkVertex>.stride * numVertices

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
        let position = event.inputDevicePose!.pose3D.position.to_simd3

        // Add 10 lines for each active event
        for _ in 0..<10 {
          let offset = randomPosition(x: 0.1, y: 0.1, z: 0.1)
          let nextPosition = position + offset
          // print("  chilarity: \(event.chirality!), position: \(position)")
          linesManager.add(position, nextPosition, time: getTimeSinceStart())
        }
      case .ended:
        break
      case .cancelled:
        break
      default:
        print("Other event: \(event)")
        break
      }
    }
    let verticesCount = linesManager.estimateVerticesCount()
    let vertexesLimit = sparksLimit * 6
    if currentVertexBufferSize < vertexesLimit {
      if verticesCount + 200 > currentVertexBufferSize {
        while verticesCount + 200 >= currentVertexBufferSize {
          currentVertexBufferSize = currentVertexBufferSize * 2
        }
        if currentVertexBufferSize > vertexesLimit {
          currentVertexBufferSize = vertexesLimit
        }
        self.createPolylinesVerticesBuffer(
          device: vertexBuffer.device, count: currentVertexBufferSize)
        self.createPolylinesIndexBuffer(device: vertexBuffer.device, count: currentVertexBufferSize)
      }
    }
    updateVertexBuffer()
  }
}
