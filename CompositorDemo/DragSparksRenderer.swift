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
  var brushWidth: Float
  var brushValue: Float = 0
  var birthTime: Float = 0
}

private struct SparkLine {
  var from: SIMD3<Float> = .zero
  var to: SIMD3<Float> = .zero
  var birthTime: Float = 0
  var color: SIMD3<Float> = .zero
}

let sparksLimit = 8000

func randomFireworksColor() -> SIMD3<Float> {
  let colorType = Float.random(in: 0...1)
  let color: SIMD3<Float>

  if colorType < 0.85 {
    // Orange-yellow colors (85% chance)
    color = SIMD3<Float>(
      Float.random(in: 0.8...1.0),  // Red: high
      Float.random(in: 0.4...0.7),  // Green: medium
      Float.random(in: 0.0...0.3)  // Blue: low
    )
  } else if colorType < 0.92 {
    // purple accents (7% chance)
    color = SIMD3<Float>(
      Float.random(in: 0.5...0.8),  // Red: medium-high
      Float.random(in: 0.0...0.3),  // Green: low
      Float.random(in: 0.5...1.0)  // Blue: medium-high
    )
  } else {
    // Red accents (8% chance)
    color = SIMD3<Float>(
      Float.random(in: 0.8...1.0),  // Red: high
      Float.random(in: 0.0...0.3),  // Green: low
      Float.random(in: 0.0...0.3)  // Blue: low
    )
  }

  return color
}

/// random function, values near from have higher probability
func randBaseFromTo(_ from: Float, _ to: Float) -> Float {
  let range = to - from
  let randomValue = Float.random(in: 0...1)
  let adjustedValue = pow(randomValue, 8)  // Adjust the exponent for more or less clustering
  return from + adjustedValue * range
}

/// it has a limit of 4000 lines, if succeeds, it will overwrite from start, tracked with cursorIdx
private struct SparksCollection {
  var coll: [SparkLine] = []
  var cursorIdx: Int = 0

  mutating func add(_ from: SIMD3<Float>, _ to: SIMD3<Float>, time: Float) {
    // Create predominantly orange-yellow colors with occasional blue/green accents
    let color = randomFireworksColor()
    let l = SparkLine(from: from, to: to, birthTime: time, color: color)
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
    let width: Float = 0.006

    for i in 0..<linesManager.coll.count {
      let line = linesManager.getLineAt(i)
      // Generate a random color for each line

      let direction = line.to - line.from

      let spark1 = SparkVertex(
        position: line.from,
        color: line.color,
        direction: direction,
        brushWidth: Float(-width),
        brushValue: 0,
        birthTime: line.birthTime
      )
      let spark2 = SparkVertex(
        position: line.from,
        color: line.color,
        direction: direction,
        brushWidth: Float(width),
        brushValue: 0,
        birthTime: line.birthTime
      )
      let spark3 = SparkVertex(
        position: line.to,
        color: line.color,
        direction: direction,
        brushWidth: Float(-width),
        brushValue: 1,
        birthTime: line.birthTime
      )
      let spark4 = SparkVertex(
        position: line.to,
        color: line.color,
        direction: direction,
        brushWidth: Float(width),
        brushValue: 1,
        birthTime: line.birthTime
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

    // brush width
    mtlVertexDescriptor.attributes[idx].format = MTLVertexFormat.float
    mtlVertexDescriptor.attributes[idx].offset = offset
    mtlVertexDescriptor.attributes[idx].bufferIndex = 0
    offset += MemoryLayout<Float>.stride
    idx += 1

    // brush value
    mtlVertexDescriptor.attributes[idx].format = MTLVertexFormat.float
    mtlVertexDescriptor.attributes[idx].offset = offset
    mtlVertexDescriptor.attributes[idx].bufferIndex = 0
    offset += MemoryLayout<Float>.stride
    idx += 1

    // birth time
    mtlVertexDescriptor.attributes[idx].format = MTLVertexFormat.float
    mtlVertexDescriptor.attributes[idx].offset = offset
    mtlVertexDescriptor.attributes[idx].bufferIndex = 0
    offset += MemoryLayout<Float>.stride
    idx += 1

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
    pipelineDescriptor.colorAttachments[0].isBlendingEnabled = true

    // Change to additive blending for particle effects
    pipelineDescriptor.colorAttachments[0].rgbBlendOperation = .add
    pipelineDescriptor.colorAttachments[0].alphaBlendOperation = .add
    pipelineDescriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
    pipelineDescriptor.colorAttachments[0].sourceAlphaBlendFactor = .one
    pipelineDescriptor.colorAttachments[0].destinationRGBBlendFactor = .one
    pipelineDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .one

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
        for _ in 0..<40 {
          let offset = randomSpherePosition(radius: 0.1)
          let nextPosition = position + offset * randBaseFromTo(0.5, 2.0)
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
