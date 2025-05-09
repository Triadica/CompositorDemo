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

let sparksLimit = 24000

func randomFireworksColor() -> SIMD3<Float> {
  let colorType = Float.random(in: 0...1)
  let color: SIMD3<Float>

  if colorType < 0.90 {
    // Orange-yellow colors (85% chance)
    color = SIMD3<Float>(
      Float.random(in: 0.8...1.0),  // Red: high
      Float.random(in: 0.4...0.7),  // Green: medium
      Float.random(in: 0.0...0.3)  // Blue: low
    )
  } else if colorType < 0.96 {
    // purple accents (6% chance)
    color = SIMD3<Float>(
      Float.random(in: 0.7...1.0),  // Red: high
      Float.random(in: 0.0...0.3),  // Green: low
      Float.random(in: 0.3...0.6)  // Blue: medium-low
    )
  } else {
    // Red accents (4% chance)
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

func getTimeSinceStart() -> Float {
  let time: UInt64 = DispatchTime.now().uptimeNanoseconds
  let timeSinceStart = Float(time) / 1_000_000_000
  return timeSinceStart
}
/// it has a limit of 4000 lines, if succeeds, it will overwrite from start, tracked with cursorIdx
private struct SparksCollection {
  var coll: [SparkLine] = []
  private var cursorIdx: Int = 0
  let emptyLine = SparkLine(
    from: SIMD3<Float>(0, 0, -1),
    to: SIMD3<Float>(1, 0, -1),
    birthTime: getTimeSinceStart(),
    color: SIMD3<Float>(1, 0, 0)
  )

  func nextIdx() -> Int {
    if cursorIdx >= sparksLimit {
      return 0
    } else {
      return cursorIdx
    }
  }

  mutating func add(_ from: SIMD3<Float>, _ to: SIMD3<Float>, time: Float) {
    // Create predominantly orange-yellow colors with occasional blue/green accents
    let color = randomFireworksColor()
    let l = SparkLine(from: from, to: to, birthTime: time, color: color)
    if coll.count < sparksLimit {
      coll.append(l)
      cursorIdx += 1
    } else if cursorIdx == sparksLimit {
      cursorIdx = 0
      coll[cursorIdx] = l
      cursorIdx += 1
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
    if idx < coll.count {
      return coll[idx]
    } else {
      return emptyLine
    }
  }

  mutating func reset() {
    coll.removeAll()
    cursorIdx = 0
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
  private var currentVertexBufferSize: Int = sparksLimit * 6

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

    var polylineVertices: UnsafeMutablePointer<SparkVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: SparkVertex.self)
    }

    for i in 0..<sparksLimit {
      self.writeInVertexBuffer(polylineVertices, i)
    }

  }

  private func writeInVertexBuffer(
    _ polylineVertices: UnsafeMutablePointer<SparkVertex>,
    _ i: Int,
  ) {
    var base = i * 6
    let line = linesManager.getLineAt(i)
    // Generate a random color for each line
    let width: Float = 0.004

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
    polylineVertices[base] = spark1
    polylineVertices[base + 1] = spark2
    polylineVertices[base + 2] = spark3
    polylineVertices[base + 3] = spark1
    polylineVertices[base + 4] = spark3
    polylineVertices[base + 5] = spark4
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
    linesManager.reset()

    var polylineVertices: UnsafeMutablePointer<SparkVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: SparkVertex.self)
    }

    for i in 0..<sparksLimit {
      self.writeInVertexBuffer(polylineVertices, i)
    }
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

    var polylineVertices: UnsafeMutablePointer<SparkVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: SparkVertex.self)
    }

    // Handle spatial events here
    for event in sortedEvents {
      // let _chirality = event.chirality
      switch event.phase {
      case .active:
        let position = event.inputDevicePose!.pose3D.position.to_simd3

        // Add 40 lines for each active event
        for _ in 0..<40 {
          let shifted = randomSpherePosition(radius: 0.002)
          let offset = randomSpherePosition(radius: 0.1)
          let startPosition = position + shifted
          let nextPosition = startPosition + offset * randBaseFromTo(0.4, 1.6)
          // print("  chilarity: \(event.chirality!), position: \(position)")
          let cursorIdx = linesManager.nextIdx()
          linesManager.add(startPosition, nextPosition, time: getTimeSinceStart())
          self.writeInVertexBuffer(polylineVertices, cursorIdx)
        }

        let isMiror = Float.random(in: 0...1) > 0.84
        if isMiror {
          let groupShifted = randomSpherePosition(radius: 0.1) * Float.random(in: 0.4...1.2)
          let startPosition = position + groupShifted
          // Add 40 lines for each active event
          for _ in 0..<20 {
            let offset = randomSpherePosition(radius: 0.1)
            let nextPosition = startPosition + offset * randBaseFromTo(0.5, 2.0) * 0.4
            // print("  chilarity: \(event.chirality!), position: \(position)")
            let cursorIdx = linesManager.nextIdx()
            linesManager.add(startPosition, nextPosition, time: getTimeSinceStart())
            self.writeInVertexBuffer(polylineVertices, cursorIdx)
          }
        }

        break
      case .ended:
        break
      case .cancelled:
        break
      default:
        print("Other event: \(event)")
        break
      }
    }
  }
}
