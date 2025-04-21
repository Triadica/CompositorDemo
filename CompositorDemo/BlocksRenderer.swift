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

private let maxFramesInFlight = 3

private let gridSize: Int = 100

private let blocksCount: Int = gridSize * gridSize
private let verticesPerBlock = 8
private let verticesCount = verticesPerBlock * blocksCount

private let indexesPerBlock = 30  // 5 faces
private let indexesCount: Int = blocksCount * indexesPerBlock

private let blockRadius: Float = 2  // half width
private let blockHeight: Float = 20

private struct CellBase {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var blockIdf: Float
  var velocity: SIMD3<Float> = .zero
}

private struct Params {
  /// prefer time relative to the start of the app, not delta time
  var time: Float
  var viewerPosition: SIMD3<Float>
  var viewerScale: Float
  var _padding: SIMD4<Float> = .zero  // required for 48 bytes alignment
}

@MainActor
class BlocksRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the lamp
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  let computeDevice: MTLDevice
  var computeBuffer: PingPongBuffer?
  let computePipeLine: MTLComputePipelineState
  let computeCommandQueue: MTLCommandQueue

  var guestureManager: GestureManager = GestureManager()

  init(layerRenderer: LayerRenderer) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    self.computeDevice = MTLCreateSystemDefaultDevice()!
    let library = computeDevice.makeDefaultLibrary()!
    let cellUpdateBase: any MTLFunction = library.makeFunction(name: "blocksComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: cellUpdateBase)

    computeCommandQueue = computeDevice.makeCommandQueue()!

    self.createBlocksVerticesBuffer(device: layerRenderer.device)
    self.createBlocksIndexBuffer(device: layerRenderer.device)
    self.createBlocksComputeBuffer(device: layerRenderer.device)
  }

  /// create and sets the vertices of the lamp
  private func createBlocksVerticesBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<BlockVertex>.stride * verticesCount
    vertexBuffer = device.makeBuffer(length: bufferLength)!
    vertexBuffer.label = "Lamp vertex buffer"
    var cellVertices: UnsafeMutablePointer<BlockVertex> {
      vertexBuffer.contents().assumingMemoryBound(to: BlockVertex.self)
    }

    let unit = 5

    for i in 0..<gridSize {
      for j in 0..<gridSize {
        let idx = i * gridSize + j
        // Random color for each lamp
        let red = Float.random(in: 0.4...0.7)
        let g = Float.random(in: 0.4...0.7)
        let b = Float.random(in: 0.4...0.7)
        var color = SIMD3<Float>(red, g, b)

        let baseIndex = idx * verticesPerBlock
        var randHeight = pow(Float.random(in: 0.1...1), 3) * blockHeight
        if i % unit == 0 || j % unit == 0 {
          randHeight = 0.2
          color = SIMD3<Float>(0.1, 0.1, 0.1)
        }

        let r: Float = blockRadius * 0.8

        let p1: SIMD3<Float> = SIMD3<Float>(-r, 0, -r)
        let p2: SIMD3<Float> = SIMD3<Float>(r, 0, -r)
        let p3: SIMD3<Float> = SIMD3<Float>(r, 0, r)
        let p4: SIMD3<Float> = SIMD3<Float>(-r, 0, r)
        let p5: SIMD3<Float> = SIMD3<Float>(-r, randHeight, -r)
        let p6: SIMD3<Float> = SIMD3<Float>(r, randHeight, -r)
        let p7: SIMD3<Float> = SIMD3<Float>(r, randHeight, r)
        let p8: SIMD3<Float> = SIMD3<Float>(-r, randHeight, r)

        cellVertices[baseIndex] = BlockVertex(
          position: p1, color: color, seed: Int32(idx))
        cellVertices[baseIndex + 1] = BlockVertex(
          position: p2, color: color, seed: Int32(idx))
        cellVertices[baseIndex + 2] = BlockVertex(
          position: p3, color: color, seed: Int32(idx))
        cellVertices[baseIndex + 3] = BlockVertex(
          position: p4, color: color, seed: Int32(idx))
        cellVertices[baseIndex + 4] = BlockVertex(
          position: p5, color: color, seed: Int32(idx))
        cellVertices[baseIndex + 5] = BlockVertex(
          position: p6, color: color, seed: Int32(idx))
        cellVertices[baseIndex + 6] = BlockVertex(
          position: p7, color: color, seed: Int32(idx))
        cellVertices[baseIndex + 7] = BlockVertex(
          position: p8, color: color, seed: Int32(idx))
      }
    }

  }

  func resetComputeState() {
    self.createBlocksComputeBuffer(device: computeDevice)
  }

  private func createBlocksIndexBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "Lamp index buffer"

    let cellIndices = indexBuffer.contents().bindMemory(
      to: UInt32.self, capacity: indexesCount)
    for i in 0..<blocksCount {
      // for vertices in each lamp, layout is top "vertices, bottom vertices, top center"
      let verticesBase = i * verticesPerBlock

      // 8 points, 5 faces(one missing since we don't need to cover the bottom)
      let indexBase = i * indexesPerBlock
      // 0, 1, 2, 3 for bottom rectangle
      // 4, 5, 6, 7 for top rectangle

      // now face front from 0145, make 2 triangles
      cellIndices[indexBase] = UInt32(verticesBase + 0)
      cellIndices[indexBase + 1] = UInt32(verticesBase + 1)
      cellIndices[indexBase + 2] = UInt32(verticesBase + 4)
      cellIndices[indexBase + 3] = UInt32(verticesBase + 1)
      cellIndices[indexBase + 4] = UInt32(verticesBase + 4)
      cellIndices[indexBase + 5] = UInt32(verticesBase + 5)
      // now face back from 2367
      cellIndices[indexBase + 6] = UInt32(verticesBase + 2)
      cellIndices[indexBase + 7] = UInt32(verticesBase + 3)
      cellIndices[indexBase + 8] = UInt32(verticesBase + 6)
      cellIndices[indexBase + 9] = UInt32(verticesBase + 3)
      cellIndices[indexBase + 10] = UInt32(verticesBase + 6)
      cellIndices[indexBase + 11] = UInt32(verticesBase + 7)
      // now face left from 0347
      cellIndices[indexBase + 12] = UInt32(verticesBase + 0)
      cellIndices[indexBase + 13] = UInt32(verticesBase + 3)
      cellIndices[indexBase + 14] = UInt32(verticesBase + 4)
      cellIndices[indexBase + 15] = UInt32(verticesBase + 3)
      cellIndices[indexBase + 16] = UInt32(verticesBase + 4)
      cellIndices[indexBase + 17] = UInt32(verticesBase + 7)
      // now face right from 1256
      cellIndices[indexBase + 18] = UInt32(verticesBase + 1)
      cellIndices[indexBase + 19] = UInt32(verticesBase + 2)
      cellIndices[indexBase + 20] = UInt32(verticesBase + 5)
      cellIndices[indexBase + 21] = UInt32(verticesBase + 2)
      cellIndices[indexBase + 22] = UInt32(verticesBase + 5)
      cellIndices[indexBase + 23] = UInt32(verticesBase + 6)
      // now face top from 4567
      cellIndices[indexBase + 24] = UInt32(verticesBase + 4)
      cellIndices[indexBase + 25] = UInt32(verticesBase + 5)
      cellIndices[indexBase + 26] = UInt32(verticesBase + 6)
      cellIndices[indexBase + 27] = UInt32(verticesBase + 4)
      cellIndices[indexBase + 28] = UInt32(verticesBase + 6)
      cellIndices[indexBase + 29] = UInt32(verticesBase + 7)
    }

  }

  private func createBlocksComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<CellBase>.stride * blocksCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Lamp compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let blocksBase = contents.bindMemory(to: CellBase.self, capacity: blocksCount)

    let middle = Float(gridSize) / 2.0

    for i in 0..<gridSize {
      for j in 0..<gridSize {
        let xOffset = (Float(i) - middle) * blockRadius * 2
        let zOffset = (Float(j) - middle) * blockRadius * 2
        let yOffset: Float = 0.0

        let lampPosition = SIMD3<Float>(xOffset, yOffset, zOffset)
        // Random color for each lamp
        let r = Float.random(in: 0.1...1.0)
        let g = Float.random(in: 0.1...1.0)
        let b = Float.random(in: 0.1...1.0)
        let color = SIMD3<Float>(r, g, b)

        blocksBase[i * gridSize + j] = CellBase(
          position: lampPosition, color: color, blockIdf: Float(i), velocity: .zero)
      }
    }

    computeBuffer.copy_to_next()
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
      MemoryLayout<BlockVertex>.stride
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

    let vertexFunction = library.makeFunction(name: "blocksVertexShader")
    let fragmentFunction = library.makeFunction(name: "blocksFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "TriangleRenderPipeline"
    pipelineDescriptor.vertexDescriptor = BlocksRenderer.buildMetalVertexDescriptor()

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

    var params = Params(
      time: dt, viewerPosition: guestureManager.viewerPosition,
      viewerScale: guestureManager.viewerScale)
    computeEncoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 2)
    let threadGroupSize = min(computePipeLine.maxTotalThreadsPerThreadgroup, 256)
    let threadsPerThreadgroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
    let threadGroups = MTLSize(
      width: (blocksCount + threadGroupSize - 1) / threadGroupSize,
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

    // let bufferLength = MemoryLayout<BlockVertex>.stride * numVertices

    encoder.setVertexBuffer(
      buffer,
      offset: 0,
      index: BufferIndex.meshPositions.rawValue)

    var params_data = Params(
      time: getTimeSinceStart(),
      viewerPosition: guestureManager.viewerPosition,
      viewerScale: guestureManager.viewerScale)

    let params: any MTLBuffer = device.makeBuffer(
      bytes: &params_data,
      length: MemoryLayout<Params>.size,
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

  func onSpatialEvents(events: SpatialEventCollection) {
    for event in events {
      guestureManager.onSpatialEvent(event: event)
    }
  }
}
