/*
 See the LICENSE.txt file for this sample’s licensing information.

 Abstract:
 A renderer that displays a set of color swatches.
 */

import Combine
import CompositorServices
import Foundation
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

private let maxFramesInFlight = 3

/// how many lines for this attractor
private let linesCount: Int = 100000
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

private struct BounceBase {
  var position: SIMD3<Float>
  var color: SIMD3<Float>
  var velocity: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
}

private struct Params {
  var time: Float
  var elapsed: Float
  var groupSize: Int32 = Int32(lineGroupSize)
  var viewerPosition: SIMD3<Float>
  var viewerScale: Float
  var viewerRotation: Float = 0.0
  var _padding: SIMD2<Float> = SIMD2<Float>(0, 0)  // Pad to 48 bytes, remove if shader expects 36 bytes
}

@MainActor
class MultiGravityRenderer: CustomRenderer {
  private var renderPipelineState: MTLRenderPipelineState & Sendable
  var computePipeLine: MTLComputePipelineState

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the lamp
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!
  private var sharedShaderAddress: SharedShaderAddress
  private var cancellables = Set<AnyCancellable>()

  let computeDevice: MTLDevice
  var computeBuffer: PingPongBuffer?
  let computeCommandQueue: MTLCommandQueue

  var gestureManager: GestureManager = GestureManager()

  init(layerRenderer: LayerRenderer, sharedShaderAddress: SharedShaderAddress) throws {
    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    self.computeDevice = MTLCreateSystemDefaultDevice()!
    let library = computeDevice.makeDefaultLibrary()!
    let attractorUpdateBase = library.makeFunction(name: "multiGravityComputeShader")!
    computePipeLine = try computeDevice.makeComputePipelineState(function: attractorUpdateBase)

    computeCommandQueue = computeDevice.makeCommandQueue()!

    self.sharedShaderAddress = sharedShaderAddress

    self.createAttractorVerticesBuffer(device: layerRenderer.device)
    self.createAttractorIndexBuffer(device: layerRenderer.device)
    self.createAttractorComputeBuffer(device: layerRenderer.device)

    setupBindings()
  }

  private func setupBindings() {
    sharedShaderAddress.$inputText
      .sink { [weak self] newUrl in  // weak reference is important to avoid retain cycles
        guard let self = self else {
          print("Class instance has been deallocated, cancelling sink operation")
          return
        }

        if !newUrl.isEmpty {
          let timestamp = Date().formatted(.dateTime.minute().second())
          print("[\(timestamp)] handle shared shader address: \(newUrl)")
          Task {
            await self.swapShaderWithFromUrl(newUrl)
          }
        }
      }
      .store(in: &cancellables)
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

  private func createAttractorComputeBuffer(device: MTLDevice) {
    let bufferLength = MemoryLayout<BounceBase>.stride * controlCount

    computeBuffer = PingPongBuffer(device: device, length: bufferLength)

    guard let computeBuffer = computeBuffer else {
      print("Failed to create compute buffer")
      return
    }
    computeBuffer.addLabel("Attractor compute buffer")

    let contents = computeBuffer.currentBuffer.contents()
    let attractorBase = contents.bindMemory(to: BounceBase.self, capacity: controlCount)

    for i in 0..<linesCount {
      let p = fibonacciGrid(n: Float(i), total: Float(linesCount))

      let color = p * 0.5 + SIMD3<Float>(0.5, 0.5, 0.5)

      for j in 0..<controlCountPerLine {
        let index = i * controlCountPerLine + j
        attractorBase[index] = BounceBase(
          position: (p * 1 + SIMD3<Float>(0, 0, -1)),
          color: color,
          velocity: SIMD3<Float>(0, 0, 0) + p * 0.0)
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

    let vertexFunction = library.makeFunction(name: "multiGravityVertexShader")
    let fragmentFunction = library.makeFunction(name: "multiGravityFragmentShader")

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

    var params = Params(
      time: delta,
      elapsed: dt,
      viewerPosition: self.gestureManager.viewerPosition,
      viewerScale: self.gestureManager.viewerScale,
      viewerRotation: self.gestureManager.viewerRotation)
    computeEncoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 2)
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

    // let bufferLength = MemoryLayout<LampsVertex>.stride * numVertices

    encoder.setVertexBuffer(
      buffer,
      offset: 0,
      index: BufferIndex.meshPositions.rawValue)

    let delta = -Float(viewStartTime.timeIntervalSinceNow)
    let dt = delta - frameDelta
    frameDelta = delta

    var params_data = Params(
      time: delta,
      elapsed: dt,
      viewerPosition: self.gestureManager.viewerPosition,
      viewerScale: self.gestureManager.viewerScale,
      viewerRotation: self.gestureManager.viewerRotation)

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

  /// track the position pinch started, following pinches define the velocity of moving, to update self.viewerPosition .
  /// other other chirality events are used for scaling the entity
  func onSpatialEvents(events: SpatialEventCollection) {
    for event in events {
      gestureManager.onSpatialEvent(event: event)
    }
  }

  /// load shader text from url with HTTP library,
  /// build library with `device.makeLibrary`,
  /// then update `renderPipelineState` and `computePipeLine` with the new shader.
  func swapShaderWithFromUrl(_ url: String) async {
    var timestamp = Date().formatted(.dateTime.minute().second())
    print("[\(timestamp)] Swapping shader with url: \(url)")
    // load shader text from url with HTTP library
    guard let shaderText = try? await URL.fetchText(from: URL(string: url)!) else {
      print("Failed to load shader text from url: \(url)")
      return
    }

    timestamp = Date().formatted(.dateTime.minute().second())
    print("[\(timestamp)] Loaded shader length: \(shaderText.count)")

    // build library with `device.makeLibrary`
    let library: MTLLibrary
    do {
      library = try await computeDevice.makeLibrary(source: shaderText, options: nil)
    } catch {
      print("Failed to create library from shader text: \(error)")
      return
    }
    timestamp = Date().formatted(.dateTime.minute().second())
    print("[\(timestamp)] Shader library created successfully")

    // does not work with render pipeline state...

    // guard let vertexFunction = library.makeFunction(name: "multiGravityVertexShader"),
    //   let fragmentFunction = library.makeFunction(name: "multiGravityFragmentShader")
    // else {
    //   print("Failed to create vertex or fragment function from library")
    //   return
    // }

    // let pipelineDescriptor = MTLRenderPipelineDescriptor()
    // pipelineDescriptor.vertexFunction = vertexFunction
    // pipelineDescriptor.fragmentFunction = fragmentFunction
    // pipelineDescriptor.vertexDescriptor = Self.buildMetalVertexDescriptor()
    // pipelineDescriptor.label = "MultiGravityRenderPipeline"

    // do {
    //   renderPipelineState = try await computeDevice.makeRenderPipelineState(
    //     descriptor: pipelineDescriptor)
    // } catch {
    //   print("Failed to create render pipeline state: \(error)")
    //   return
    // }

    let newComputePipeline: MTLComputePipelineState

    do {
      newComputePipeline = try await computeDevice.makeComputePipelineState(
        function: library.makeFunction(name: "computeCellMoving")!)
    } catch {
      print("Failed to create compute pipeline state: \(error)")
      return
    }

    timestamp = Date().formatted(.dateTime.minute().second())
    print("[\(timestamp)] Compute pipeline state created successfully")

    await MainActor.run {
      // update the compute pipeline state
      self.computePipeLine = newComputePipeline

      timestamp = Date().formatted(.dateTime.minute().second())
      print("[\(timestamp)] Compute pipeline swapped")
    }

  }

}

extension URL {
  static func fetchText(from url: URL) async throws -> String {
    do {
      var request = URLRequest(url: url)
      request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData

      let (data, response) = try await URLSession.shared.data(for: request)

      guard let httpResponse = response as? HTTPURLResponse,
        (200...299).contains(httpResponse.statusCode)
      else {
        if let httpResponse = response as? HTTPURLResponse {
          throw URLError(
            .badServerResponse,
            userInfo: [
              NSLocalizedDescriptionKey:
                "Server returned non-success status code: \(httpResponse.statusCode)"
            ])
        } else {
          throw URLError(
            .badServerResponse,
            userInfo: [
              NSLocalizedDescriptionKey: "Non-HTTP response or unknown error"
            ])
        }
      }

      guard let content = String(data: data, encoding: .utf8) else {
        throw URLError(.cannotDecodeContentData)

      }

      return content
    } catch let urlError as URLError {
      throw urlError
    } catch {
      throw error
    }
  }
}
