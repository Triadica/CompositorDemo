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

let demoResourceUrl = "http://192.168.31.166:8080/demo.json"

/// struct of triangle, with 3 points and a color of agba
struct Triangle: Decodable {
  var p1: Point3D
  var p2: Point3D
  var p3: Point3D
  var color: SIMD4<Float>

  enum CodingKeys: String, CodingKey {
    case p1, p2, p3, color
  }

  init(p1: Point3D, p2: Point3D, p3: Point3D, color: SIMD4<Float>) {
    self.p1 = p1
    self.p2 = p2
    self.p3 = p3
    self.color = color
  }

  init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    // Decode points from arrays
    if let p1Array = try? container.decode([Float].self, forKey: .p1) {
      guard p1Array.count >= 3 else {
        throw DecodingError.dataCorruptedError(
          forKey: .p1, in: container,
          debugDescription: "Point array must have at least 3 elements")
      }
      p1 = Point3D(x: Double(p1Array[0]), y: Double(p1Array[1]), z: Double(p1Array[2]))
    } else {
      p1 = try container.decode(Point3D.self, forKey: .p1)
    }

    if let p2Array = try? container.decode([Float].self, forKey: .p2) {
      guard p2Array.count >= 3 else {
        throw DecodingError.dataCorruptedError(
          forKey: .p2, in: container,
          debugDescription: "Point array must have at least 3 elements")
      }
      p2 = Point3D(x: Double(p2Array[0]), y: Double(p2Array[1]), z: Double(p2Array[2]))
    } else {
      p2 = try container.decode(Point3D.self, forKey: .p2)
    }

    if let p3Array = try? container.decode([Float].self, forKey: .p3) {
      guard p3Array.count >= 3 else {
        throw DecodingError.dataCorruptedError(
          forKey: .p3, in: container,
          debugDescription: "Point array must have at least 3 elements")
      }
      p3 = Point3D(x: Double(p3Array[0]), y: Double(p3Array[1]), z: Double(p3Array[2]))
    } else {
      p3 = try container.decode(Point3D.self, forKey: .p3)
    }

    // Handle SIMD4 color decoding
    if let colorArray = try? container.decode([Float].self, forKey: .color) {
      guard colorArray.count >= 4 else {
        throw DecodingError.dataCorruptedError(
          forKey: .color, in: container,
          debugDescription: "Color array must have at least 4 elements")
      }
      color = SIMD4<Float>(colorArray[0], colorArray[1], colorArray[2], colorArray[3])
    } else {
      color = SIMD4<Float>(1, 1, 1, 1)  // Default white color
    }
  }
}

@MainActor
class JsonGenRenderer: CustomRenderer {
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  /// a buffer to hold the vertices of the polyline
  var vertexBuffer: MTLBuffer!

  var indexBuffer: MTLBuffer!

  /// tracks buffer size, increased when points getting enormous, should be larger than 0
  private var currentVertexBufferSize: Int = 6

  private var trianglesList: [Triangle] = []

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

    for i in 0..<trianglesList.count {
      let triangle = trianglesList[i]
      // Generate a random color for each line
      // Convert SIMD4 color to SIMD3 by taking just the RGB components (ignoring alpha)
      let color = SIMD3<Float>(triangle.color.x, triangle.color.y, triangle.color.z)

      polylineVertices[pos] = PolylineVertex(
        position: triangle.p1.to_simd3,
        color: color,
        direction: SIMD3<Float>(0, 0, 0),
        seed: Int32(0)
      )

      pos += 1

      polylineVertices[pos] = PolylineVertex(
        position: triangle.p2.to_simd3,
        color: color,
        direction: SIMD3<Float>(0, 0, 0),
        seed: Int32(0)
      )
      pos += 1

      polylineVertices[pos] = PolylineVertex(
        position: triangle.p3.to_simd3,
        color: color,
        direction: SIMD3<Float>(0, 0, 0),
        seed: Int32(0)
      )
      pos += 1
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

    let vertexFunction = library.makeFunction(name: "jsonGenVertexShader")
    let fragmentFunction = library.makeFunction(name: "jsonGenFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "JsonGenRenderPipeline"
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
    // Parse the triangles list from JSON
    let decoder = JSONDecoder()
    print("Fetching JSON from \(demoResourceUrl)")
    do {
      guard let url = URL(string: demoResourceUrl) else {
        print("Invalid URL")
        return
      }

      // Create a new URLSession with no caching
      let config: URLSessionConfiguration = URLSessionConfiguration.default
      config.requestCachePolicy = .reloadIgnoringLocalCacheData
      config.urlCache = nil

      // Create and configure a URL session task
      // Use the custom URLSession with no caching
      let session = URLSession(configuration: config)
      let task = session.dataTask(with: url) { [weak self] data, response, error in
        guard let self = self else { return }

        if let error = error {
          print("Error fetching JSON: \(error)")
          return
        }

        guard let data = data else {
          print("No data received")
          return
        }

        print("Received JSON data. parsing...")

        do {
          // Parse the JSON data
          let parsedTriangles = try decoder.decode([Triangle].self, from: data)
          Task { @MainActor in
            self.trianglesList = parsedTriangles
            self.currentVertexBufferSize = self.trianglesList.count * 3

            // Create new buffers with the updated size
            self.createPolylinesVerticesBuffer(
              device: self.vertexBuffer.device,
              count: self.currentVertexBufferSize
            )
            self.createPolylinesIndexBuffer(
              device: self.vertexBuffer.device,
              count: self.currentVertexBufferSize
            )

            print("Successfully parsed \(self.trianglesList.count) triangles")
          }
        } catch {
          print("Error parsing triangles from server: \(error)")
        }
      }

      // Start the task
      task.resume()
    }
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

    // let sortedEvents = events.sorted {
    //   guard let chirality1 = $0.chirality, let chirality2 = $1.chirality else {
    //     return false
    //   }
    //   return chirality1.hashValue <= chirality2.hashValue
    // }

    // // Handle spatial events here
    // for event in sortedEvents {
    //   // let _chirality = event.chirality
    //   switch event.phase {
    //   case .active:
    //     let position = event.inputDevicePose!.pose3D.position
    //     linesManager.addPoint(position)
    //   case .ended:
    //     linesManager.finishCurrent()
    //   case .cancelled:
    //     linesManager.finishCurrent()

    //   default:
    //     print("Other event: \(event)")
    //     break
    //   }
    // }
    // updateVertexBuffer()
  }
}
