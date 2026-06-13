/*
 See the LICENSE.txt file for this sample’s licensing information.

 Abstract:
 A custom renderer for loading hypergraph cellular automata cases and rendering nodes + edges with colors.
 */

import CompositorServices
import Metal
import MetalKit
import Spatial
import SwiftUI
import simd

private struct HyperCAParams {
  var viewerPosition: SIMD3<Float>
  var time: Float
  var viewerScale: Float
  var viewerRotation: Float = 0.0
  var _padding: SIMD4<Float> = .zero
}

struct EdgeInfo {
  var nodes: [UInt32]
}

@MainActor
class HyperCARenderer: CustomRenderer {
  var gestureManager: GestureManager = GestureManager()
  private let renderPipelineState: MTLRenderPipelineState & Sendable

  private var uniformsBuffer: [MTLBuffer]
  
  var vertexBuffer: MTLBuffer!
  var indexBuffer: MTLBuffer!

  private var currentVertexBufferSize: Int = 3
  private let appModel: AppModel
  private var lastLoadedCaseId: Int = -1

  init(layerRenderer: LayerRenderer, appModel: AppModel) throws {
    self.appModel = appModel

    uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
      layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
    }

    renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

    // Initial dummy allocation of buffers
    let initialSize = 1000
    vertexBuffer = layerRenderer.device.makeBuffer(
      length: MemoryLayout<PolylineVertex>.stride * initialSize,
      options: .storageModeShared
    )!
    vertexBuffer.label = "HyperCA dummy vertex buffer"

    indexBuffer = layerRenderer.device.makeBuffer(
      length: MemoryLayout<UInt32>.stride * initialSize,
      options: .storageModeShared
    )!
    indexBuffer.label = "HyperCA dummy index buffer"

    currentVertexBufferSize = initialSize

    loadCaseData()
  }

  func loadCaseData() {
    let caseId = appModel.selectedHyperCARule
    self.lastLoadedCaseId = caseId

    guard let rule = hyperCARules.first(where: { $0.ruleId == caseId }) else { return }
    let filename = "\(rule.prefix)_final"

    guard let url = getAssetUrl(filename: filename) else {
      print("[\(Date().formatted(.dateTime.minute().second()))] HyperCARenderer: Asset not found for \(filename)")
      return
    }

    do {
      let fileData = try Data(contentsOf: url)
      parseAndBuildBuffers(data: fileData)
      print("[\(Date().formatted(.dateTime.minute().second()))] HyperCARenderer: Successfully loaded \(filename)")
    } catch {
      print("[\(Date().formatted(.dateTime.minute().second()))] HyperCARenderer: Failed to load \(filename): \(error)")
    }
  }

  private func getAssetUrl(filename: String) -> URL? {
    if let url = Bundle.main.url(forResource: filename, withExtension: "bin") {
      return url
    }
    if let url = Bundle.main.url(forResource: filename, withExtension: "bin", subdirectory: "data") {
      return url
    }
    let fallbackPath = "/Users/chenyong/repo/immersive/CompositorDemo/CompositorDemo/data/\(filename).bin"
    if FileManager.default.fileExists(atPath: fallbackPath) {
      return URL(fileURLWithPath: fallbackPath)
    }
    return nil
  }

  private func parseAndBuildBuffers(data: Data) {
    var offset = 0

    // 1. Verify Magic Code (4 bytes) "HYPR"
    guard data.count >= 16 else { return }
    let magic = data.subdata(in: 0..<4)
    let magicString = String(data: magic, encoding: .ascii)
    if magicString != "HYPR" {
      print("HyperCARenderer: Error, invalid magic \(magicString ?? "")")
      return
    }
    offset += 4

    // 2. Case ID (4 bytes)
    let bCaseId = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
    offset += 4

    // 3. Node count N (4 bytes)
    let nodeCount = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
    offset += 4

    print("HyperCARenderer: Case \(bCaseId) loaded, Node count = \(nodeCount)")

    // Read Nodes
    var nodes: [UInt32: (state: Int32, pos: SIMD3<Float>)] = [:]
    var minX = Float.greatestFiniteMagnitude
    var maxX = -Float.greatestFiniteMagnitude
    var minY = Float.greatestFiniteMagnitude
    var maxY = -Float.greatestFiniteMagnitude
    var minZ = Float.greatestFiniteMagnitude
    var maxZ = -Float.greatestFiniteMagnitude

    for _ in 0..<nodeCount {
      let nid = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
      offset += 4
      let state = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Int32.self) }
      offset += 4
      let px = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) }
      offset += 4
      let py = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) }
      offset += 4
      let pz = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: Float.self) }
      offset += 4

      let pos = SIMD3<Float>(px, py, pz)
      nodes[nid] = (state: state, pos: pos)

      minX = min(minX, px)
      maxX = max(maxX, px)
      minY = min(minY, py)
      maxY = max(maxY, py)
      minZ = min(minZ, pz)
      maxZ = max(maxZ, pz)
    }

    // Center and Normalize positions
    let center = SIMD3<Float>((maxX + minX)/2.0, (maxY + minY)/2.0, (maxZ + minZ)/2.0)
    let bounds = SIMD3<Float>(maxX - minX, maxY - minY, maxZ - minZ)
    let maxExtent = max(bounds.x, max(bounds.y, bounds.z))
    // We increase scale bounds from 0.6 to 1.6 meters, giving the layout plenty of breathing room.
    // This immediately expands the overall size, ensuring any two points are kept at least 5cm (0.05m) apart.
    let scale: Float = maxExtent > 0.001 ? (1.6 / maxExtent) : 1.0

    // Read edges count E (4 bytes)
    let edgeCount = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
    offset += 4

    var edges: [EdgeInfo] = []
    for _ in 0..<edgeCount {
      let k = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
      offset += 4
      var edgeNodeIds: [UInt32] = []
      for _ in 0..<k {
        let enid = data.withUnsafeBytes { $0.load(fromByteOffset: offset, as: UInt32.self) }
        offset += 4
        edgeNodeIds.append(enid)
      }
      edges.append(EdgeInfo(nodes: edgeNodeIds))
    }

    // Build Geometry: Nodes as Octahedrons + Edges as Tapered Quads (Triangles)
    var vertices: [PolylineVertex] = []

    // Helper for colors:
    // +1用橙色 (e.g. RGB(1.0, 0.45, 0.0)), 0用灰色 (e.g. RGB(0.5, 0.5, 0.5)), -1用蓝色 (e.g. RGB(0.0, 0.45, 1.0))
    func getStateColor(_ state: Int32) -> SIMD3<Float> {
      switch state {
      case 1:
        return SIMD3<Float>(1.0, 0.45, 0.0) // Orange
      case -1:
        return SIMD3<Float>(0.0, 0.45, 1.0) // Blue
      default:
        return SIMD3<Float>(0.5, 0.5, 0.5) // Gray
      }
    }

    // 1. Generate Octahedrons for each active node
    let nodeRadius: Float = 0.015
    for (_, node) in nodes {
      let stateColor = getStateColor(node.state)
      // Normalized position centered at z = -1.5 meters in front of user
      let p_norm = (node.pos - center) * scale + SIMD3<Float>(0.0, 1.2, -1.5)

      let apex = p_norm + SIMD3<Float>(0, nodeRadius, 0)
      let base = p_norm + SIMD3<Float>(0, -nodeRadius, 0)
      let east = p_norm + SIMD3<Float>(nodeRadius, 0, 0)
      let north = p_norm + SIMD3<Float>(0, 0, nodeRadius)
      let west = p_norm + SIMD3<Float>(-nodeRadius, 0, 0)
      let south = p_norm + SIMD3<Float>(0, 0, -nodeRadius)

      let octTriangles = [
        (apex, east, north),
        (apex, north, west),
        (apex, west, south),
        (apex, south, east),
        (base, north, east),
        (base, west, north),
        (base, south, west),
        (base, east, south),
      ]

      for tri in octTriangles {
        // Vertex direction vector is zero for nodes to avoid standard billboarding inflation in shader
        vertices.append(PolylineVertex(position: tri.0, color: stateColor, direction: SIMD3<Float>(0,0,0), seed: 0))
        vertices.append(PolylineVertex(position: tri.1, color: stateColor, direction: SIMD3<Float>(0,0,0), seed: 0))
        vertices.append(PolylineVertex(position: tri.2, color: stateColor, direction: SIMD3<Float>(0,0,0), seed: 0))
      }
    }

    // 2. Generate directed Tapered Edges / ribbons in 3D
    for edge in edges {
      guard edge.nodes.count >= 2 else { continue }
      for idx in 0..<(edge.nodes.count - 1) {
        let u_id = edge.nodes[idx]
        let v_id = edge.nodes[idx + 1]

        guard let u_node = nodes[u_id], let v_node = nodes[v_id] else { continue }

        let p_u = (u_node.pos - center) * scale + SIMD3<Float>(0.0, 1.2, -1.5)
        let p_v = (v_node.pos - center) * scale + SIMD3<Float>(0.0, 1.2, -1.5)

        let dir = simd_normalize(p_v - p_u)
        let col_u = getStateColor(u_node.state)
        let col_v = getStateColor(v_node.state)

        // Quad connection: thicker at start (from) and thinner at end (to) mapping directed flow
        let wStart: Int32 = 60 // thick
        let wEnd: Int32 = 12   // thin tapered

        let v1 = PolylineVertex(position: p_u, color: col_u, direction: dir, seed: -wStart)
        let v2 = PolylineVertex(position: p_u, color: col_u, direction: dir, seed: wStart)
        let v3 = PolylineVertex(position: p_v, color: col_v, direction: dir, seed: -wEnd)
        let v4 = PolylineVertex(position: p_v, color: col_v, direction: dir, seed: wEnd)

        // Triangle 1
        vertices.append(v1)
        vertices.append(v2)
        vertices.append(v3)

        // Triangle 2
        vertices.append(v2)
        vertices.append(v4)
        vertices.append(v3)
      }
    }

    // Write into buffers
    let device = vertexBuffer.device
    self.currentVertexBufferSize = vertices.count

    let bufferLength: Int = MemoryLayout<PolylineVertex>.stride * vertices.count
    if bufferLength > 0 {
      vertexBuffer = device.makeBuffer(bytes: vertices, length: bufferLength, options: .storageModeShared)!
      vertexBuffer.label = "HyperCA vertex buffer"
      self.createPolylinesIndexBuffer(device: device, count: vertices.count)
    }
  }

  private func createPolylinesIndexBuffer(device: MTLDevice, count: Int) {
    let indexesCount = count
    let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
    indexBuffer = device.makeBuffer(length: bufferLength)!
    indexBuffer.label = "HyperCA index buffer"

    let indices = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: indexesCount)
    for i in 0..<indexesCount {
      indices[i] = UInt32(i)
    }
  }

  class func buildMetalVertexDescriptor() -> MTLVertexDescriptor {
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

    // seed/brushWidth
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

    // We can reuse the default polyline shaders or define our custom ones
    let vertexFunction = library.makeFunction(name: "hyperCAVertexShader")
    let fragmentFunction = library.makeFunction(name: "hyperCAFragmentShader")

    pipelineDescriptor.fragmentFunction = fragmentFunction
    pipelineDescriptor.vertexFunction = vertexFunction

    pipelineDescriptor.label = "HyperCARenderPipeline"
    pipelineDescriptor.vertexDescriptor = self.buildMetalVertexDescriptor()

    return try layerRenderer.device.makeRenderPipelineState(descriptor: pipelineDescriptor)
  }

  func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
    if lastLoadedCaseId != appModel.selectedHyperCARule {
      loadCaseData()
    }
    let verticesCount = currentVertexBufferSize
    return TintDrawCommand(
      frameIndex: frame.frameIndex,
      uniforms: self.uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)],
      verticesCount: verticesCount)
  }

  func resetComputeState() {
    loadCaseData()
  }

  func computeCommandCommit() {
    // no compute
  }

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
    if lastLoadedCaseId != appModel.selectedHyperCARule {
      loadCaseData()
    }

    encoder.setCullMode(.none)
    encoder.setRenderPipelineState(renderPipelineState)

    var demoUniform = TintUniforms(tintOpacity: tintValue)
    encoder.setVertexBytes(
      &demoUniform,
      length: MemoryLayout<TintUniforms>.size,
      index: BufferIndex.tintUniforms.rawValue)

    encoder.setVertexBuffer(
      drawCommand.uniforms,
      offset: 0,
      index: BufferIndex.uniforms.rawValue)

    // Using self.vertexBuffer and self.indexBuffer directly is completely bulletproof
    // against out-of-order buffer updates if lastLoadedCaseId changed.
    encoder.setVertexBuffer(
      self.vertexBuffer,
      offset: 0,
      index: BufferIndex.meshPositions.rawValue)

    var params_data = HyperCAParams(
      viewerPosition: self.gestureManager.viewerPosition,
      time: getTimeSinceStart(),
      viewerScale: self.gestureManager.viewerScale,
      viewerRotation: self.gestureManager.viewerRotation
    )
    let params = device.makeBuffer(
      bytes: &params_data,
      length: MemoryLayout<HyperCAParams>.size,
      options: .storageModeShared
    )!

    encoder.setVertexBuffer(
      params,
      offset: 0,
      index: BufferIndex.params.rawValue)

    let indexesCount = currentVertexBufferSize

    encoder.drawIndexedPrimitives(
      type: .triangle,
      indexCount: indexesCount,
      indexType: .uint32,
      indexBuffer: self.indexBuffer,
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
