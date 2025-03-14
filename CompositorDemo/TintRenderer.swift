/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
A renderer that displays a set of color swatches.
*/

import CompositorServices
import Metal
import MetalKit
import Spatial
import simd

let maxFramesInFlight = 3

let lampCount: Int = 200
let patelPerLamp: Int = 12
let verticesPerLamp: Int = 6 * patelPerLamp  // 6 vertices per rectangle
let numVertices: Int = lampCount * verticesPerLamp

let verticalScale: Float = 0.4
let upperRadius: Float = 0.14
let lowerRadius: Float = 0.18

@MainActor
class TintRenderer {
    private let renderPipelineState: MTLRenderPipelineState & Sendable

    private var uniformsBuffer: [MTLBuffer]
    /// a buffer to hold the vertices of the lamp
    var lampVerticesBuffer: MTLBuffer!

    init(layerRenderer: LayerRenderer) throws {
        uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
            layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
        }

        renderPipelineState = try Self.makePipelineDescriptor(layerRenderer: layerRenderer)
        self.createLampVerticesBuffer(device: layerRenderer.device)
    }

    /// create and sets the vertices of the lamp
    private func createLampVerticesBuffer(device: MTLDevice) {
        let bufferLength = MemoryLayout<Vertex>.stride * numVertices
        lampVerticesBuffer = device.makeBuffer(length: bufferLength)!
        lampVerticesBuffer.label = "Lamp vertex buffer"
        var lampVertices: UnsafeMutablePointer<Vertex> {
            lampVerticesBuffer.contents().assumingMemoryBound(to: Vertex.self)
        }

        for i in 0..<lampCount {
            // Random position offsets for each lamp
            let xOffset = Float.random(in: -5...5)
            let zOffset = Float.random(in: -5...5)
            let yOffset = Float.random(in: 0...4)

            let lampPosition = SIMD3<Float>(xOffset, yOffset, zOffset)
            // Random color for each lamp
            let r = Float.random(in: 0.1...1.0)
            let g = Float.random(in: 0.1...1.0)
            let b = Float.random(in: 0.1...1.0)
            let color = SIMD3<Float>(r, g, b)
            let dimColor = color * 0.5
            let baseIndex = i * verticesPerLamp

            for p in 0..<patelPerLamp {
                let angle = Float(p) * (2 * Float.pi / Float(patelPerLamp))
                let nextAngle = Float(p + 1) * (2 * Float.pi / Float(patelPerLamp))

                // Calculate the four corners of this rectangular petal
                // Calculate the upper and lower points of petals on x-z plane
                // upper ring
                let upperEdge = SIMD3<Float>(
                    cos(angle) * upperRadius, verticalScale, sin(angle) * upperRadius)
                let upperEdgeNext = SIMD3<Float>(
                    cos(nextAngle) * upperRadius, verticalScale, sin(nextAngle) * upperRadius)

                // lower ring
                let lowerEdge = SIMD3<Float>(
                    cos(angle) * lowerRadius, 0, sin(angle) * lowerRadius)
                let lowerEdgeNext = SIMD3<Float>(
                    cos(nextAngle) * lowerRadius, 0, sin(nextAngle) * lowerRadius)

                let vertexBase = baseIndex + p * 6

                // First triangle of rectangle (inner1, outer1, inner2)
                lampVertices[vertexBase] = Vertex(
                    position: upperEdge + lampPosition, color: color)
                lampVertices[vertexBase + 1] = Vertex(
                    position: lowerEdge + lampPosition,
                    color: dimColor
                )
                lampVertices[vertexBase + 2] = Vertex(
                    position: upperEdgeNext + lampPosition,
                    color: color
                )

                // Second triangle of rectangle (inner2, outer1, outer2)
                lampVertices[vertexBase + 3] = Vertex(
                    position: upperEdgeNext + lampPosition,
                    color: color
                )
                lampVertices[vertexBase + 4] = Vertex(
                    position: lowerEdge + lampPosition,
                    color: dimColor
                )
                lampVertices[vertexBase + 5] = Vertex(
                    position: lowerEdgeNext + lampPosition,
                    color: dimColor
                )
            }
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

        return mtlVertexDescriptor
    }

    private static func makePipelineDescriptor(layerRenderer: LayerRenderer) throws
        -> MTLRenderPipelineState
    {
        let pipelineDescriptor = Renderer.defaultRenderPipelineDescriptor(
            layerRenderer: layerRenderer)

        let library = layerRenderer.device.makeDefaultLibrary()!

        let vertexFunction = library.makeFunction(name: "tintVertexShader")
        let fragmentFunction = library.makeFunction(name: "tintFragmentShader")

        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexFunction = vertexFunction

        pipelineDescriptor.label = "TriangleRenderPipeline"
        pipelineDescriptor.vertexDescriptor = TintRenderer.buildMetalVertexDescriptor()

        return try layerRenderer.device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
        return TintDrawCommand(
            frameIndex: frame.frameIndex,
            uniforms: self.uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)])
    }

    @RendererActor
    func encodeDraw(
        _ drawCommand: TintDrawCommand,
        encoder: MTLRenderCommandEncoder,
        drawable: LayerRenderer.Drawable,
        device: MTLDevice, tintValue: Float,
        buffer: MTLBuffer
    ) {
        encoder.setCullMode(.none)

        encoder.setRenderPipelineState(renderPipelineState)

        var tintUniform: TintUniforms = TintUniforms(tintOpacity: tintValue)
        encoder.setVertexBytes(
            &tintUniform,
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
        encoder.drawPrimitives(
            type: .triangle,
            vertexStart: 0,
            vertexCount: numVertices
        )
    }

    @RendererActor
    func updateUniformBuffers(
        _ drawCommand: TintDrawCommand,
        drawable: LayerRenderer.Drawable
    ) {
        drawCommand.uniforms.contents().assumingMemoryBound(to: Uniforms.self).pointee = Uniforms(
            drawable: drawable)
    }
}

@RendererActor
struct TintDrawCommand {
    @RendererActor
    fileprivate struct DrawCommand {
        let buffer: MTLBuffer
        let vertexCount: Int
    }

    fileprivate let drawCommand: DrawCommand
    fileprivate let frameIndex: LayerFrameIndex
    fileprivate let uniforms: MTLBuffer & Sendable

    @MainActor
    fileprivate init(frameIndex: LayerFrameIndex, uniforms: MTLBuffer) {
        self.drawCommand = DrawCommand(buffer: uniforms, vertexCount: numVertices)
        self.frameIndex = frameIndex
        self.uniforms = uniforms
    }
}
