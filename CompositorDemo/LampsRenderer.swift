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

let lampCount: Int = 2000
let patelPerLamp: Int = 12
let verticesPerLamp = patelPerLamp * 2 + 1
let verticesCount = verticesPerLamp * lampCount

let rectIndexesPerRect: Int = 6 * patelPerLamp  // 6 vertices per rectangle
let ceilingIndexesPerLamp: Int = patelPerLamp * 3  // cover the top of the lamp with triangles
// prepare the vertices for the lamp, 1 extra vertex for the top center of the lamp
let indexesPerLamp = rectIndexesPerRect + ceilingIndexesPerLamp
// prepare the indices for the lamp
let indexesCount: Int = lampCount * indexesPerLamp

let verticalScale: Float = 0.4
let upperRadius: Float = 0.14
let lowerRadius: Float = 0.18

struct LampBase {
    var position: SIMD3<Float>
    var color: SIMD3<Float>
    var seed: Float
}

struct Params {
    var time: Float
}

@MainActor
class LampsRenderer: CustomRenderer {
    private let renderPipelineState: MTLRenderPipelineState & Sendable

    private var uniformsBuffer: [MTLBuffer]
    /// a buffer to hold the vertices of the lamp
    var vertexBuffer: MTLBuffer!

    var indexBuffer: MTLBuffer!

    let computeDevice: MTLDevice
    var computeBuffer: PingPongBuffer?
    let computePipeLine: MTLComputePipelineState
    let computeCommandQueue: MTLCommandQueue

    init(layerRenderer: LayerRenderer) throws {
        uniformsBuffer = (0..<Renderer.maxFramesInFlight).map { _ in
            layerRenderer.device.makeBuffer(length: MemoryLayout<PathProperties>.uniformStride)!
        }

        renderPipelineState = try Self.makeRenderPipelineDescriptor(layerRenderer: layerRenderer)

        self.computeDevice = MTLCreateSystemDefaultDevice()!
        let library = computeDevice.makeDefaultLibrary()!
        let lampsUpdateBase = library.makeFunction(name: "lampsComputeShader")!
        computePipeLine = try computeDevice.makeComputePipelineState(function: lampsUpdateBase)

        computeCommandQueue = computeDevice.makeCommandQueue()!

        self.createLampVerticesBuffer(device: layerRenderer.device)
        self.createLampIndexBuffer(device: layerRenderer.device)
        self.createLampComputeBuffer(device: layerRenderer.device)
    }

    /// create and sets the vertices of the lamp
    private func createLampVerticesBuffer(device: MTLDevice) {
        let bufferLength = MemoryLayout<Vertex>.stride * verticesCount
        vertexBuffer = device.makeBuffer(length: bufferLength)!
        vertexBuffer.label = "Lamp vertex buffer"
        var lampVertices: UnsafeMutablePointer<Vertex> {
            vertexBuffer.contents().assumingMemoryBound(to: Vertex.self)
        }

        for i in 0..<lampCount {
            // Random position offsets for each lamp
            // let xOffset = Float.random(in: -40...40)
            // let zOffset = Float.random(in: -40...2)
            // let yOffset = Float.random(in: 0...20)

            // let lampPosition = SIMD3<Float>(xOffset, yOffset, zOffset)
            // Random color for each lamp
            let r = Float.random(in: 0.1...1.0)
            let g = Float.random(in: 0.1...1.0)
            let b = Float.random(in: 0.1...1.0)
            let color = SIMD3<Float>(r, g, b)
            let dimColor = color * 0.5
            let baseIndex = i * verticesPerLamp

            for p in 0..<patelPerLamp {
                let angle = Float(p) * (2 * Float.pi / Float(patelPerLamp))

                // Calculate the four corners of this rectangular petal
                // Calculate the upper and lower points of petals on x-z plane
                // upper ring
                let upperEdge = SIMD3<Float>(
                    cos(angle) * upperRadius, verticalScale, sin(angle) * upperRadius)

                // lower ring
                let lowerEdge = SIMD3<Float>(
                    cos(angle) * lowerRadius, 0, sin(angle) * lowerRadius)

                let vertexBase = baseIndex + p

                // First triangle of rectangle (inner1, outer1, inner2)
                lampVertices[vertexBase] = Vertex(
                    position: upperEdge, color: color, seed: Int32(i))
                lampVertices[vertexBase + patelPerLamp] = Vertex(
                    position: lowerEdge,
                    color: dimColor,
                    seed: Int32(i)
                )
            }
            // top center of the lamp
            lampVertices[baseIndex + patelPerLamp * 2] = Vertex(
                position: SIMD3<Float>(0, verticalScale, 0),
                color: color * 1.2,
                seed: Int32(i)
            )
        }
    }

    func resetComputeState() {
        self.createLampComputeBuffer(device: computeDevice)
    }

    private func createLampIndexBuffer(device: MTLDevice) {
        let bufferLength = MemoryLayout<UInt32>.stride * indexesCount
        indexBuffer = device.makeBuffer(length: bufferLength)!
        indexBuffer.label = "Lamp index buffer"

        let lampIndices = indexBuffer.contents().bindMemory(
            to: UInt32.self, capacity: indexesCount)
        for i in 0..<lampCount {
            // for vertices in each lamp, layout is top "vertices, bottom vertices, top center"
            let verticesBase = i * verticesPerLamp

            let indexBase = i * indexesPerLamp
            // rect angles of patel size
            for p in 0..<patelPerLamp {
                let vertexBase = verticesBase + p
                let nextVertexBase = verticesBase + (p + 1) % patelPerLamp
                let nextIndexBase = indexBase + p * 6
                // First triangle of rectangle (inner1, outer1, inner2)
                lampIndices[nextIndexBase] = UInt32(vertexBase)
                lampIndices[nextIndexBase + 1] = UInt32(vertexBase + patelPerLamp)
                lampIndices[nextIndexBase + 2] = UInt32(nextVertexBase)

                // Second triangle of rectangle (inner2, outer1, outer2)
                lampIndices[nextIndexBase + 3] = UInt32(nextVertexBase)
                lampIndices[nextIndexBase + 4] = UInt32(vertexBase + patelPerLamp)
                lampIndices[nextIndexBase + 5] = UInt32(nextVertexBase + patelPerLamp)
            }
            // cover the top of the lamp with triangles
            let topCenter = verticesBase + patelPerLamp * 2
            let topCenterIndexBase = indexBase + rectIndexesPerRect
            for p in 0..<patelPerLamp {
                let vertexBase = verticesBase + p
                let nextVertexBase = verticesBase + (p + 1) % patelPerLamp
                let nextIndexBase = topCenterIndexBase + p * 3
                // First triangle of rectangle (inner1, outer1, inner2)
                lampIndices[nextIndexBase] = UInt32(vertexBase)
                lampIndices[nextIndexBase + 1] = UInt32(topCenter)
                lampIndices[nextIndexBase + 2] = UInt32(nextVertexBase)
            }
        }

    }

    private func createLampComputeBuffer(device: MTLDevice) {
        let bufferLength = MemoryLayout<LampBase>.stride * lampCount

        computeBuffer = PingPongBuffer(device: device, length: bufferLength)

        guard let computeBuffer = computeBuffer else {
            print("Failed to create compute buffer")
            return
        }
        computeBuffer.addLabel("Lamp compute buffer")

        let contents = computeBuffer.currentBuffer.contents()
        let lampBase = contents.bindMemory(to: LampBase.self, capacity: lampCount)

        for i in 0..<lampCount {
            // Random position offsets for each lamp
            let xOffset = Float.random(in: -20...20)
            let zOffset = Float.random(in: -30...10)
            let yOffset = Float.random(in: 0...2)

            let lampPosition = SIMD3<Float>(xOffset, yOffset, zOffset)
            // Random color for each lamp
            let r = Float.random(in: 0.1...1.0)
            let g = Float.random(in: 0.1...1.0)
            let b = Float.random(in: 0.1...1.0)
            let color = SIMD3<Float>(r, g, b)
            let dimColor = color * 0.5

            lampBase[i] = LampBase(position: lampPosition, color: color, seed: Float(i))
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
            MemoryLayout<Vertex>.stride
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

        let vertexFunction = library.makeFunction(name: "lampsVertexShader")
        let fragmentFunction = library.makeFunction(name: "lampsFragmentShader")

        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexFunction = vertexFunction

        pipelineDescriptor.label = "TriangleRenderPipeline"
        pipelineDescriptor.vertexDescriptor = LampsRenderer.buildMetalVertexDescriptor()

        return try layerRenderer.device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    func drawCommand(frame: LayerRenderer.Frame) throws -> TintDrawCommand {
        return TintDrawCommand(
            frameIndex: frame.frameIndex,
            uniforms: self.uniformsBuffer[Int(frame.frameIndex % Renderer.maxFramesInFlight)])
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

        var params = Params(time: dt)
        computeEncoder.setBytes(&params, length: MemoryLayout<Params>.size, index: 2)
        let threadGroupSize = min(computePipeLine.maxTotalThreadsPerThreadgroup, 256)
        let threadsPerThreadgroup = MTLSize(width: threadGroupSize, height: 1, depth: 1)
        let threadGroups = MTLSize(
            width: (lampCount + threadGroupSize - 1) / threadGroupSize,
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

        // let bufferLength = MemoryLayout<Vertex>.stride * numVertices

        encoder.setVertexBuffer(
            buffer,
            offset: 0,
            index: BufferIndex.meshPositions.rawValue)

        var params_data = Params(time: 0)

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
}
