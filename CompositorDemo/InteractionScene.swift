/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The main scene for presenting and interacting with content.
*/

import CompositorServices
import SwiftUI

struct ImmersiveInteractionScene: Scene {

    @Environment(AppModel.self) var appModel

    static let id = "ImmersiveInteractionScene"

    var body: some Scene {
        ImmersiveSpace(id: Self.id) {
            CompositorLayer(configuration: ContentStageConfiguration()) { layerRenderer in

                let lampsRenderer: LampsRenderer
                do {
                    lampsRenderer = try LampsRenderer(layerRenderer: layerRenderer)
                } catch {
                    fatalError("Failed to create lamps renderer \(error)")
                }

                Task(priority: .high) { @RendererActor in
                    Task { @MainActor in
                        appModel.lampsRenderer = lampsRenderer
                    }

                    let renderer = try await Renderer(
                        layerRenderer,
                        appModel,
                        lampsRenderer)
                    try await renderer.renderLoop()

                    Task { @MainActor in
                        appModel.lampsRenderer = nil
                    }
                }
            }
        }
        .immersionStyle(selection: .constant(appModel.immersionStyle), in: .mixed, .full)
        .upperLimbVisibility(appModel.upperLimbVisibility)
    }
}
