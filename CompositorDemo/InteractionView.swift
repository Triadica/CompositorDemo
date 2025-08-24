/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
A view for changing app state.
*/

import SwiftUI

private enum VisibilityState: String, CaseIterable, Identifiable {
  case visibleState, hiddenState, automaticState
  var id: Self { self }
  var state: Visibility {
    switch self {
    case .visibleState:
      return .visible
    case .hiddenState:
      return .hidden
    case .automaticState:
      return .automatic
    }
  }
}

private enum IStyle: String, CaseIterable, Identifiable {
  case mixedStyle, fullStyle
  var id: Self { self }
  var style: ImmersionStyle {
    switch self {
    case .mixedStyle:
      return .mixed
    case .fullStyle:
      return .full
    }
  }
}

struct InteractionView: View {

  @Environment(\.scenePhase) private var scenePhase
  @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace
  @Environment(\.openImmersiveSpace) var openImmersiveSpace
  @Environment(AppModel.self) var appModel

  @State private var selectedLVState: VisibilityState = .visibleState
  @State private var selectedIStyle: IStyle = .fullStyle

  @State private var opacity = 1.0

  @EnvironmentObject var computeStateNotify: ResetComputeState

  @EnvironmentObject var sharedShaderAddress: SharedShaderAddress
  @State private var textInput: String = "http://192.168.31.166:8080/link.metal"

  @State private var selectedDemo: DemoTab = .octahedron

  var body: some View {
    HStack {
      Picker("Demo", selection: $selectedDemo) {
        Text("Octahedron").tag(DemoTab.octahedron)
        Text("Lamps").tag(DemoTab.lamps)
        Text("Polylines").tag(DemoTab.polylines)
        Text("Triangles").tag(DemoTab.triangles)
        Text("JSON Gen").tag(DemoTab.jsonGen)
        Text("Attractor").tag(DemoTab.attractor)
        Text("Blocks").tag(DemoTab.blocks)
        Text("Images").tag(DemoTab.images)
        Text("Drag Sparks").tag(DemoTab.dragSparks)
        Text("Bounce In Ball").tag(DemoTab.bounceInBall)
        Text("Bounce In Cube").tag(DemoTab.bounceInCube)
        Text("Bounce Around Ball").tag(DemoTab.bounceAroundBall)
        Text("Bounce Around Cube").tag(DemoTab.bounceAroundCube)
        Text("Bounce Gravity").tag(DemoTab.bounceGravity)
        Text("Multi Gravity").tag(DemoTab.multiGravity)
        Text("Conflict Force").tag(DemoTab.conflictForce)
        Text("Rain").tag(DemoTab.rain)
        Text("Dome").tag(DemoTab.dome)
      }.pickerStyle(.wheel).padding(.bottom, 32).frame(
        width: 300,
        height: 400,
        alignment: .center)
      VStack {
        Button {
          appModel.showImmersiveSpace.toggle()
        } label: {
          Text(
            appModel.showImmersiveSpace
              ? "Hide Immersive Space" : "Show Immersive Space")
        }
        .animation(.none, value: 0)
        .fontWeight(.semibold)
        if appModel.showImmersiveSpace {
          VStack {
            HStack {
              Text("Immersion Style")
              Picker("Immersion Style", selection: $selectedIStyle) {
                Text("Mixed").tag(IStyle.mixedStyle)
                Text("Full").tag(IStyle.fullStyle)
              }
            }
            HStack {
              Text("Upper Limbs")
              Picker("Upper Limb Visibility", selection: $selectedLVState) {
                Text("Visible").tag(VisibilityState.visibleState)
                Text("Hidden").tag(VisibilityState.hiddenState)
                Text("Automatic").tag(VisibilityState.automaticState)
              }
            }
            // Text("Tint Opacity \(opacity)")
            //     .fontWeight(.semibold)
            //     .padding(20)

            // Slider(value: $opacity, in: 0...1) {
            //     Text("Tint Opacity")
            // } minimumValueLabel: {
            //     Text("0")
            // } maximumValueLabel: {
            //     Text("1")
            // }
            HStack {
              Button {
                // to reset states in compute shader
                computeStateNotify.reset += 1
              } label: {
                Text("Reset Base")
              }
              .padding(.vertical, 30)  // Adds 10 points of padding on top and bottom
            }

            if selectedDemo == .multiGravity {
              VStack {
                TextField("Shader Url", text: $textInput)
                  .textFieldStyle(RoundedBorderTextFieldStyle())

                Button("Send Url") {
                  let timestamp = Date().formatted(.dateTime.minute().second())
                  print("[\(timestamp)] Sending URL: \(textInput)")
                  self.sharedShaderAddress.inputText = textInput
                }
              }
              .frame(width: 300)
            }

          }
        }
      }
    }
    .padding()
    .frame(width: 800, height: appModel.showImmersiveSpace ? 600 : 300)
    .onChange(of: scenePhase) { _, newPhase in
      Task { @MainActor in
        if newPhase == .background {
          appModel.showImmersiveSpace = false
        }
      }
    }
    .onChange(of: selectedLVState) { _, newState in
      appModel.upperLimbVisibility = newState.state
    }
    .onChange(of: opacity) { _, newState in
      appModel.opacity = Float(opacity)
    }
    .onChange(of: selectedIStyle) { _, newStyle in
      appModel.immersionStyle = newStyle.style
    }
    .onChange(of: selectedDemo) { _, newDemo in
      appModel.selectedTab = newDemo
    }
  }
}

#Preview(windowStyle: .automatic) {
  InteractionView()
    .environment(AppModel())
}
