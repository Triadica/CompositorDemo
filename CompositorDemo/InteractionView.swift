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

  @State private var selectedDemo: DemoTab = .flowers
  @State private var isUpdatingDemo = false
  @State private var backgroundTask: Task<Void, Never>? = nil

  var body: some View {
    HStack {
      Picker("Demo", selection: $selectedDemo) {
        Text("Flowers").tag(DemoTab.flowers)
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
        Text("Spread Around Ball").tag(DemoTab.spreadAroundBall)
        Text("Spread In Ball").tag(DemoTab.spreadInBall)
        Text("Bounce Around Cube").tag(DemoTab.bounceAroundCube)
        Text("Bounce Gravity").tag(DemoTab.bounceGravity)
        Text("Multi Gravity").tag(DemoTab.multiGravity)
        Text("Conflict Force").tag(DemoTab.conflictForce)
        Text("Rain").tag(DemoTab.rain)
        Text("Dome").tag(DemoTab.dome)
        Text("Mag Field").tag(DemoTab.magField)
        Text("Black Hole").tag(DemoTab.blackHole)
        Text("Wind Tunnel").tag(DemoTab.windTunnel)
      }.pickerStyle(.wheel).padding(.bottom, 32).frame(
        width: 300,
        height: 400,
        alignment: .center)
      VStack {
        Button {
          let timestamp = Date().formatted(.dateTime.minute().second())
          print(
            "[\(timestamp)] InteractionView: User toggled immersive space (current: \(appModel.showImmersiveSpace))"
          )
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
                let timestamp = Date().formatted(.dateTime.minute().second())
                print("[\(timestamp)] InteractionView: User pressed Reset Base button")
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
      let timestamp = Date().formatted(.dateTime.minute().second())
      print("[\(timestamp)] InteractionView: Scene phase changed to \(newPhase)")

      // 取消之前的后台任务
      backgroundTask?.cancel()
      backgroundTask = nil

      Task { @MainActor in
        if newPhase == .background {
          print(
            "[\(Date().formatted(.dateTime.minute().second()))] InteractionView: App going to background, starting delay timer"
          )
          // 添加5秒延迟，避免短暂的后台状态导致immersive空间退出
          backgroundTask = Task {
            do {
              try await Task.sleep(nanoseconds: 5_000_000_000)  // 5秒
              if !Task.isCancelled {
                print(
                  "[\(Date().formatted(.dateTime.minute().second()))] InteractionView: Background timeout reached, hiding immersive space"
                )
                appModel.showImmersiveSpace = false
              }
            } catch {
              // Task被取消，不做任何操作
              print(
                "[\(Date().formatted(.dateTime.minute().second()))] InteractionView: Background timer cancelled"
              )
            }
          }
        } else if newPhase == .active {
          print(
            "[\(Date().formatted(.dateTime.minute().second()))] InteractionView: App became active, cancelling background timer"
          )
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
      // 防止UIPickerView并发更新冲突
      guard !isUpdatingDemo else { return }
      isUpdatingDemo = true

      let timestamp = Date().formatted(.dateTime.minute().second())
      print(
        "[\(timestamp)] InteractionView: Demo selection changed from \(appModel.selectedTab) to \(newDemo)"
      )

      Task { @MainActor in
        // 添加短暂延迟以避免并发更新
        try? await Task.sleep(nanoseconds: 50_000_000)  // 50ms
        print(
          "[\(Date().formatted(.dateTime.minute().second()))] InteractionView: Applying demo change to \(newDemo)"
        )
        appModel.selectedTab = newDemo
        isUpdatingDemo = false
      }
    }
  }
}

#Preview(windowStyle: .automatic) {
  InteractionView()
    .environment(AppModel())
}
