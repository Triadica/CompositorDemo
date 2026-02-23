import Foundation
import GameController
import simd

/// Manages gamepad/controller input and tracks player movement state.
/// Adapted from immersive-attractors for use in CompositorDemo.
class GameManager {
  private struct ControllerState {
    var leftStick: SIMD2<Float> = .zero
    var rightStick: SIMD2<Float> = .zero
    var buttonA: Bool = false
    var buttonB: Bool = false
    var buttonX: Bool = false
    var buttonY: Bool = false
    var boostActive: Bool = false
    // D-pad
    var dpadUp: Bool = false
    var dpadDown: Bool = false
    var dpadLeft: Bool = false
    var dpadRight: Bool = false
  }

  private let controllerQueue = DispatchQueue(label: "compositor-demo.controller.state")
  private var controllerState = ControllerState()
  private var lastInputLogTime: TimeInterval = 0

  private let movementSpeed: Float = 1.2
  private let yawSpeed: Float = .pi / 2.0
  private let deadZone: Float = 0.12
  private let boostMovementMultiplier: Float = 5.0
  private let boostYawMultiplier: Float = 2.0

  private(set) var playerOffset: SIMD3<Float> = .zero
  private(set) var yawAngle: Float = 0

  init() {
    setupControllerObserver()
  }

  // MARK: - Setup

  func setupControllerObserver() {
    NotificationCenter.default.addObserver(
      forName: .GCControllerDidConnect, object: nil, queue: nil
    ) { [weak self] notification in
      guard let controller = notification.object as? GCController else { return }
      self?.register(controller: controller)
    }

    NotificationCenter.default.addObserver(
      forName: .GCControllerDidDisconnect, object: nil, queue: nil
    ) { [weak self] notification in
      guard let controller = notification.object as? GCController else { return }
      print(
        "[GameManager] Controller disconnected: \(controller.vendorName ?? "Unknown Controller")")
    }

    GCController.startWirelessControllerDiscovery(completionHandler: nil)
    for controller in GCController.controllers() {
      register(controller: controller)
    }
  }

  private func register(controller: GCController) {
    print("[GameManager] Controller connected: \(controller.vendorName ?? "Unknown Controller")")
    guard let gamepad = controller.extendedGamepad else {
      print("[GameManager] Connected controller has no extended gamepad profile")
      return
    }

    gamepad.valueChangedHandler = { [weak self] gamepad, element in
      self?.handleInput(gamepad: gamepad, element: element)
    }
  }

  // MARK: - Input Handling

  private func handleInput(gamepad: GCExtendedGamepad, element: GCControllerElement) {
    let leftStick = SIMD2<Float>(
      gamepad.leftThumbstick.xAxis.value, gamepad.leftThumbstick.yAxis.value)
    let rightStick = SIMD2<Float>(
      gamepad.rightThumbstick.xAxis.value, gamepad.rightThumbstick.yAxis.value)
    let buttonA = gamepad.buttonA.isPressed
    let buttonB = gamepad.buttonB.isPressed
    let buttonX = gamepad.buttonX.isPressed
    let buttonY = gamepad.buttonY.isPressed
    let boostActive = gamepad.leftShoulder.isPressed || gamepad.rightShoulder.isPressed

    let dpadUp = gamepad.dpad.up.isPressed
    let dpadDown = gamepad.dpad.down.isPressed
    let dpadLeft = gamepad.dpad.left.isPressed
    let dpadRight = gamepad.dpad.right.isPressed

    controllerQueue.sync {
      controllerState.leftStick = leftStick
      controllerState.rightStick = rightStick
      controllerState.buttonA = buttonA
      controllerState.buttonB = buttonB
      controllerState.buttonX = buttonX
      controllerState.buttonY = buttonY
      controllerState.boostActive = boostActive
      controllerState.dpadUp = dpadUp
      controllerState.dpadDown = dpadDown
      controllerState.dpadLeft = dpadLeft
      controllerState.dpadRight = dpadRight
    }

    logInputEvent(
      element: element, leftStick: leftStick, rightStick: rightStick, buttonA: buttonA,
      boost: boostActive)
  }

  private func logInputEvent(
    element: GCControllerElement, leftStick: SIMD2<Float>, rightStick: SIMD2<Float>,
    buttonA: Bool, boost: Bool
  ) {
    let now = Date().timeIntervalSince1970
    guard now - lastInputLogTime > 0.05 else { return }
    lastInputLogTime = now

    let elementName = String(describing: type(of: element))
    let formattedLeft = String(format: "(%.2f, %.2f)", leftStick.x, leftStick.y)
    let formattedRight = String(format: "(%.2f, %.2f)", rightStick.x, rightStick.y)
    print(
      "[GameManager] Input \(elementName) left=\(formattedLeft) right=\(formattedRight) A=\(buttonA) boost=\(boost)"
    )
  }

  // MARK: - State Update

  /// Reset the player state (position and rotation)
  func resetState() {
    playerOffset = .zero
    yawAngle = 0
  }

  /// Update the player position and rotation based on current stick input.
  /// Movement is accumulated in viewer-local space (no yaw applied to displacement).
  /// GestureManager rotates to world space for shaders that need it (onScene=true).
  /// - Parameters:
  ///   - deltaTime: Time elapsed since last update (seconds).
  /// - Returns: Updated rig transform matrix.
  @discardableResult
  func updateRigState(deltaTime: Float) -> simd_float4x4 {
    controllerQueue.sync {
      let primaryStickInput = applyDeadZone(controllerState.leftStick)
      let secondaryStickInput = applyDeadZone(controllerState.rightStick)
      let movementMultiplier = controllerState.boostActive ? boostMovementMultiplier : 1.0
      let yawMultiplier = controllerState.boostActive ? boostYawMultiplier : 1.0

      let forwardInput = primaryStickInput.y  // left stick Y  → forward/back
      let yawInput = primaryStickInput.x  // left stick X  → turn
      let strafeInput = secondaryStickInput.x  // right stick X → strafe
      let verticalInput = secondaryStickInput.y  // right stick Y → up/down

      // Reduce turning speed slightly when actively turning
      let turnSpeedReduction = 1.0 - abs(yawInput) * 0.5
      yawAngle -= yawInput * yawSpeed * yawMultiplier * turnSpeedReduction * deltaTime
      yawAngle = wrapAngle(yawAngle)

      // Accumulate movement in world space using current yaw.
      // Shader rotation: x'=x·cosθ-z·sinθ, z'=x·sinθ+z·cosθ (standard CCW)
      // Viewer world-forward = (-sinθ, 0, -cosθ),  world-right = (cosθ, 0, -sinθ)
      let cosY = cos(yawAngle)
      let sinY = sin(yawAngle)
      let speed = movementSpeed * movementMultiplier * deltaTime
      playerOffset.x += (-sinY * forwardInput + cosY * strafeInput) * speed
      playerOffset.z += (-cosY * forwardInput - sinY * strafeInput) * speed
      playerOffset.y += verticalInput * speed

      return matrix_identity_float4x4  // rig transform not used in CompositorDemo
    }
  }

  // MARK: - Helpers

  private func applyDeadZone(_ input: SIMD2<Float>) -> SIMD2<Float> {
    let magnitude = simd_length(input)
    guard magnitude > deadZone else { return .zero }
    let scaled = (magnitude - deadZone) / (1 - deadZone)
    return (input / max(magnitude, 0.0001)) * scaled
  }

  private func wrapAngle(_ angle: Float) -> Float {
    var value = angle
    let twoPi: Float = .pi * 2
    value = fmod(value, twoPi)
    if value > .pi {
      value -= twoPi
    } else if value < -.pi {
      value += twoPi
    }
    return value
  }
}
