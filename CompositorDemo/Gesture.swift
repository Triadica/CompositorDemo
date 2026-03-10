/// implement basic moving and scaling gesture, however not simply following your finger.
/// it will still perform operation when you finger finished moving since it changes velocity.

import GameController
import QuartzCore
import RealityKit
import Spatial
import SwiftUI
import simd

private struct PinchHappen {
  var position: SIMD3<Float>
  var chirality: Chirality
}

class GestureManager {
  // MARK: - Gamepad support

  /// When true, movement and rotation are controlled by the connected gamepad.
  /// Spatial gesture control is disabled. Set to false to re-enable gesture control.
  var useGamepad: Bool = true
  private let _gameManager: GameManager = GameManager()
  private var _lastGamepadUpdate: TimeInterval = 0

  // MARK: - Gesture-controlled backing stores (used when useGamepad = false)

  /// update this with gesture events
  private var primaryStarted: PinchHappen? = nil
  private var secondaryStarted: PinchHappen? = nil

  private var _viewerPosition: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
  private var _viewerScale: Float = 1.0
  private var _viewerRotation: Float = 0.0

  /// Current viewer position — driven by gamepad (useGamepad=true) or gestures (useGamepad=false).
  /// - For onScene=false renderers (rotate-then-translate shaders): returns local-space offset directly.
  /// - For onScene=true renderers (translate-then-rotate shaders): rotates offset to world space.
  var viewerPosition: SIMD3<Float> {
    get {
      if useGamepad {
        updateGamepadIfNeeded()
        let local = _gameManager.playerOffset
        if onScene {
          // translate-then-rotate shaders (Lamps, Rain, Dome, Blocks):
          // playerOffset is already in world space → return as-is.
          return local
        } else {
          // rotate-then-translate shaders (SpreadAroundBall, BounceInBall, etc.):
          // Shader rotates scene by viewerRotation first, so viewerPosition must be
          // expressed in the post-rotation frame → rotate world offset by +yawAngle.
          let R = _gameManager.yawAngle
          let cosR = cos(R)
          let sinR = sin(R)
          return SIMD3<Float>(
            local.x * cosR - local.z * sinR,
            local.y,
            local.x * sinR + local.z * cosR
          )
        }
      }
      return _viewerPosition
    }
    set { _viewerPosition = newValue }
  }

  /// Current viewer scale — always 1.0 when useGamepad=true, gesture-controlled otherwise
  var viewerScale: Float {
    get { useGamepad ? 1.0 : _viewerScale }
    set { _viewerScale = newValue }
  }

  /// Current viewer rotation (yaw, radians) — driven by gamepad or gestures
  var viewerRotation: Float {
    get {
      if useGamepad {
        updateGamepadIfNeeded()
        return _gameManager.yawAngle
      }
      return _viewerRotation
    }
    set { _viewerRotation = newValue }
  }

  /// Reset gamepad player position and rotation to origin
  func resetGamepadState() {
    _gameManager.resetState()
  }

  private func updateGamepadIfNeeded() {
    let now = CACurrentMediaTime()
    guard _lastGamepadUpdate > 0 else {
      _lastGamepadUpdate = now
      return
    }
    let delta = Float(now - _lastGamepadUpdate)
    guard delta > 0.001 else { return }  // skip if <1ms since last update
    _lastGamepadUpdate = now
    _gameManager.updateRigState(deltaTime: delta)
  }

  /// initial length when the other chirality pinch started
  var pinchBaseLength: Float = 0.0
  /// initial angle when the other chirality pinch started
  var pinchBaseRadian: Float = 0.0

  /// compare with latest primary pinch position to be smoother
  var primaryPinchRealtimePosition: SIMD3<Float> = SIMD3<Float>(0, 0, 0)

  var onScene: Bool = false
  var gestureDirection: Float {
    if onScene {
      return -1.0
    } else {
      return 1.0
    }
  }

  init(onScene: Bool = false) {
    self.onScene = onScene
  }

  /// track the position pinch started, following pinches define the velocity of moving, to update self.viewerPosition .
  /// other other chirality events are used for scaling the entity
  func onSpatialEvent(event: SpatialEventCollection.Event) {
    // Gesture control is disabled when using gamepad
    guard !useGamepad else { return }

    guard let chirality = event.chirality,
      event.inputDevicePose?.pose3D != nil,
      event.inputDevicePose?.pose3D.position != nil,
      event.inputDevicePose?.pose3D.rotation != nil
    else {
      return
    }

    if let primaryStarted = self.primaryStarted {
      handleFollowingPinch(
        event: event,
        chirality: chirality,
        primaryPinch: primaryStarted
      )
    } else {
      handlePrimaryPinchStart(event: event, chirality: chirality)
    }

    if event.phase == .ended {
      self.secondaryStarted = nil
    }
  }

  private func handlePrimaryPinchStart(event: SpatialEventCollection.Event, chirality: Chirality) {
    if event.phase == .active {
      if let secondaryStarted = self.secondaryStarted {
        if secondaryStarted.chirality == chirality {
          // nothing
        } else {
          primaryStarted = PinchHappen(
            position: event.inputDevicePose!.pose3D.position.to_simd3,
            chirality: chirality
          )
          primaryPinchRealtimePosition = primaryStarted!.position
        }
      } else {
        primaryStarted = PinchHappen(
          position: event.inputDevicePose!.pose3D.position.to_simd3,
          chirality: chirality
        )
        primaryPinchRealtimePosition = primaryStarted!.position
      }
    }
  }

  private func handleFollowingPinch(
    event: SpatialEventCollection.Event,
    chirality: Chirality,
    primaryPinch: PinchHappen
  ) {

    if event.phase == .ended {
      if event.chirality == primaryPinch.chirality {
        self.primaryStarted = nil
        self.secondaryStarted = nil
      } else {
        self.primaryStarted = nil
        self.secondaryStarted = nil
      }
    } else if event.phase == .active {
      guard let pinchPosition = event.inputDevicePose?.pose3D.position.to_simd3 else {
        return
      }

      if event.chirality == primaryPinch.chirality {
        if secondaryStarted == nil {
          // update the viewer position
          var delta = pinchPosition - primaryPinch.position

          if self.gestureDirection < 0 {
            // on scene, we need rotate the delta vector
            let rotation: Float = -viewerRotation
            let cosRadian = cos(rotation)
            let sinRadian = sin(rotation)
            /// make new delta since we rotate the world viewer
            delta = SIMD3<Float>(
              delta.x * cosRadian - delta.z * sinRadian,
              delta.y,
              delta.x * sinRadian + delta.z * cosRadian
            )
          }
          self.viewerPosition -= delta * 0.1 * gestureDirection
        }
        primaryPinchRealtimePosition = pinchPosition
      } else {
        let realtimeP1 = primaryPinchRealtimePosition
        let pinchDelta = simd_distance(pinchPosition, realtimeP1)
        let pinchRadian = atan2(
          pinchPosition.z - realtimeP1.z, pinchPosition.x - realtimeP1.x)
        if let secondaryStarted = secondaryStarted {

          let pinchAt2 = SIMD2(pinchPosition.x, pinchPosition.z)
          let startAt2 = SIMD2(realtimeP1.x, realtimeP1.z)
          let secondaryStart2 = SIMD2(secondaryStarted.position.x, secondaryStarted.position.z)

          let secondaryDirection = simd_normalize(pinchAt2 - secondaryStart2)
          let secondaryArmDirection = simd_normalize(startAt2 - secondaryStart2)
          let guessScaleOrRotate = abs(simd_dot(secondaryDirection, secondaryArmDirection))

          if guessScaleOrRotate > 0.8 {

            let ratio: Float = pow(pinchDelta / pinchBaseLength, 0.2)
            self.viewerScale *= ratio
          } else if guessScaleOrRotate < 0.7 {

            var deltaRadian = pinchRadian - pinchBaseRadian
            if deltaRadian > .pi {
              deltaRadian -= 2 * .pi
            } else if deltaRadian < -.pi {
              deltaRadian += 2 * .pi
            }
            self.viewerRotation += deltaRadian * 0.02 * gestureDirection
          }
        } else {
          pinchBaseLength = pinchDelta
          pinchBaseRadian = pinchRadian
          secondaryStarted = PinchHappen(
            position: pinchPosition,
            chirality: event.chirality!
          )

        }

      }
    }
  }
}

extension Point3D {
  /// turn into SIMD3
  fileprivate var to_simd3: SIMD3<Float> {
    return SIMD3<Float>(Float(x), Float(y), Float(z))
  }
}
