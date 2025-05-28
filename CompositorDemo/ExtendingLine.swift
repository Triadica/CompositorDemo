import SwiftUI

/// a line during extending tracks last point, new points that are too closer are skipped
/// if line is stable, then all points are in the list `stablePoints`
struct ExtendingLine {
  private var stablePoints: [Point3D] = []
  private var lastPoint: Point3D? = .none
  var miniSkip: Double = 0.004
  var color: SIMD3<Float> = SIMD3<Float>(0.5, 0.5, 0.5)

  init(miniSkip: Double = 0.004) {
    self.miniSkip = miniSkip
  }

  var count: Int {
    if lastPoint != nil {
      return stablePoints.count + 1
    } else {
      return stablePoints.count
    }
  }

  /// if point hat
  mutating func addPoint(_ point: Point3D) {
    if let lastP = lastPoint {
      let distance = lastP.distance(to: point)
      if distance > miniSkip {
        stablePoints.append(lastP)
        lastPoint = point
      }
    } else {
      lastPoint = point
    }
  }

  func getPointAt(_ index: Int) -> Point3D {
    if index < stablePoints.count {
      return stablePoints[index]
    } else if index == stablePoints.count {
      if let lastP = lastPoint {
        return lastP
      } else {
        fatalError("No last point")
      }
    } else {
      fatalError("Index out of bounds")
    }
  }

  mutating func stabilize() {
    if let lastP = lastPoint {
      stablePoints.append(lastP)
      lastPoint = nil
    }
  }

  mutating func isStable() -> Bool {
    if let lastP: Point3D = lastPoint {
      return stablePoints.contains { $0.distance(to: lastP) < miniSkip }
    }
    return false
  }

  mutating func random_color() {
    color = SIMD3<Float>(
      Float.random(in: 0.0...1.0),
      Float.random(in: 0.0...1.0),
      Float.random(in: 0.0...1.0)
    )
  }
}

struct LinesManager {
  private var lines: [ExtendingLine] = []
  var maxLines: Int = 100
  var miniSkip: Double = 0.004
  private var currentLine: ExtendingLine = ExtendingLine()

  init(miniSkip: Double = 0.004) {
    self.miniSkip = miniSkip
  }

  mutating func addPoint(_ point: Point3D) {
    if lines.count < maxLines {
      currentLine.addPoint(point)
    } else {
      print("Max lines reached")
    }
  }

  mutating func finishCurrent() {
    currentLine.stabilize()
    lines.append(currentLine)
    currentLine = ExtendingLine(miniSkip: miniSkip)
    currentLine.random_color()
  }

  var count: Int {
    lines.count + 1
  }

  func getLineAt(_ index: Int) -> ExtendingLine {
    if index < lines.count {
      return lines[index]
    } else if index == lines.count {
      return currentLine
    } else {
      fatalError("Index out of bounds")
    }
  }

  /// remove the last line
  mutating func removeLastLine() {
    if lines.count > 0 {
      lines.removeLast()
    }
  }
}

func randomPosition(x: Float, y: Float, z: Float) -> SIMD3<Float> {
  return SIMD3<Float>(
    Float.random(in: -x...x),
    Float.random(in: -y...y),
    Float.random(in: -z...z)
  )
}

/// generate random position on a sphere, with fibonacci grid
func randomSpherePosition(radius: Float) -> SIMD3<Float> {
  let phi = Float(acos(1 - 2 * Float.random(in: 0...1)))
  let theta = Float(2 * .pi * Float.random(in: 0...1))
  let x = radius * sin(phi) * cos(theta)
  let y = radius * sin(phi) * sin(theta)
  let z = radius * cos(phi)
  return SIMD3<Float>(x, y, z)
}
