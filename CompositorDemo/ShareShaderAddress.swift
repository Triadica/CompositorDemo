//
//  ShareShaderAddress.swift
//  CompositorDemo
//
//  Created by chen on 2025/7/11.
//  Copyright © 2025 Apple. All rights reserved.
//

import Combine  // 确保导入 Combine 框架
import Foundation

class SharedShaderAddress: ObservableObject {
  @Published var inputText: String = ""
}
