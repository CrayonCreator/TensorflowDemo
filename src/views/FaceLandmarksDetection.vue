<template>
  <div class="face-landmarks-container">
    <div class="header">
      <h1>人脸关键点检测</h1>
      <p>使用 TensorFlow.js MediaPipe FaceMesh 模型进行实时人脸关键点检测</p>
    </div>

    <div class="controls">
      <div class="control-group">
        <label>
          <input type="checkbox" v-model="showBoundingBox" />
          显示边界框
        </label>
      </div>
      <div class="control-group">
        <label>
          <input type="checkbox" v-model="showTriangulation" />
          显示三角网格
        </label>
      </div>
      <div class="control-group">
        <label>
          <input type="checkbox" v-model="showKeypoints" />
          显示关键点
        </label>
      </div>
      <div class="control-group">
        <button @click="toggleCamera" :disabled="isLoading">
          {{ isRunning ? '停止摄像头' : '启动摄像头' }}
        </button>
      </div>
    </div>

    <div class="status" v-if="status">
      {{ status }}
    </div>

    <div class="canvas-wrapper" v-show="isRunning">
      <canvas ref="canvasRef" id="output"
        style="border: 2px solid #ddd; border-radius: 8px; max-width: 100%; height: auto; z-index: 10; background: transparent;"></canvas>
      <video ref="videoRef" id="video" playsinline autoplay muted style="
          -webkit-transform: scaleX(-1);
          transform: scaleX(-1);
          visibility: hidden;
          width: auto;
          height: auto;
          position: absolute;
          z-index: -1;
        "></video>
    </div>

    <div class="stats" v-if="fps">
      <p>FPS: {{ fps.toFixed(1) }}</p>
      <p>检测到的人脸数量: {{ faceCount }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection'
import '@tensorflow/tfjs-backend-webgl'
import '@tensorflow/tfjs-backend-wasm'

// 引用
const videoRef = ref<HTMLVideoElement>()
const canvasRef = ref<HTMLCanvasElement>()

// 状态
const isLoading = ref(false)
const isRunning = ref(false)
const status = ref('')
const fps = ref(0)
const faceCount = ref(0)

// 控制选项
const showBoundingBox = ref(true)
const showTriangulation = ref(true)
const showKeypoints = ref(false)

// 检测器和流
let detector: faceLandmarksDetection.FaceLandmarksDetector | null = null
let stream: MediaStream | null = null
let animationId: number | null = null
let ctx: CanvasRenderingContext2D | null = null

// 性能统计
let lastTime = 0
let frameCount = 0

// 颜色配置（增强可见性）
const COLORS = {
  boundingBox: '#FF0000',      // 红色 - 边界框
  triangulation: '#00FFFF',    // 青色 - 三角网格（更亮）
  keypoints: '#FFFF00',        // 黄色 - 关键点（更亮）
  lips: '#FF69B4',             // 粉色 - 嘴唇
  leftEye: '#00FF00',          // 绿色 - 左眼
  leftEyebrow: '#00FF00',      // 绿色 - 左眉毛
  leftIris: '#00FF00',         // 绿色 - 左虹膜
  rightEye: '#FF0000',         // 红色 - 右眼
  rightEyebrow: '#FF0000',     // 红色 - 右眉毛
  rightIris: '#FF0000',        // 红色 - 右虹膜
  faceOval: '#FFFFFF',         // 白色 - 面部轮廓
}

// MediaPipe FaceMesh 完整三角剖分数据（来自官方demo）
const TRIANGULATION = [
  127, 34, 139, 11, 0, 37, 232, 231, 120, 72, 37, 39, 128, 121, 47, 232, 121,
  128, 104, 69, 67, 175, 171, 148, 157, 154, 155, 118, 50, 101, 73, 39, 40, 9,
  151, 108, 48, 115, 131, 194, 204, 211, 74, 40, 185, 80, 42, 183, 40, 92,
  186, 230, 229, 118, 202, 212, 214, 83, 18, 17, 76, 61, 146, 160, 29, 30, 56,
  157, 173, 106, 204, 194, 135, 214, 192, 203, 165, 98, 21, 71, 68, 51, 45, 4,
  144, 24, 23, 77, 146, 91, 205, 50, 187, 201, 200, 18, 91, 106, 182, 90, 91,
  181, 85, 84, 17, 206, 203, 36, 148, 171, 140, 92, 40, 39, 193, 189, 244,
  159, 158, 28, 247, 246, 161, 236, 3, 196, 54, 68, 104, 193, 168, 8, 117,
  228, 31, 189, 193, 55, 98, 97, 99, 126, 47, 100, 166, 79, 218, 155, 154, 26,
  209, 49, 131, 135, 136, 150, 47, 126, 217, 223, 52, 53, 45, 51, 134, 211,
  170, 140, 67, 69, 108, 43, 106, 91, 230, 119, 120, 226, 130, 247, 63, 53,
  52, 238, 20, 242, 46, 70, 156, 78, 62, 96, 46, 53, 63, 143, 34, 227, 173,
  155, 133, 123, 117, 111, 44, 125, 19, 236, 134, 51, 216, 206, 205, 154, 153,
  22, 39, 37, 167, 200, 201, 208, 36, 142, 100, 57, 212, 202, 20, 60, 99, 28,
  158, 157, 35, 226, 113, 160, 159, 27, 204, 202, 210, 113, 225, 46, 43, 202,
  204, 62, 76, 77, 137, 123, 116, 41, 38, 72, 203, 129, 142, 64, 98, 240, 49,
  102, 64, 41, 73, 74, 212, 216, 207, 42, 74, 184, 169, 170, 211, 170, 149,
  176, 105, 66, 69, 122, 6, 168, 123, 147, 187, 96, 77, 90, 65, 55, 107, 89,
  90, 180, 101, 100, 120, 63, 105, 104, 93, 137, 227, 15, 86, 85, 129, 102,
  49, 14, 87, 86, 55, 8, 9, 100, 47, 121, 145, 23, 22, 88, 89, 179, 6, 122,
  196, 88, 95, 96, 138, 172, 136, 215, 58, 172, 115, 48, 219, 42, 80, 81, 195,
  3, 51, 43, 146, 61, 171, 175, 199, 81, 82, 38, 53, 46, 225, 144, 163, 110,
  246, 33, 7, 52, 65, 66, 229, 228, 117, 34, 127, 234, 107, 108, 69, 109, 108,
  151, 48, 64, 235, 62, 78, 191, 129, 209, 126, 111, 35, 143, 163, 161, 246,
  117, 123, 50, 222, 65, 52, 19, 125, 141, 221, 55, 65, 3, 195, 197, 25, 7,
  33, 220, 237, 44, 70, 71, 139, 122, 193, 245, 247, 130, 33, 71, 21, 162,
  153, 158, 159, 170, 169, 150, 188, 174, 196, 216, 186, 92, 144, 160, 161, 2,
  97, 167, 141, 125, 241, 164, 167, 37, 72, 38, 12, 145, 159, 160, 38, 82, 13,
  63, 68, 71, 226, 35, 111, 158, 153, 154, 101, 50, 205, 206, 92, 165, 209,
  198, 217, 165, 167, 97, 220, 115, 218, 133, 112, 243, 239, 238, 241, 214,
  135, 169, 190, 173, 133, 171, 208, 32, 125, 44, 237, 86, 87, 178, 85, 86,
  179, 84, 85, 180, 83, 84, 181, 201, 83, 182, 137, 93, 132, 76, 62, 183, 61,
  76, 184, 57, 61, 185, 212, 57, 186, 214, 207, 187, 34, 143, 156, 79, 239,
  237, 123, 137, 177, 44, 1, 4, 201, 194, 32, 64, 102, 129, 213, 215, 138, 59,
  166, 219, 242, 99, 97, 2, 94, 141, 75, 59, 235, 24, 110, 228, 25, 130, 226,
  23, 24, 229, 22, 23, 230, 26, 22, 231, 112, 26, 232, 189, 190, 243, 221, 56,
  190, 28, 56, 221, 27, 28, 222, 29, 27, 223, 30, 29, 224, 247, 30, 225, 238,
  79, 20, 166, 59, 75, 60, 75, 240, 147, 177, 215, 20, 79, 166, 187, 147, 213,
  112, 233, 244, 233, 128, 245, 128, 114, 188, 114, 217, 174, 131, 115, 220,
  217, 198, 236, 198, 131, 134, 177, 132, 58, 143, 35, 124, 110, 163, 7, 228,
  110, 25, 356, 389, 368, 11, 302, 267, 452, 350, 349, 302, 303, 269, 357,
  343, 277, 452, 453, 357, 333, 332, 297, 175, 152, 377, 384, 398, 382, 347,
  348, 330, 303, 304, 270, 9, 336, 337, 278, 279, 360, 418, 262, 431, 304,
  408, 409, 310, 415, 407, 270, 409, 410, 450, 348, 347, 422, 430, 434, 313,
  314, 17, 306, 307, 375, 387, 388, 260, 286, 414, 398, 335, 406, 418, 364,
  367, 416, 423, 358, 327, 251, 284, 298, 281, 5, 4, 373, 374, 253, 307, 320,
  321, 425, 427, 411, 421, 313, 18, 321, 405, 406, 320, 404, 405, 315, 16, 17,
  426, 425, 266, 377, 400, 369, 322, 391, 269, 417, 465, 464, 386, 257, 258,
  466, 260, 388, 456, 399, 419, 284, 332, 333, 417, 285, 8, 346, 340, 261,
  413, 441, 285, 327, 460, 328, 355, 371, 329, 392, 439, 438, 382, 341, 256,
  429, 420, 360, 364, 394, 379, 277, 343, 437, 443, 444, 283, 275, 440, 363,
  431, 262, 369, 297, 338, 337, 273, 375, 321, 450, 451, 349, 446, 342, 467,
  293, 334, 282, 458, 461, 462, 276, 353, 383, 308, 324, 325, 276, 300, 293,
  372, 345, 447, 382, 398, 362, 352, 345, 340, 274, 1, 19, 456, 248, 281, 436,
  427, 425, 381, 256, 252, 269, 391, 393, 200, 199, 428, 266, 330, 329, 287,
  273, 422, 250, 462, 328, 258, 286, 384, 265, 353, 342, 387, 259, 257, 424,
  431, 430, 342, 353, 276, 273, 335, 424, 292, 325, 307, 366, 447, 345, 271,
  303, 302, 423, 266, 371, 294, 455, 460, 279, 278, 294, 271, 272, 304, 432,
  434, 427, 272, 407, 408, 394, 430, 431, 395, 369, 400, 334, 333, 299, 351,
  417, 168, 352, 280, 411, 325, 319, 320, 295, 296, 336, 319, 403, 404, 330,
  348, 349, 293, 298, 333, 323, 454, 447, 15, 16, 315, 358, 429, 279, 14, 15,
  316, 285, 336, 9, 329, 349, 350, 374, 380, 252, 318, 402, 403, 6, 197, 419,
  318, 319, 325, 367, 364, 365, 435, 367, 397, 344, 438, 439, 272, 271, 311,
  195, 5, 281, 273, 287, 291, 396, 428, 199, 311, 271, 268, 283, 444, 445,
  373, 254, 339, 263, 466, 249, 282, 334, 296, 449, 347, 346, 264, 447, 454,
  336, 296, 299, 338, 10, 151, 278, 439, 455, 292, 407, 415, 358, 371, 355,
  340, 345, 372, 390, 249, 466, 346, 347, 280, 442, 443, 282, 19, 94, 370,
  441, 442, 295, 248, 419, 197, 263, 255, 359, 440, 275, 274, 300, 383, 368,
  351, 412, 465, 263, 467, 466, 301, 368, 389, 380, 374, 386, 395, 378, 379,
  412, 351, 419, 436, 426, 322, 373, 390, 388, 2, 164, 393, 370, 462, 461,
  164, 0, 267, 302, 11, 12, 374, 373, 387, 268, 12, 13, 293, 300, 301, 446,
  261, 340, 385, 384, 381, 330, 266, 425, 426, 423, 391, 429, 355, 437, 391,
  327, 326, 440, 457, 438, 341, 382, 362, 459, 457, 461, 434, 430, 394, 414,
  463, 362, 396, 369, 262, 354, 461, 457, 316, 403, 402, 315, 404, 403, 314,
  405, 404, 313, 406, 405, 421, 418, 406, 366, 401, 361, 306, 408, 407, 291,
  409, 408, 287, 410, 409, 432, 436, 410, 434, 416, 411, 264, 368, 383, 309,
  438, 457, 352, 376, 401, 274, 275, 4, 421, 428, 262, 294, 327, 358, 433,
  416, 367, 289, 455, 439, 462, 370, 326, 2, 326, 370, 305, 460, 455, 254,
  449, 448, 255, 261, 446, 253, 450, 449, 252, 451, 450, 256, 452, 451, 341,
  453, 452, 413, 464, 463, 441, 413, 414, 258, 442, 441, 257, 443, 442, 259,
  444, 443, 260, 445, 444, 467, 342, 445, 459, 458, 250, 289, 392, 290, 290,
  328, 460, 376, 433, 435, 250, 290, 392, 411, 416, 433, 341, 463, 464, 453,
  464, 465, 357, 465, 412, 343, 412, 399, 360, 363, 440, 437, 399, 456, 420,
  456, 363, 401, 435, 288, 372, 383, 294, 456, 420, 429, 358, 371, 355, 340,
  345, 372, 390, 249, 466, 346, 347, 280, 442, 443, 282, 19, 94, 370, 441,
  442, 295, 248, 419, 197, 263, 255, 359, 440, 275, 274, 300, 383, 368, 351,
  412, 465, 263, 467, 466, 301, 368, 389, 380, 374, 386, 395, 378, 379, 412,
  351, 419, 436, 426, 322, 373, 390, 388, 2, 164, 393, 370, 462, 461, 164, 0,
  267, 302, 11, 12, 374, 373, 387, 268, 12, 13, 293, 300, 301, 446, 261, 340,
  385, 384, 381, 330, 266, 425, 426, 423, 391, 429, 355, 437, 391, 327, 326,
  440, 457, 438, 341, 382, 362, 459, 457, 461, 434, 430, 394, 414, 463, 362,
  396, 369, 262, 354, 461, 457, 316, 403, 402, 315, 404, 403, 314, 405, 404,
  313, 406, 405, 421, 418, 406, 366, 401, 361, 306, 408, 407, 291, 409, 408,
  287, 410, 409, 432, 436, 410, 434, 416, 411, 264, 368, 383, 309, 438, 457,
  352, 376, 401, 274, 275, 4, 421, 428, 262, 294, 327, 358, 433, 416, 367,
  289, 455, 439, 462, 370, 326, 2, 326, 370, 305, 460, 455, 254, 449, 448,
  255, 261, 446, 253, 450, 449, 252, 451, 450, 256, 452, 451, 341, 453, 452,
  413, 464, 463, 441, 413, 414, 258, 442, 441, 257, 443, 442, 259, 444, 443,
  260, 445, 444, 467, 342, 445, 459, 458, 250, 289, 392, 290, 290, 328, 460,
  376, 433, 435, 250, 290, 392, 411, 416, 433, 341, 463, 464, 453, 464, 465,
  357, 465, 412, 343, 412, 399, 360, 363, 440, 437, 399, 456, 420, 456, 363
]

// 初始化检测器
async function initDetector() {
  try {
    status.value = '正在初始化模型...'

    detector = await faceLandmarksDetection.createDetector(
      faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
      {
        runtime: 'tfjs',
        refineLandmarks: true,
        maxFaces: 2,
      }
    )

    status.value = '模型初始化完成'
  } catch (error) {
    console.error('初始化检测器失败:', error)
    status.value = '模型初始化失败'
  }
}

// 设置摄像头
async function setupCamera() {
  try {
    const video = videoRef.value
    const canvas = canvasRef.value

    if (!video || !canvas) return false

    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 640,
        height: 480,
        facingMode: 'user'
      },
      audio: false
    })

    video.srcObject = stream

    await new Promise((resolve) => {
      video.onloadedmetadata = () => resolve(video)
    })

    video.play()

    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight

    video.width = videoWidth
    video.height = videoHeight
    canvas.width = videoWidth
    canvas.height = videoHeight

    ctx = canvas.getContext('2d')

    return true
  } catch (error) {
    console.error('摄像头设置失败:', error)
    status.value = '无法访问摄像头'
    return false
  }
}

// 绘制路径
function drawPath(ctx: CanvasRenderingContext2D, points: number[][], closePath: boolean) {
  const region = new Path2D()
  region.moveTo(points[0][0], points[0][1])
  for (let i = 1; i < points.length; i++) {
    const point = points[i]
    region.lineTo(point[0], point[1])
  }

  if (closePath) {
    region.closePath()
  }
  ctx.stroke(region)
}

// 绘制检测结果
function drawResults(faces: faceLandmarksDetection.Face[]) {
  if (!ctx) {
    console.error('Canvas context not available');
    return;
  }

  faces.forEach((face) => {
    const keypoints = face.keypoints.map((keypoint) => [
      canvasRef.value!.width - keypoint.x,  // x轴镜像反转
      keypoint.y
    ])

    // 绘制边界框
    if (showBoundingBox.value && face.box && ctx) {
      ctx.strokeStyle = COLORS.boundingBox
      ctx.lineWidth = 6

      const box = face.box
      // 对边界框坐标也进行x轴镜像
      const mirroredBox = {
        xMin: canvasRef.value!.width - box.xMax,
        xMax: canvasRef.value!.width - box.xMin,
        yMin: box.yMin,
        yMax: box.yMax
      }

      ctx.beginPath()
      ctx.rect(mirroredBox.xMin, mirroredBox.yMin, mirroredBox.xMax - mirroredBox.xMin, mirroredBox.yMax - mirroredBox.yMin)
      ctx.stroke()
    }

    // 绘制三角网格
    if (showTriangulation.value && ctx) {
      ctx.strokeStyle = COLORS.triangulation
      ctx.lineWidth = 2

      for (let i = 0; i < TRIANGULATION.length / 3; i++) {
        const points = [
          TRIANGULATION[i * 3],
          TRIANGULATION[i * 3 + 1],
          TRIANGULATION[i * 3 + 2],
        ].map((index) => keypoints[index])

        if (points[0] && points[1] && points[2]) {
          ctx.beginPath()
          ctx.moveTo(points[0][0], points[0][1])
          ctx.lineTo(points[1][0], points[1][1])
          ctx.lineTo(points[2][0], points[2][1])
          ctx.closePath()
          ctx.stroke()
        }
      }
    }

    // 绘制关键点
    if (showKeypoints.value && ctx) {
      ctx.fillStyle = COLORS.keypoints
      for (const [x, y] of keypoints) {
        ctx.beginPath()
        ctx.arc(x, y, 4, 0, 2 * Math.PI)
        ctx.fill()
      }
    }

    // 绘制面部轮廓
    if (face.keypoints.length >= 468 && ctx) {
      // 获取面部轮廓索引
      const contours = faceLandmarksDetection.util.getKeypointIndexByContour(
        faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh
      )

      for (const [label, contour] of Object.entries(contours)) {
        ctx.strokeStyle = COLORS[label as keyof typeof COLORS] || COLORS.faceOval
        ctx.lineWidth = 3

        const path = contour.map((index: number) => keypoints[index])
        if (path.every(value => value != undefined)) {
          drawPath(ctx, path, false)
        }
      }
    }
  })
}

// 渲染循环
async function renderPrediction() {
  if (!isRunning.value || !detector || !videoRef.value || !canvasRef.value || !ctx) {
    return
  }

  const video = videoRef.value

  if (video.readyState < 2) {
    animationId = requestAnimationFrame(renderPrediction)
    return
  }

  try {
    // 先进行人脸检测
    const faces = await detector.estimateFaces(video, { flipHorizontal: false })

    // 清空画布
    ctx.clearRect(0, 0, canvasRef.value!.width, canvasRef.value!.height)

    // 绘制镜像的视频帧
    ctx.save()
    ctx.scale(-1, 1)
    ctx.drawImage(video, -canvasRef.value!.width, 0, canvasRef.value!.width, canvasRef.value!.height)
    ctx.restore()

    faceCount.value = faces.length

    // 绘制检测结果（在视频帧之后）
    if (faces.length > 0) {
      drawResults(faces)
    }

    // 计算FPS
    const currentTime = performance.now()
    frameCount++
    if (currentTime - lastTime >= 1000) {
      fps.value = (frameCount * 1000) / (currentTime - lastTime)
      frameCount = 0
      lastTime = currentTime
    }
  } catch (error) {
    console.error('检测过程中出错:', error)
  }

  animationId = requestAnimationFrame(renderPrediction)
}

// 启动/停止摄像头
async function toggleCamera() {
  if (isRunning.value) {
    // 停止
    isRunning.value = false
    if (animationId) {
      cancelAnimationFrame(animationId)
      animationId = null
    }
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      stream = null
    }
    status.value = '摄像头已停止'
  } else {
    // 启动
    if (!detector) {
      await initDetector()
    }

    if (!detector) {
      status.value = '检测器初始化失败'
      return
    }

    isLoading.value = true
    status.value = '正在启动摄像头...'

    const success = await setupCamera()

    if (success) {
      isRunning.value = true
      status.value = '检测中...'
      lastTime = performance.now()
      frameCount = 0
      await nextTick()
      renderPrediction()
    }

    isLoading.value = false
  }
}

onMounted(async () => {
  // 初始化检测器
  await initDetector()
})

onUnmounted(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
  if (stream) {
    stream.getTracks().forEach(track => track.stop())
  }
  if (detector) {
    detector.dispose()
  }
})
</script>

<style scoped>
.face-landmarks-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h1 {
  color: #333;
  margin-bottom: 10px;
}

.header p {
  color: #666;
  font-size: 14px;
}

.controls {
  display: flex;
  gap: 20px;
  justify-content: center;
  align-items: center;
  margin-bottom: 20px;
  flex-wrap: wrap;
}

.control-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.control-group label {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 14px;
  cursor: pointer;
}

.control-group button {
  padding: 8px 16px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.control-group button:hover:not(:disabled) {
  background: #0056b3;
}

.control-group button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.status {
  text-align: center;
  margin: 20px 0;
  padding: 10px;
  background: #f8f9fa;
  border-radius: 4px;
  color: #495057;
}

.canvas-wrapper {
  position: relative;
  display: flex;
  justify-content: center;
  margin: 20px 0;
}

#output {
  border: 2px solid #ddd;
  border-radius: 8px;
  max-width: 100%;
  height: auto;
}

.stats {
  display: flex;
  justify-content: center;
  gap: 30px;
  margin-top: 20px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
}

.stats p {
  margin: 0;
  font-weight: bold;
  color: #495057;
}

@media (max-width: 768px) {
  .controls {
    flex-direction: column;
    gap: 15px;
  }

  .stats {
    flex-direction: column;
    gap: 10px;
    text-align: center;
  }
}
</style>
