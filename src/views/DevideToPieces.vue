<template>
  <div class="app-container">
    <div class="title">Devide</div>
    <div class="toolPanel">
      <label class="custom-file-upload btn primary">
        <input type="file" accept="image/*" @change="handleFileUpload" />
        Select Image
      </label>
      <Button @click="SegmentateImage" :disabled="isLoading">
        {{ isLoading ? '分割中...' : 'Segmentation' }}
      </Button>
    </div>
    <div class="display">
      <!-- UploadedImage -->
      <div class="uploadedImage">
        <img v-if="selectedImage" :src="selectedImage" alt="Uploaded image" />
        <div v-else class="no-image-placeholder">Please select an image</div>
      </div>
      <!-- SementedImage -->
      <div class="sementedImage" style="position: relative;">
        <canvas v-if="isSegmentated" alt="Uploaded image" />
        <div v-else-if="!isLoading" class="no-image-placeholder">Please select an image and click the segment button
        </div>
        <LoadingSpinner v-if="isLoading" overlay size="large" text="正在分割图像，请稍候..." />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick } from 'vue';
import * as tf from '@tensorflow/tfjs';
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import Button from '../components/ui/Button.vue';
import LoadingSpinner from '../components/ui/LoadingSpinner.vue';

const isSegmentated = ref<Boolean>(false);
const selectedImage = ref<string | null>(null);
const isLoading = ref<boolean>(false);
const handleFileUpload = (event: Event) => {
  const input = event.target as HTMLInputElement;
  if (input.files && input.files.length > 0) {
    const file = input.files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
      selectedImage.value = e.target?.result as string;
      // Reset segmentation state
      isSegmentated.value = false;
      isLoading.value = false;
    };
    reader.readAsDataURL(file);
  }
}

// init backend
tf.setBackend('webgl').then(() => {
  console.log('TensorFlow backend initialized:', tf.getBackend());
}).catch((error: unknown) => {
  console.error('Error initializing TensorFlow backend:', error);
  return tf.setBackend('cpu');
}).then(() => {
  console.log('TensorFlow backend set to CPU');
}).catch((error: unknown) => {
  console.log(error);
})

const SegmentateImage = async () => {
  if (!selectedImage.value) {
    alert('请先选择一张图片');
    return;
  }

  isLoading.value = true;
  isSegmentated.value = false;

  try {
    const model = bodySegmentation.SupportedModels.BodyPix;
    const segmenterConfig = {
      architecture: 'ResNet50' as const,
      outputStride: 16,
      quantBytes: 4
    };
    const segmenter = await bodySegmentation.createSegmenter(model, segmenterConfig);
    const segmentationConfig = {
      multiSegmentation: true,
      flipHorizontal: false,
      segmentBodyParts: true,
      interResolution: 'full',
      refineSteps: 20
    }

    const img = document.querySelector('.uploadedImage img') as HTMLImageElement;

    const people = await segmenter.segmentPeople(img, segmentationConfig);

    // Set segmentation state to true so canvas is rendered
    isSegmentated.value = true;

    // Wait for the next DOM update to ensure canvas exists
    await nextTick();

    // console.log(people);
    // console.log(people[0].mask.getUnderlyingType());
    // console.log(people[0].mask.toCanvasImageSource());
    // console.log(people[0].mask.toImageData());
    // console.log(people[0].mask.toTensor());
    // console.log(people[0].maskValueToLabel());
    // Get the original image dimensions

    // Get the displayed dimensions of the image (might be scaled)
    const displayedWidth = img.width;
    const displayedHeight = img.height;

    // Set up the canvas with the same dimensions as the displayed image
    const canvas = document.querySelector('.sementedImage canvas') as HTMLCanvasElement;
    if (!canvas) return;
    canvas.width = displayedWidth;
    canvas.height = displayedHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(img, 0, 0, displayedWidth, displayedHeight);

    const segmentation = people[0];

    const bodyPartSegmentation = await segmentation.mask.toImageData();
    const maskData = bodyPartSegmentation.data;

    const originalImageData = ctx.getImageData(0, 0, displayedWidth, displayedHeight);
    const originalData = originalImageData.data;

    const resultImageData = ctx.createImageData(displayedWidth, displayedHeight);
    const resultData = resultImageData.data;

    const colorMap: { [key: number]: [number, number, number] } = {
      1: [255, 0, 0],      // Red - face
      2: [0, 255, 0],      // Green - left upper arm
      3: [0, 0, 255],      // Blue - right upper arm
      4: [255, 255, 0],    // Yellow - left lower arm
      5: [255, 0, 255],    // Magenta - right lower arm
      6: [0, 255, 255],    // Cyan - left hand
      7: [128, 0, 0],      // Maroon - right hand
      8: [0, 128, 0],      // Dark green - torso
      9: [0, 0, 128],      // Navy - left upper leg
      10: [128, 128, 0],   // Olive - right upper leg
      11: [128, 0, 128],   // Purple - left lower leg
      12: [0, 128, 128],   // Teal - right lower leg
      13: [128, 128, 128], // Gray - left foot
      14: [64, 0, 0],      // Dark red - right foot
      15: [0, 64, 0],      // Dark green 2 - hair
      16: [0, 0, 64],      // Dark blue - left eye
      17: [64, 64, 0],     // Dark yellow - right eye
      18: [64, 0, 64],     // Dark magenta - left ear
      19: [0, 64, 64],     // Dark cyan - right ear
      20: [192, 0, 0],     // Bright red - neck
      21: [0, 192, 0],     // Bright green - nose
      22: [0, 0, 192],     // Bright blue - mouth
      23: [192, 192, 0],   // Bright yellow - shoulders
      24: [192, 0, 192]    // Bright magenta - waist
    };

    for (let i = 0; i < maskData.length; i += 4) {
      const partId = maskData[i]; // 身体部位的ID

      if (partId > 0) { // 如果不是背景
        const [r, g, b] = colorMap[partId] || [255, 255, 255]; // 获取对应颜色或默认为白色

        resultData[i] = (originalData[i] + r) / 2;
        resultData[i + 1] = (originalData[i + 1] + g) / 2;
        resultData[i + 2] = (originalData[i + 2] + b) / 2;
        resultData[i + 3] = 150;
      } else {
        // 背景 - 设为透明
        resultData[i] = 0;
        resultData[i + 1] = 0;
        resultData[i + 2] = 0;
        resultData[i + 3] = 0; // 完全透明
      }
    }

    // 将处理后的图像绘制到canvas
    ctx.putImageData(resultImageData, 0, 0);
  } catch (error) {
    console.error('分割过程中发生错误:', error);
    alert('分割失败，请重试');
    isSegmentated.value = false;
  } finally {
    isLoading.value = false;
  }
}
</script>

<style scoped lang="less">
.app-container {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.title {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 20px;
  color: #333;
}

.toolPanel {
  display: flex;
  gap: 15px;
  margin-bottom: 20px;
  align-items: center;

  input[type="file"] {
    display: none;
  }

  .custom-file-upload {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0 16px;
    font-weight: 500;
    transition: background-color 0.2s, opacity 0.2s;
    font-size: 1rem;
    height: 40px;

    &:hover {
      background-color: #2980b9;
    }
  }

  .btn {
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    cursor: pointer;
    font-weight: 500;
    transition: background-color 0.2s, opacity 0.2s;
    font-size: 1rem;

    &:hover {
      background-color: #cfd0d1;
    }

    &.primary {
      background-color: #3498db;
      color: white;

      &:hover {
        background-color: #2980b9;
      }
    }
  }
}

.display {
  display: flex;
  gap: 20px;
  width: 100%;
  max-width: 1200px;
  flex: 1;

  @media (max-width: 768px) {
    flex-direction: column;
  }

  .uploadedImage,
  .sementedImage {
    flex: 1;
    min-height: 300px;
    border: 1px solid #ddd;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #f9f9f9;
    overflow: hidden;
    position: relative;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);

    img,
    canvas {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }

    .no-image-placeholder {
      color: #888;
      padding: 20px;
      text-align: center;
      font-size: 14px;
    }
  }
}
</style>
