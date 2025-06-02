<template>
  <div class="app-container">
    <div class="title">Body Segmentation</div>
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

const SegmentateImage = async () => {
  if (!selectedImage.value) {
    alert("请先选择一张图片");
    return;
  }

  isLoading.value = true;
  isSegmentated.value = true;

  // 等待DOM更新
  await nextTick();

  try {
    const img = document.querySelector('.uploadedImage img') as HTMLImageElement;

    if (!img.complete) {
      await new Promise(resolve => {
        img.onload = resolve;
      });
    }

    const segmenterConfig = {
      runtime: 'tfjs' as const,
      modelType: 'general' as const
    };

    const segmenter = await bodySegmentation.createSegmenter(
      bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation,
      segmenterConfig
    );

    console.log(segmenter);


    const segmentationCanvas = document.createElement('canvas');
    const container = document.querySelector('.sementedImage') as HTMLElement;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;

    const imgWidth = img.naturalWidth;
    const imgHeight = img.naturalHeight;

    const scale = Math.min(
      containerWidth / imgWidth,
      containerHeight / imgHeight
    );

    const scaledWidth = Math.floor(imgWidth * scale);
    const scaledHeight = Math.floor(imgHeight * scale);

    segmentationCanvas.width = scaledWidth;
    segmentationCanvas.height = scaledHeight;
    const segCtx = segmentationCanvas.getContext('2d');
    if (!segCtx) {
      console.error('Failed to get canvas context');
      return;
    }
    segCtx.drawImage(img, 0, 0, scaledWidth, scaledHeight);

    console.log('Segmenter created, now segmenting image...');
    const segmentation = await segmenter.segmentPeople(segmentationCanvas);
    console.log('Segmentation complete');
    console.log(segmentation);


    const coloredPartImage = await bodySegmentation.toBinaryMask(segmentation);
    console.log(coloredPartImage);


    const opacity = 0.7;
    const flipHorizontal = false;
    const maskBlurAmount = 3;

    const canvas = document.querySelector('.sementedImage canvas') as HTMLCanvasElement;
    canvas.width = scaledWidth;
    canvas.height = scaledHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Failed to get canvas context');
      return;
    }

    const scaledImgCanvas = document.createElement('canvas');
    scaledImgCanvas.width = scaledWidth;
    scaledImgCanvas.height = scaledHeight;
    const scaledImgCtx = scaledImgCanvas.getContext('2d');
    if (!scaledImgCtx) {
      console.error('Failed to get scaled image canvas context');
      return;
    }
    scaledImgCtx.drawImage(segmentationCanvas, 0, 0);

    if (coloredPartImage.width !== scaledWidth || coloredPartImage.height !== scaledHeight) {
      console.warn("changing mask size to match image size");

      const adjustedMaskData = new ImageData(scaledWidth, scaledHeight);

      const minWidth = Math.min(coloredPartImage.width, scaledWidth);
      const minHeight = Math.min(coloredPartImage.height, scaledHeight);

      for (let y = 0; y < minHeight; y++) {
        for (let x = 0; x < minWidth; x++) {
          const srcIdx = (y * coloredPartImage.width + x) * 4;
          const dstIdx = (y * scaledWidth + x) * 4;

          adjustedMaskData.data[dstIdx] = coloredPartImage.data[srcIdx];
          adjustedMaskData.data[dstIdx + 1] = coloredPartImage.data[srcIdx + 1];
          adjustedMaskData.data[dstIdx + 2] = coloredPartImage.data[srcIdx + 2];
          adjustedMaskData.data[dstIdx + 3] = coloredPartImage.data[srcIdx + 3];
        }
      }

      bodySegmentation.drawMask(
        canvas,
        scaledImgCanvas,
        adjustedMaskData,
        opacity,
        maskBlurAmount,
        flipHorizontal
      );
    } else {
      bodySegmentation.drawMask(
        canvas,
        scaledImgCanvas,
        coloredPartImage,
        opacity,
        maskBlurAmount,
        flipHorizontal
      );
    }

    isSegmentated.value = true;
  } catch (error) {
    console.error('分割过程中发生错误:', error);
    isSegmentated.value = false;
    alert('分割失败，请重试');
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
