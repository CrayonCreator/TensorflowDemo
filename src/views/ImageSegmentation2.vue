<template>
  <div class="app-container">
    <div class="title">Body Pix</div>
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
      outputStride: 16 as const,
      quantBytes: 4 as const
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

    // First, draw the original image
    ctx.drawImage(img, 0, 0, displayedWidth, displayedHeight);

    // Then overlay the mask with transparency
    people[0].mask.toImageData().then((imageData) => {
      // Create a temporary canvas to process the mask
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = imageData.width;
      tempCanvas.height = imageData.height;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) return;

      // Put the mask data on the temporary canvas
      tempCtx.putImageData(imageData, 0, 0);

      // Draw the mask over the original image with transparency
      ctx.globalAlpha = 0.5; // 50% transparency
      ctx.globalCompositeOperation = 'source-atop'; // Only draw where original image exists
      ctx.drawImage(tempCanvas, 0, 0, displayedWidth, displayedHeight);

      // Reset canvas settings
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = 'source-over';

      // Set segmentation state to true after successful processing
      isSegmentated.value = true;
      isLoading.value = false;
    });

  } catch (error) {
    console.error('分割过程中发生错误:', error);
    alert('分割失败，请重试');
    isSegmentated.value = false;
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

