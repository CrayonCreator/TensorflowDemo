<template>
  <div class="home">
    <h1 class="title">GeGe Need</h1>
    <div class="button-container">
      <Button type="primary" size="large" @click="ToImageSegmentation1">Image Segmentation1</Button>
      <Button type="primary" size="large" @click="ToImageSegmentation2">Image Segmentation2</Button>
      <Button type="primary" size="large" @click="ToDevideToPieces">Devide To Pieces</Button>
      <Button type="primary" size="large" @click="ToFaceLandmarks">Face Landmarks Detection</Button>
    </div>
  </div>
</template>

<script setup lang="ts">
import Button from '../components/ui/Button.vue';

const ToImageSegmentation1 = () => {
  window.location.href = '/image-segmentation1';
}
const ToImageSegmentation2 = () => {
  window.location.href = '/image-segmentation2';
}

const ToDevideToPieces = () => {
  window.location.href = '/devide-to-pieces'
}

const ToFaceLandmarks = () => {
  window.location.href = '/face-landmarks'
}

defineOptions({
  name: 'HomePage'
});
</script>

<style scoped lang="less">
.home {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 20px;
}

.title {
  font-size: 32px;
  font-weight: 1000;
  color: #333333;
  margin-bottom: 40px;
  text-align: center;
}

.button-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
  width: 100%;
  max-width: 400px;

  @media (max-width: 768px) {
    max-width: 90%;
  }

  :deep(.btn) {
    background-color: #444444;
    color: white;

    &:hover {
      background-color: #333333;
    }
  }
}
</style>
