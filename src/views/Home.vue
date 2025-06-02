<template>
  <div class="home">
    <Button @click="ToImageSegmentation1">Image Segmentation1</Button>
    <Button @click="ToImageSegmentation2">Image Segmentation2</Button>
    <Button @click="ToDevideToPieces">Devide To Pieces</Button>
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

defineOptions({
  name: 'HomePage'
});
</script>

<style scoped>
Button {
  margin-top: 20px;
}
</style>
