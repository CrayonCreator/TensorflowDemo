<script setup lang="ts">
interface Props {
  size?: 'small' | 'medium' | 'large'
  color?: string
  overlay?: boolean
  text?: string
}

withDefaults(defineProps<Props>(), {
  size: 'medium',
  color: '#3498db',
  overlay: false,
  text: ''
})
</script>

<template>
  <div 
    class="loading-container" 
    :class="{ 'overlay': overlay }"
  >
    <div class="spinner-wrapper">
      <div 
        class="spinner" 
        :class="size"
        :style="{ borderTopColor: color }"
      ></div>
      <p v-if="text" class="loading-text">{{ text }}</p>
    </div>
  </div>
</template>

<style scoped>
.loading-container {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 60px;
}

.loading-container.overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(2px);
  z-index: 1000;
}

.spinner-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.spinner {
  border: 3px solid #f3f3f3;
  border-top: 3px solid #3498db;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.spinner.small {
  width: 20px;
  height: 20px;
  border-width: 2px;
}

.spinner.medium {
  width: 40px;
  height: 40px;
  border-width: 3px;
}

.spinner.large {
  width: 60px;
  height: 60px;
  border-width: 4px;
}

.loading-text {
  margin: 0;
  font-size: 14px;
  color: #666;
  font-weight: 500;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
</style>
