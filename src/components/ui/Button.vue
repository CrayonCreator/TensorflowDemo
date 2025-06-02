<script setup lang="ts">
defineProps<{
  type?: 'primary' | 'secondary' | 'danger' | 'success'
  size?: 'small' | 'medium' | 'large'
  disabled?: boolean
}>();

const emit = defineEmits<{
  click: [event: MouseEvent]
}>();
</script>

<template>
  <button
    class="btn"
    :class="[type, size]"
    :disabled="disabled"
    @click="(event) => emit('click', event)"
  >
    <slot></slot>
  </button>
</template>

<style scoped>
.btn {
  border: none;
  border-radius: 4px;
  padding: 0 16px;
  cursor: pointer;
  font-weight: 500;
  transition: background-color 0.2s, opacity 0.2s;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.btn:hover {
  background-color: #cfd0d1;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.primary {
  background-color: #3498db;
  color: white;
}

.secondary {
  background-color: #95a5a6;
  color: white;
}

.danger {
  background-color: #e74c3c;
  color: white;
}

.success {
  background-color: #2ecc71;
  color: white;
}

.small {
  font-size: 0.8rem;
  height: 32px;
  padding: 4px 8px;
}

.medium {
  font-size: 1rem;
}

.large {
  font-size: 1.2rem;
  height: 48px;
  padding: 12px 24px;
}
</style>
