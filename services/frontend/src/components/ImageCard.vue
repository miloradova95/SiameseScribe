<template>
  <div class="image-card">
    <img :src="fileUrl" :alt="image.fileName" class="preview" />
    <div class="info">
      <p class="name">{{ image.fileName }}</p>
      <p v-if="image.group" class="group">Group: {{ image.group }}</p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  image: {
    type: Object,
    required: true,
  },
})

const fileUrl = computed(() => `http://localhost:8000/images/${props.image.id}/file`)
</script>

<style scoped>
.image-card {
  border: 1px solid #ddd;
  border-radius: 8px;
  overflow: hidden;
  width: 220px;
  background: #fafafa;
}

.preview {
  width: 100%;
  height: 160px;
  object-fit: cover;
  display: block;
}

.info {
  padding: 8px 12px;
}

.name {
  margin: 0;
  font-weight: 600;
  font-size: 0.85rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.group {
  margin: 4px 0 0;
  font-size: 0.75rem;
  color: #666;
}

</style>
