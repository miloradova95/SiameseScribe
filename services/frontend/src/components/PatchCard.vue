<template>
  <div class="patch-card">
    <img :src="fileUrl" :alt="patch.file_path" class="preview" />
    <div class="info">
      <p class="name">{{ patch.file_path }}</p>
      <p v-if="patch.group" class="group">Group: {{ patch.group }}</p>
      <p v-if="patch.codex" class="codex">Codex: {{ patch.codex }}</p>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { getPatchFileUrl } from '../services/patch-service'

const props = defineProps({
  patch: {
    type: Object,
    required: true,
  },
})

const fileUrl = computed(() => getPatchFileUrl(props.patch.id))
</script>

<style scoped>
.patch-card {
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

.group,
.codex {
  margin: 4px 0 0 0;
  font-size: 0.75rem;
  color: #666;
}
</style>
