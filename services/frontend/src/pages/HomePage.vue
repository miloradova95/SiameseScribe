<template>
  <div class="home">
    <h1>Home</h1>
    <section class="image-section">
      <h2>Sample Image</h2>
      <div v-if="loading" class="state">Loading...</div>
      <div v-else-if="error" class="state error">{{ error }}</div>
      <template v-else-if="image">
        <p class="filepath">{{ image.filePath }}</p>
        <ImageCard :image="image" />
      </template>
    </section>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import ImageCard from '../components/ImageCard.vue'

const image = ref(null)
const loading = ref(false)
const error = ref(null)

onMounted(async () => {
  loading.value = true
  try {
    const res = await fetch('http://localhost:8000/images/1')
    if (!res.ok) throw new Error(`Image not found (${res.status})`)
    image.value = await res.json()
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.home {
  padding: 24px;
}

.image-section {
  margin-top: 24px;
}

h2 {
  margin-bottom: 12px;
}

.filepath {
  font-size: 0.8rem;
  color: #666;
  margin-bottom: 8px;
  word-break: break-all;
}

.state {
  color: #888;
}

.error {
  color: #c0392b;
}
</style>
