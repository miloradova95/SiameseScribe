<template>
  <div class="browse-page">
    <h1>Overview of all Uploads</h1>

    <button @click="loadRandom" :disabled="loading">
      {{ loading ? 'Loading...' : 'Random Image + Patches' }}
    </button>

    <div v-if="image" class="result">
      <div class="image-section">
        <h2>{{ image.fileName }}</h2>
        <img :src="`http://localhost:8000/images/${image.id}/file`" :alt="image.fileName" class="main-image" />
        <p class="meta">Group: {{ image.group }}</p>
      </div>

      <div class="patches-section">
        <h3>First 4 Patches</h3>
        <div class="patches-grid">
          <div v-for="patch in patches" :key="patch.id" class="patch-card">
            <img :src="`http://localhost:8000/patches/${patch.id}/file`" :alt="`Patch ${patch.id}`" class="patch-image" />
            <p class="patch-meta">x={{ patch.bbox.x }}, y={{ patch.bbox.y }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const image = ref(null)
const patches = ref([])
const loading = ref(false)

async function loadRandom() {
  loading.value = true
  image.value = null
  patches.value = []
  try {
    const imgRes = await fetch('http://localhost:8000/images/random')
    if (!imgRes.ok) throw new Error('Failed to fetch random image')
    const img = await imgRes.json()

    const patchRes = await fetch(`http://localhost:8000/images/${img.id}/patches`)
    if (!patchRes.ok) throw new Error('Failed to fetch patches')
    const allPatches = await patchRes.json()

    image.value = img
    patches.value = allPatches.slice(0, 4)
  } catch (err) {
    console.error(err)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.browse-page {
  padding: 2rem;
  font-family: sans-serif;
}

button {
  padding: 0.6rem 1.4rem;
  font-size: 1rem;
  cursor: pointer;
  margin-bottom: 2rem;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.result {
  display: flex;
  gap: 2rem;
  align-items: flex-start;
  flex-wrap: wrap;
}

.image-section h2 {
  margin: 0 0 0.5rem;
  font-size: 1rem;
  word-break: break-all;
}

.main-image {
  max-width: 400px;
  max-height: 400px;
  border: 1px solid #ccc;
  display: block;
}

.meta {
  font-size: 0.85rem;
  color: #666;
  margin-top: 0.4rem;
}

.patches-section h3 {
  margin: 0 0 0.75rem;
}

.patches-grid {
  display: grid;
  grid-template-columns: repeat(2, 128px);
  gap: 0.75rem;
}

.patch-card {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.patch-image {
  width: 128px;
  height: 128px;
  border: 1px solid #ccc;
  object-fit: cover;
}

.patch-meta {
  font-size: 0.75rem;
  color: #888;
  margin: 0.25rem 0 0;
}
</style>
