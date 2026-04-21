<template>
  <div class="home">
    <h1>Home</h1>
    <section class="image-section">
      <h2>Sample Image</h2>
      <div v-if="loading" class="state">Loading...</div>
      <div v-else-if="error" class="state error">{{ error }}</div>
      <template v-else-if="image">
        <p class="filepath">{{ test }}</p>
        <ImageCard :image="image" />
      </template>
    </section>
    <section class="patch-section">
      <h2>Search Patch by File Name</h2>
      <div class="search-form">
        <input v-model="fileName" type="text" placeholder="Enter file name" />
        <button @click="searchPatch" :disabled="patchLoading">Search</button>
      </div>
      <div v-if="patchLoading" class="state">Searching...</div>
      <div v-else-if="patchError" class="state error">{{ patchError }}</div>
      <template v-else-if="patch">
        <PatchCard :patch="patch" />
      </template>
    </section>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import ImageCard from '../components/ImageCard.vue'
import PatchCard from '../components/PatchCard.vue'
import { apiUrl } from '../lib/api'

const image = ref(null)
const loading = ref(false)
const error = ref(null)
const test = ref("")

const fileName = ref("")
const patch = ref(null)
const patchLoading = ref(false)
const patchError = ref(null)

const counter = 1

onMounted(async () => {
  loading.value = true
  error.value = null
  try {
    const res = await fetch(apiUrl(`/images/${counter}`))
    if (!res.ok) throw new Error(`Image not found (${res.status})`)
    image.value = await res.json()
    test.value = image.value.filePath
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
})

const searchPatch = async () => {
  if (!fileName.value.trim()) return
  patchLoading.value = true
  patchError.value = null
  patch.value = null
  try {
    const res = await fetch(apiUrl(`/patches/by-file-name/${encodeURIComponent(fileName.value)}`))
    if (!res.ok) throw new Error(`Patch not found (${res.status})`)
    patch.value = await res.json()
  } catch (e) {
    patchError.value = e.message
  } finally {
    patchLoading.value = false
  }
}
</script>

<style scoped>
.home {
  padding: 24px;
}

.image-section {
  margin-top: 24px;
}

.patch-section {
  margin-top: 48px;
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

.search-form {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

input {
  flex: 1;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

button {
  padding: 8px 16px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

button:disabled {
  background: #ccc;
  cursor: not-allowed;
}
</style>
