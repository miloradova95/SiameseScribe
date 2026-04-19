<template>
  <div class="upload-page">
    <h1>Upload Image</h1>

    <form class="upload-form" @submit.prevent="handleUpload">
      <div class="field">
        <label for="file-input">Image file</label>
        <input
          id="file-input"
          type="file"
          accept="image/*"
          @change="onFileChange"
        />
        <div v-if="localPreview" class="local-preview">
          <img :src="localPreview" alt="preview" />
        </div>
      </div>

      <div class="field">
        <label for="group-input">Group <span class="optional">(optional)</span></label>
        <input
          id="group-input"
          v-model="group"
          type="text"
          placeholder="e.g. cats"
        />
      </div>

      <button type="submit" :disabled="!selectedFile || uploading">
        {{ uploading ? 'Uploading…' : 'Upload' }}
      </button>
    </form>

    <div v-if="error" class="state error">{{ error }}</div>

    <section v-if="uploadedImage" class="result">
      <h2>Uploaded</h2>
      <ImageCard :image="uploadedImage" />
    </section>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ImageCard from '../components/ImageCard.vue'

const selectedFile = ref(null)
const localPreview = ref(null)
const group = ref('')
const uploading = ref(false)
const error = ref(null)
const uploadedImage = ref(null)

function onFileChange(e) {
  const file = e.target.files[0]
  if (!file) return
  selectedFile.value = file
  localPreview.value = URL.createObjectURL(file)
  uploadedImage.value = null
  error.value = null
}

async function handleUpload() {
  if (!selectedFile.value) return
  uploading.value = true
  error.value = null
  uploadedImage.value = null

  const form = new FormData()
  form.append('file', selectedFile.value)
  if (group.value.trim()) form.append('group', group.value.trim())

  try {
    const res = await fetch('http://localhost:8000/images/upload', {
      method: 'POST',
      body: form,
    })
    if (!res.ok) {
      const detail = await res.text()
      throw new Error(`Upload failed (${res.status}): ${detail}`)
    }
    uploadedImage.value = await res.json()
  } catch (e) {
    error.value = e.message
  } finally {
    uploading.value = false
  }
}
</script>

<style scoped>
.upload-page {
  padding: 24px;
  max-width: 480px;
}

.upload-form {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-top: 20px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

label {
  font-weight: 600;
  font-size: 0.9rem;
}

.optional {
  font-weight: 400;
  color: #888;
}

input[type='text'] {
  padding: 8px 10px;
  border: 1px solid #ccc;
  border-radius: 6px;
  font-size: 0.9rem;
}

.local-preview img {
  margin-top: 8px;
  max-width: 100%;
  max-height: 200px;
  border-radius: 6px;
  object-fit: cover;
  border: 1px solid #ddd;
}

button {
  align-self: flex-start;
  padding: 8px 20px;
  background: #2c3e50;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.state {
  margin-top: 16px;
  color: #888;
}

.error {
  color: #c0392b;
}

.result {
  margin-top: 28px;
}

.result h2 {
  margin-bottom: 12px;
}
</style>
