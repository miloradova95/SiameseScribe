<template>
  <div class="relative min-h-screen overflow-hidden bg-brand-bg font-body text-brand-text">
    <div
      class="pointer-events-none absolute inset-0 bg-cover bg-center opacity-[0.06]"
      style="background-image: url('/ornament-bg.png')"
    ></div>

    <main class="mx-auto flex max-w-6xl flex-col items-center px-6 pb-20 pt-6 text-center md:pt-10">
      <h1
        class="max-w-5xl font-display text-6xl leading-none tracking-tight text-brand-text md:text-8xl"
      >
        Upload Your Image
      </h1>

      <form
        class="mt-16 flex w-full max-w-3xl flex-col items-center"
        @submit.prevent="handleUpload"
      >
        <label
          for="file-input"
          class="flex w-full cursor-pointer flex-col items-center justify-center rounded-[28px] border-2 border-dashed border-brand-muted bg-white/20 px-8 py-16 transition hover:bg-white/30"
        >
          <input
            id="file-input"
            type="file"
            accept="image/*"
            @change="onFileChange"
            class="hidden"
          />

          <div class="mb-6 text-brand-accent">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="h-16 w-16"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="1.8"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M12 16V4m0 0l-4 4m4-4l4 4M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1"
              />
            </svg>
          </div>

          <p class="text-2xl font-semibold text-brand-text">
            <span class="text-brand-accent">Drag &amp; Drop</span> your image
          </p>

          <p class="mt-2 text-xl text-brand-text">
            or
            <span class="font-semibold text-brand-accent underline underline-offset-2">
              Browse
            </span>
            on your computer
          </p>

          <div v-if="localPreview" class="mt-8 w-full max-w-xl">
            <div class="overflow-hidden rounded-2xl border border-brand-line bg-white shadow-sm">
              <img :src="localPreview" alt="preview" class="max-h-[320px] w-full object-cover" />
            </div>
          </div>
        </label>

        <div class="mt-8 w-full max-w-md text-left">
          <label for="group-input" class="mb-2 block text-sm font-medium text-[#5a4a42]">
            Group <span class="text-[#9b8e86]">(optional)</span>
          </label>

          <input
            id="group-input"
            v-model="group"
            type="text"
            placeholder="e.g. cats"
            class="w-full rounded-full border border-brand-muted bg-brand-surface px-5 py-3 text-sm text-brand-text outline-none transition placeholder:text-[#aa9d95] focus:border-brand-accent focus:ring-2 focus:ring-brand-accent/10"
          />
        </div>

        <button
          type="submit"
          :disabled="!selectedFile || uploading"
          class="mt-10 inline-flex min-w-[160px] items-center justify-center rounded-full border border-brand-text bg-brand-bg px-8 py-3 text-xl font-medium text-brand-text transition hover:bg-brand-accent-soft disabled:cursor-not-allowed disabled:opacity-50"
        >
          {{ uploading ? 'Uploadingâ€¦' : 'Upload' }}
        </button>
      </form>

      <div
        v-if="error"
        class="mt-6 w-full max-w-2xl rounded-2xl border border-brand-line bg-brand-surface px-5 py-4 text-left text-sm text-brand-text shadow-sm"
      >
        {{ error }}
      </div>

      <section v-if="uploadedImage" class="mt-12 w-full max-w-3xl text-left">
        <h2 class="mb-4 text-2xl font-semibold text-brand-text">Uploaded</h2>
        <p class="mb-4 text-sm text-[#7b6e66]">
          {{ uploadedPatches.length }} patch{{ uploadedPatches.length === 1 ? '' : 'es' }} generated
        </p>
        <ImageCard :image="uploadedImage" />
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ImageCard from '../components/ImageCard.vue'
import { runUploadPipeline } from '../services/image-service'

const selectedFile = ref(null)
const localPreview = ref(null)
const group = ref('')
const uploading = ref(false)
const error = ref(null)
const uploadedImage = ref(null)
const uploadedPatches = ref([])

function onFileChange(e) {
  const file = e.target.files[0]
  if (!file) return
  selectedFile.value = file
  localPreview.value = URL.createObjectURL(file)
  uploadedImage.value = null
  uploadedPatches.value = []
  error.value = null
}

async function handleUpload() {
  if (!selectedFile.value) return
  uploading.value = true
  error.value = null
  uploadedImage.value = null
  uploadedPatches.value = []

  try {
    const result = await runUploadPipeline(selectedFile.value, group.value)
    uploadedImage.value = result.image
    uploadedPatches.value = result.patches
  } catch (e) {
    error.value = e.message
  } finally {
    uploading.value = false
  }
}
</script>
