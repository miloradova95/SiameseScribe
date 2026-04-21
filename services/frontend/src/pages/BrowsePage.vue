<template>
<div class="page-shell flex flex-col items-center">
  <main class="page-main !max-w-[1400px] !items-center text-center">
        <section class="mb-12 w-full px-6 md:px-12">
  <div class="flex w-full flex-col gap-6 md:flex-row md:items-center md:justify-between">
    
    <!-- LEFT -->
    <div class="text-left">
      <h1 class="page-title !max-w-none !text-4xl md:!text-5xl">
        Overview of all Uploads
      </h1>
      <p class="mt-2 text-sm text-brand-subtle">
        Browse all uploaded images and inspect patches for the selected item.
      </p>
    </div>

    <!-- RIGHT -->
    <button
      @click="loadRandom"
      :disabled="loading || images.length === 0"
      class="btn-primary-outline whitespace-nowrap self-start md:ml-auto"
    >
      {{ loading ? 'Loading...' : 'Random Image + Patches' }}
    </button>

  </div>
</section>

      <div v-if="error" class="error-box !mt-0 mb-6">
        {{ error }}
      </div>

      <section class="grid grid-cols-1 gap-8 xl:grid-cols-[minmax(0,1fr)_340px]">
        <BrowseGrid
          :images="images"
          :selected-image-id="selectedImage?.id"
          :loading="initialLoading"
          @select="selectImage"
        />

        <BrowseSidebar
          :image="selectedImage"
          :patches="patches"
          :patch-loading="patchLoading"
        />
      </section>
    </main>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import BrowseGrid from '@/components/browse/BrowseGrid.vue'
import BrowseSidebar from '@/components/browse/BrowseSidebar.vue'

const images = ref([])
const selectedImage = ref(null)
const patches = ref([])
const initialLoading = ref(false)
const loading = ref(false)
const patchLoading = ref(false)
const error = ref(null)

async function loadAllImages() {
  initialLoading.value = true
  error.value = null

  try {
    const res = await fetch('http://localhost:8000/images')
    if (!res.ok) throw new Error('Failed to fetch images')

    const data = await res.json()
    images.value = Array.isArray(data) ? data : []

    if (images.value.length > 0) {
      await selectImage(images.value[0])
    }
  } catch (err) {
    error.value = err.message || 'Etwas ist schiefgelaufen.'
  } finally {
    initialLoading.value = false
  }
}

async function loadPatchesForImage(imageId) {
  patchLoading.value = true
  patches.value = []

  try {
    const res = await fetch(`http://localhost:8000/images/${imageId}/patches`)
    if (!res.ok) throw new Error('Failed to fetch patches')

    const data = await res.json()
    patches.value = Array.isArray(data) ? data.slice(0, 4) : []
  } catch (err) {
    error.value = err.message || 'Failed to fetch patches'
  } finally {
    patchLoading.value = false
  }
}

async function selectImage(img) {
  selectedImage.value = img
  error.value = null
  await loadPatchesForImage(img.id)
}

async function loadRandom() {
  if (!images.value.length) return

  loading.value = true
  error.value = null

  try {
    const randomIndex = Math.floor(Math.random() * images.value.length)
    const randomImage = images.value[randomIndex]
    await selectImage(randomImage)
  } catch (err) {
    error.value = err.message || 'Failed to load random image'
  } finally {
    loading.value = false
  }
}

onMounted(loadAllImages)
</script>