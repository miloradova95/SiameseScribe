<template>
  <div class="min-h-screen bg-[#f7f3ef] text-[#6c4f3d]">
    <main class="mx-auto max-w-[1860px] px-8 pb-10 pt-8">

      <section class="mt-28 border-b border-[#ddd3ca] pb-6">
        <div class="relative flex items-center justify-center">
          <div class="flex items-center gap-5 text-[18px]">
            <button class="text-[#c3b7ad] transition hover:text-[#8b6b59]">
              Pages
            </button>
            <span class="text-[#d0c5bc]">|</span>
            <button class="font-medium text-[#6c4f3d]">
              Pen Flourishes
            </button>
            <span class="text-[#d0c5bc]">|</span>
            <button class="text-[#c3b7ad] transition hover:text-[#8b6b59]">
              Patches
            </button>
          </div>

          <div class="absolute right-0 flex items-center gap-10 text-[17px] text-[#6c4f3d]">
            <button class="transition hover:opacity-70">
              Filter
            </button>
            <button class="transition hover:opacity-70">
              Sorted by
            </button>
            <button
              :disabled="loading || images.length === 0"
              class="transition hover:opacity-70 disabled:cursor-not-allowed disabled:opacity-40"
              @click="loadRandom"
            >
              {{ loading ? 'Loading...' : 'Random' }}
            </button>
          </div>
        </div>
      </section>

      <div
        v-if="error"
        class="mt-6 rounded-2xl border border-[#d6b7a8] bg-[#fff7f2] px-5 py-4 text-sm text-[#a25532]"
      >
        {{ error }}
      </div>

      <section class="mt-10 grid grid-cols-1 gap-10 xl:grid-cols-[minmax(0,1fr)_420px]">
        <BrowseGrid
          :images="images"
          :selected-image-id="selectedImage?.id"
          :loading="initialLoading"
          @select="selectImage"
        />

        <BrowseInspector
          :image="selectedImage"
          :patches="patches"
          :patch-loading="patchLoading"
          @zoom="openZoom"
        />
      </section>
    </main>

      <Teleport to="body">
      <transition
        enter-active-class="transition duration-200 ease-out"
        enter-from-class="opacity-0"
        enter-to-class="opacity-100"
        leave-active-class="transition duration-150 ease-in"
        leave-from-class="opacity-100"
        leave-to-class="opacity-0"
      >
        <div
          v-if="zoomOpen && selectedImage"
          class="fixed inset-0 z-[100] bg-black/55 backdrop-blur-[2px]"
          @click.self="closeZoom"
        >
          <div class="relative flex h-full w-full items-center justify-center px-6 py-10">
            <button
              type="button"
              class="absolute right-6 top-6 text-white/90 transition hover:opacity-70"
              @click="closeZoom"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                class="h-10 w-10"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                stroke-width="1.8"
              >
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M6 6l12 12M18 6L6 18"
                />
              </svg>
            </button>

            <div class="relative flex max-h-full max-w-full items-center justify-center">
              <img
                :src="apiUrl(`/images/${selectedImage.id}/file`)"
                :alt="selectedImage.fileName"
                class="max-h-[82vh] max-w-[80vw] rounded-[10px] object-contain shadow-2xl transition-transform duration-150"
                :style="{ transform: `scale(${zoomLevel})` }"
              />

              <div class="absolute bottom-4 right-[-60px] flex flex-col gap-2">
                <button
                  type="button"
                  class="flex h-10 w-10 items-center justify-center rounded-lg bg-white/90 text-[#5a4032] shadow transition hover:bg-white"
                  @click.stop="zoomIn"
                >
                  +
                </button>

                <button
                  type="button"
                  class="flex h-10 w-10 items-center justify-center rounded-lg bg-white/90 text-[#5a4032] shadow transition hover:bg-white"
                  @click.stop="zoomOut"
                >
                  −
                </button>
              </div>
            </div>
          </div>
        </div>
      </transition>
    </Teleport>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue'
import BrowseGrid from '@/features/browse/BrowseGrid.vue'
import BrowseInspector from '@/features/browse/BrowseInspector.vue'
import { apiUrl } from '@/lib/api'

const images = ref([])
const selectedImage = ref(null)
const patches = ref([])
const initialLoading = ref(false)
const loading = ref(false)
const patchLoading = ref(false)
const error = ref(null)

const zoomOpen = ref(false)
const zoomLevel = ref(1)

let currentPatchRequestId = 0

async function loadAllImages() {
  initialLoading.value = true
  error.value = null

  try {
    const res = await fetch(apiUrl('/images'))
    if (!res.ok) throw new Error('Failed to fetch images.')

    const data = await res.json()
    images.value = Array.isArray(data) ? data : []

    if (images.value.length > 0) {
      await selectImage(images.value[0])
    }
  } catch (err) {
    error.value = err.message || 'Something went wrong.'
  } finally {
    initialLoading.value = false
  }
}

async function loadPatchesForImage(imageId) {
  const requestId = ++currentPatchRequestId
  patchLoading.value = true
  patches.value = []

  try {
    const res = await fetch(apiUrl(`/images/${imageId}/patches`))
    if (!res.ok) throw new Error('Failed to fetch patches.')

    const data = await res.json()

    if (requestId !== currentPatchRequestId) return
    patches.value = Array.isArray(data) ? data.slice(0, 4) : []
  } catch (err) {
    if (requestId !== currentPatchRequestId) return
    error.value = err.message || 'Failed to fetch patches.'
  } finally {
    if (requestId === currentPatchRequestId) {
      patchLoading.value = false
    }
  }
}

async function selectImage(img) {
  if (!img?.id) return
  selectedImage.value = img
  error.value = null

  await loadPatchesForImage(img.id)
}

async function loadRandom() {
  if (images.value.length === 0) return

  loading.value = true
  error.value = null

  try {
    if (images.value.length === 1) {
      await selectImage(images.value[0])
      return
    }

    let randomImage = null

    do {
      const randomIndex = Math.floor(Math.random() * images.value.length)
      randomImage = images.value[randomIndex]
    } while (randomImage.id === selectedImage.value?.id)

    await selectImage(randomImage)
  } catch (err) {
    error.value = err.message || 'Failed to load random image.'
  } finally {
    loading.value = false
  }
}

function openZoom() {
  if (!selectedImage.value) return
  zoomLevel.value = 1
  zoomOpen.value = true
}

function closeZoom() {
  zoomOpen.value = false
}

function zoomIn() {
  zoomLevel.value = Math.min(zoomLevel.value + 0.2, 3)
}

function zoomOut() {
  zoomLevel.value = Math.max(zoomLevel.value - 0.2, 0.6)
}

function handleKeydown(event) {
  if (event.key === 'Escape') {
    closeZoom()
  }
}

onMounted(() => {
  loadAllImages()
  window.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>