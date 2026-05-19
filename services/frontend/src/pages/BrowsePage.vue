<template>
  <div class="min-h-screen bg-[#f7f3ef] text-[#6c4f3d]">
    <main class="mx-auto max-w-[1860px] px-8 pb-10 pt-8">
      <section class="mt-6 border-b border-[#ddd3ca] pb-6">
        <div class="relative flex items-center justify-center">
          <div class="flex items-center gap-5 text-[18px]">
            <button class="text-[#c3b7ad] transition hover:text-[#8b6b59]">Pages</button>
            <span class="text-[#d0c5bc]">|</span>
            <button class="font-medium text-[#6c4f3d]">Pen Flourishes</button>
            <span class="text-[#d0c5bc]">|</span>
            <button class="text-[#c3b7ad] transition hover:text-[#8b6b59]">Patches</button>
          </div>

          <div class="absolute right-0 flex items-center gap-10 text-[17px]">
            <button class="transition hover:opacity-70">Filter</button>
            <button class="transition hover:opacity-70">Sorted by</button>

            <RandomImageButton
              :loading="loading"
              :disabled="images.length === 0"
              @click="loadRandom"
            />
          </div>
        </div>
      </section>

      <div
        v-if="error"
        class="mt-6 rounded-2xl border border-[#d6b7a8] bg-[#fff7f2] px-5 py-4 text-sm text-[#a25532]"
      >
        {{ error }}
      </div>

      <section class="mt-10 grid grid-cols-1 items-start gap-10 xl:grid-cols-[minmax(0,1fr)_420px]">
        <div>
          <BrowseGrid
            :images="paginatedImages"
            :selected-image-id="selectedImage?.id"
            :loading="initialLoading"
            :annotated-image-ids="annotatedImageIds"
            @select="selectImage"
          />
        </div>

        <BrowseInspector
          :image="selectedImage"
          :patches="patches"
          :patch-loading="patchLoading"
          @zoom="openZoom"
        />
      </section>
      <Pagination
        :current-page="currentPage"
        :total-pages="totalPages"
        :page-size="pageSize"
        @change="goToPage"
      />
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
              ✕
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
                  -
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
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import BrowseGrid from '@/features/browse/components/BrowseGrid.vue'
import BrowseInspector from '@/features/browse/BrowseInspector.vue'
import RandomImageButton from '@/components/RandomImageButton.vue'
import { apiUrl } from '@/lib/api'
import { fetchImages } from '@/services/image-service'
import { fetchMyFeedback, fetchPatchesByImageId } from '@/services/patch-service'
import Pagination from '@/features/browse/components/Pagination.vue'

const images = ref([])
const selectedImage = ref(null)
const patches = ref([])
const initialLoading = ref(false)
const loading = ref(false)
const patchLoading = ref(false)
const error = ref(null)
const annotatedImageIds = ref([])

const currentPage = ref(1)
const pageSize = ref(24)

const zoomOpen = ref(false)
const zoomLevel = ref(1)

let currentPatchRequestId = 0

const totalPages = computed(() => {
  return Math.max(1, Math.ceil(images.value.length / pageSize.value))
})

const paginatedImages = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return images.value.slice(start, start + pageSize.value)
})

async function goToPage(page) {
  currentPage.value = Math.min(Math.max(page, 1), totalPages.value)

  if (paginatedImages.value.length > 0) {
    await selectImage(paginatedImages.value[0])
  }
}

async function loadAllImages() {
  initialLoading.value = true
  error.value = null

  try {
    const data = await fetchImages()
    images.value = Array.isArray(data) ? data : []
    currentPage.value = 1

    if (paginatedImages.value.length > 0) {
      await selectImage(paginatedImages.value[0])
    }
  } catch (err) {
    error.value = err.message || 'Etwas ist schiefgelaufen.'
  } finally {
    initialLoading.value = false
  }
}

async function loadPatchesForImage(imageId) {
  const requestId = ++currentPatchRequestId
  patchLoading.value = true
  patches.value = []

  try {
    const data = await fetchPatchesByImageId(imageId)

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

async function loadAnnotatedImages() {
  try {
    const feedbackItems = await fetchMyFeedback()
    const ids = new Set()

    for (const item of Array.isArray(feedbackItems) ? feedbackItems : []) {
      if (item?.query_patch_source_image_id != null) {
        ids.add(String(item.query_patch_source_image_id))
      }
    }

    annotatedImageIds.value = [...ids]
  } catch (err) {
    console.error('Error loading annotated images:', err)
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
    let randomImage = null

    if (images.value.length === 1) {
      randomImage = images.value[0]
    } else {
      do {
        const randomIndex = Math.floor(Math.random() * images.value.length)
        randomImage = images.value[randomIndex]
      } while (randomImage.id === selectedImage.value?.id)
    }

    const index = images.value.findIndex((img) => img.id === randomImage.id)

    if (index !== -1) {
      currentPage.value = Math.floor(index / pageSize.value) + 1
    }

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
  loadAnnotatedImages()
  window.addEventListener('keydown', handleKeydown)
})

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>
